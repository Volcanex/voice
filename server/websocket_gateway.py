"""
WebSocket Gateway for the voice assistant.

Handles WebSocket connections and routes messages between clients and modules.
"""
import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Set, Callable, Awaitable

import fastapi
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .state_manager import StateManager
from .modules.asr_module import ASRModule
from .modules.llm_module import LLMModule
from .modules.csm_module import CSMModule

logger = logging.getLogger(__name__)

# Message models
class ClientMessage(BaseModel):
    """Base model for messages from clients."""
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)

class ServerMessage(BaseModel):
    """Base model for messages from server to clients."""
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        # Use model_dump_json to avoid deprecation warning in Pydantic v2
        if hasattr(self, 'model_dump_json'):
            return self.model_dump_json()
        else:
            # Fallback for older Pydantic versions
            return self.json()

class WebSocketManager:
    """
    Manages WebSocket connections and message routing.
    """
    def __init__(
        self, 
        state_manager: StateManager,
        asr_module: ASRModule,
        llm_module: LLMModule,
        csm_module: CSMModule
    ):
        """
        Initialize the WebSocket manager.
        
        Args:
            state_manager: State manager instance
            asr_module: ASR module instance
            llm_module: LLM module instance
            csm_module: CSM module instance
        """
        self.state_manager = state_manager
        self.asr_module = asr_module
        self.llm_module = llm_module
        self.csm_module = csm_module
        
        self.active_connections: Dict[str, WebSocket] = {}
        self.handlers: Dict[str, Callable[[str, ClientMessage], Awaitable[None]]] = {
            "init": self.handle_init,
            "audio": self.handle_audio,
            "text": self.handle_text,
            "end_stream": self.handle_end_stream,
            "clear_history": self.handle_clear_history,
            "set_preference": self.handle_set_preference
        }
        
        logger.info("WebSocket Manager initialized")

    async def handle_connection(self, websocket: WebSocket) -> None:
        """
        Handle a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
        """
        # Accept the connection
        await websocket.accept()
        
        # Generate a session ID
        session_id = str(uuid.uuid4())
        self.active_connections[session_id] = websocket
        
        # Create a conversation
        conversation_id = self.state_manager.create_conversation(session_id)
        
        logger.info(f"New WebSocket connection: {session_id}")
        
        # Send welcome message
        welcome_message = ServerMessage(
            type="init",
            payload={
                "session_id": session_id,
                "conversation_id": conversation_id,
                "message": "Connected to voice assistant server"
            }
        )
        await websocket.send_text(welcome_message.to_json())
        
        try:
            # Message handling loop
            async for message in websocket.iter_text():
                try:
                    client_message = ClientMessage.parse_raw(message)
                    message_type = client_message.type
                    
                    # Route to appropriate handler
                    if message_type in self.handlers:
                        await self.handlers[message_type](session_id, client_message)
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                        error_message = ServerMessage(
                            type="error",
                            payload={"error": f"Unknown message type: {message_type}"}
                        )
                        await websocket.send_text(error_message.to_json())
                        
                except json.JSONDecodeError:
                    logger.error("Failed to parse client message")
                    error_message = ServerMessage(
                        type="error",
                        payload={"error": "Invalid JSON format"}
                    )
                    await websocket.send_text(error_message.to_json())
                    
                except Exception as e:
                    logger.exception("Error handling client message")
                    error_message = ServerMessage(
                        type="error",
                        payload={"error": str(e)}
                    )
                    await websocket.send_text(error_message.to_json())
                    
        except WebSocketDisconnect:
            # Clean up on disconnect
            self.active_connections.pop(session_id, None)
            self.state_manager.remove_session(session_id)
            logger.info(f"WebSocket disconnected: {session_id}")
            
        except Exception as e:
            # Handle other exceptions
            logger.exception(f"WebSocket error: {e}")
            self.active_connections.pop(session_id, None)
            self.state_manager.remove_session(session_id)

    async def handle_init(self, session_id: str, message: ClientMessage) -> None:
        """
        Handle initialization messages from clients.
        
        Args:
            session_id: Client session ID
            message: Client message
        """
        websocket = self.active_connections.get(session_id)
        if not websocket:
            logger.warning(f"Websocket not found for session {session_id}")
            return
            
        # Initialize modules if needed
        await self.asr_module.initialize()
        await self.llm_module.initialize()
        await self.csm_module.initialize()
        
        # Send ready message
        ready_message = ServerMessage(
            type="ready",
            payload={"message": "Voice assistant ready"}
        )
        await websocket.send_text(ready_message.to_json())

    async def handle_audio(self, session_id: str, message: ClientMessage) -> None:
        """
        Handle audio messages from clients.
        
        Args:
            session_id: Client session ID
            message: Client message with audio data
        """
        websocket = self.active_connections.get(session_id)
        if not websocket:
            logger.warning(f"Websocket not found for session {session_id}")
            return
            
        # Get audio data
        audio_data = message.payload.get("audio")
        conversation_id = message.payload.get("conversation_id")
        is_final = message.payload.get("is_final", False)
        
        if not audio_data or not conversation_id:
            logger.warning("Missing audio data or conversation ID")
            error_message = ServerMessage(
                type="error",
                payload={"error": "Missing audio data or conversation ID"}
            )
            await websocket.send_text(error_message.to_json())
            return
            
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Process audio with ASR
            transcription = await self.asr_module.process_audio_stream(
                audio_bytes, is_final
            )
            
            if transcription:
                # Send transcription to client
                transcription_message = ServerMessage(
                    type="transcription",
                    payload={
                        "text": transcription,
                        "is_final": is_final
                    }
                )
                await websocket.send_text(transcription_message.to_json())
                
                # If final chunk, add to conversation and generate response
                if is_final:
                    # Add user message to conversation
                    self.state_manager.add_message(
                        conversation_id, "user", transcription
                    )
                    
                    # Get conversation history
                    history = self.state_manager.get_conversation_history(conversation_id)
                    system_prompt = self.state_manager.get_system_prompt()
                    
                    # Start streaming response generation
                    response_text = ""
                    async for token in self.llm_module.generate_response_stream(
                        history, system_prompt
                    ):
                        # Accumulate response
                        response_text += token
                        
                        # Send token to client
                        token_message = ServerMessage(
                            type="response_token",
                            payload={"token": token}
                        )
                        await websocket.send_text(token_message.to_json())
                        
                    # Add assistant message to conversation
                    self.state_manager.add_message(
                        conversation_id, "assistant", response_text
                    )
                    
                    # Generate speech for response
                    async for audio_chunk in self.csm_module.synthesize_speech_stream(
                        response_text, history
                    ):
                        # Encode audio to base64
                        audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
                        
                        # Send audio chunk to client
                        audio_message = ServerMessage(
                            type="audio",
                            payload={
                                "audio": audio_b64,
                                "is_final": False
                            }
                        )
                        await websocket.send_text(audio_message.to_json())
                        
                    # Send final message
                    final_message = ServerMessage(
                        type="response_complete",
                        payload={
                            "text": response_text
                        }
                    )
                    await websocket.send_text(final_message.to_json())
                    
        except Exception as e:
            logger.exception(f"Error processing audio: {e}")
            error_message = ServerMessage(
                type="error",
                payload={"error": str(e)}
            )
            await websocket.send_text(error_message.to_json())

    async def handle_text(self, session_id: str, message: ClientMessage) -> None:
        """
        Handle text messages from clients.
        
        Args:
            session_id: Client session ID
            message: Client message with text data
        """
        websocket = self.active_connections.get(session_id)
        if not websocket:
            logger.warning(f"Websocket not found for session {session_id}")
            return
            
        # Get text data
        text = message.payload.get("text")
        conversation_id = message.payload.get("conversation_id")
        
        if not text or not conversation_id:
            logger.warning("Missing text or conversation ID")
            error_message = ServerMessage(
                type="error",
                payload={"error": "Missing text or conversation ID"}
            )
            await websocket.send_text(error_message.to_json())
            return
            
        try:
            # Add user message to conversation
            self.state_manager.add_message(conversation_id, "user", text)
            
            # Get conversation history
            history = self.state_manager.get_conversation_history(conversation_id)
            system_prompt = self.state_manager.get_system_prompt()
            
            # Start streaming response generation
            response_text = ""
            async for token in self.llm_module.generate_response_stream(
                history, system_prompt
            ):
                # Accumulate response
                response_text += token
                
                # Send token to client
                token_message = ServerMessage(
                    type="response_token",
                    payload={"token": token}
                )
                await websocket.send_text(token_message.to_json())
                
            # Add assistant message to conversation
            self.state_manager.add_message(conversation_id, "assistant", response_text)
            
            # Generate speech for response
            async for audio_chunk in self.csm_module.synthesize_speech_stream(
                response_text, history
            ):
                # Encode audio to base64
                audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
                
                # Send audio chunk to client
                audio_message = ServerMessage(
                    type="audio",
                    payload={
                        "audio": audio_b64,
                        "is_final": False
                    }
                )
                await websocket.send_text(audio_message.to_json())
                
            # Send final message
            final_message = ServerMessage(
                type="response_complete",
                payload={
                    "text": response_text
                }
            )
            await websocket.send_text(final_message.to_json())
            
        except Exception as e:
            logger.exception(f"Error processing text: {e}")
            error_message = ServerMessage(
                type="error",
                payload={"error": str(e)}
            )
            await websocket.send_text(error_message.to_json())

    async def handle_end_stream(self, session_id: str, message: ClientMessage) -> None:
        """
        Handle end_stream messages from clients.
        
        Args:
            session_id: Client session ID
            message: Client message
        """
        # Reset streaming state in ASR module
        self.asr_module.reset_stream()
        
        websocket = self.active_connections.get(session_id)
        if not websocket:
            logger.warning(f"Websocket not found for session {session_id}")
            return
            
        # Send acknowledgement
        ack_message = ServerMessage(
            type="stream_ended",
            payload={"message": "Audio stream ended"}
        )
        await websocket.send_text(ack_message.to_json())

    async def handle_clear_history(
        self, session_id: str, message: ClientMessage
    ) -> None:
        """
        Handle clear_history messages from clients.
        
        Args:
            session_id: Client session ID
            message: Client message
        """
        websocket = self.active_connections.get(session_id)
        if not websocket:
            logger.warning(f"Websocket not found for session {session_id}")
            return
            
        conversation_id = message.payload.get("conversation_id")
        if not conversation_id:
            logger.warning("Missing conversation ID")
            error_message = ServerMessage(
                type="error",
                payload={"error": "Missing conversation ID"}
            )
            await websocket.send_text(error_message.to_json())
            return
            
        # Clear conversation history
        self.state_manager.clear_conversation(conversation_id)
        
        # Send acknowledgement
        ack_message = ServerMessage(
            type="history_cleared",
            payload={"message": "Conversation history cleared"}
        )
        await websocket.send_text(ack_message.to_json())

    async def handle_set_preference(
        self, session_id: str, message: ClientMessage
    ) -> None:
        """
        Handle set_preference messages from clients.
        
        Args:
            session_id: Client session ID
            message: Client message with preference data
        """
        websocket = self.active_connections.get(session_id)
        if not websocket:
            logger.warning(f"Websocket not found for session {session_id}")
            return
            
        preference_key = message.payload.get("key")
        preference_value = message.payload.get("value")
        
        if not preference_key:
            logger.warning("Missing preference key")
            error_message = ServerMessage(
                type="error",
                payload={"error": "Missing preference key"}
            )
            await websocket.send_text(error_message.to_json())
            return
            
        # Set user preference
        self.state_manager.set_user_preference(
            session_id, preference_key, preference_value
        )
        
        # Send acknowledgement
        ack_message = ServerMessage(
            type="preference_set",
            payload={
                "key": preference_key,
                "value": preference_value
            }
        )
        await websocket.send_text(ack_message.to_json())

    async def broadcast_message(self, message: ServerMessage) -> None:
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        json_message = message.to_json()
        disconnected_sessions = []
        
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json_message)
            except Exception:
                logger.exception(f"Failed to send message to {session_id}")
                disconnected_sessions.append(session_id)
                
        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            self.active_connections.pop(session_id, None)
            self.state_manager.remove_session(session_id)

    async def shutdown(self) -> None:
        """
        Clean up resources on shutdown.
        """
        # Close all connections
        for session_id, websocket in list(self.active_connections.items()):
            try:
                await websocket.close()
            except Exception:
                pass
                
        self.active_connections.clear()
        
        # Shut down modules
        await self.asr_module.shutdown()
        await self.llm_module.shutdown()
        await self.csm_module.shutdown()