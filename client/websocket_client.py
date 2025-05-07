"""
WebSocket client for the voice assistant.

This module provides a WebSocket client for communication with the server.
"""
import asyncio
import base64
import json
import logging
import time
from typing import Dict, Any, Optional, Callable, List
import uuid

import pyaudio
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

logger = logging.getLogger(__name__)

class WebSocketClient:
    """
    WebSocket client for communication with the voice assistant server.
    """
    def __init__(self, server_url: str):
        """
        Initialize the WebSocket client.
        
        Args:
            server_url: WebSocket server URL
        """
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
        
        # Message handlers
        self.on_message: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
        
        # Audio recording
        self.audio = None
        self.audio_stream = None
        self.is_recording = False
        self.recording_task = None
        self.conversation_id = None
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        
        logger.info("WebSocket client initialized")

    async def connect(self):
        """
        Connect to the WebSocket server.
        """
        if self.is_connected:
            return
            
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            
            logger.info(f"Connected to WebSocket server at {self.server_url}")
            
            # Start message handling loop
            asyncio.create_task(self._handle_messages())
            
            # Call connect handler
            if self.on_connect:
                await self.on_connect()
                
        except Exception as e:
            logger.exception(f"Failed to connect: {e}")
            self.is_connected = False
            self.websocket = None
            raise

    async def disconnect(self):
        """
        Disconnect from the WebSocket server.
        """
        if not self.is_connected or not self.websocket:
            return
            
        # Stop recording if active
        if self.is_recording:
            await self.stop_recording()
            
        try:
            await self.websocket.close()
        except Exception as e:
            logger.exception(f"Error during disconnect: {e}")
            
        self.is_connected = False
        self.websocket = None
        
        logger.info("Disconnected from WebSocket server")
        
        # Call disconnect handler
        if self.on_disconnect:
            await self.on_disconnect()

    async def _handle_messages(self):
        """
        Handle incoming messages from the server.
        """
        if not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Call message handler
                    if self.on_message:
                        await self.on_message(data)
                        
                except json.JSONDecodeError:
                    logger.error("Failed to parse server message")
                    
                except Exception as e:
                    logger.exception(f"Error handling message: {e}")
                    
        except ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.is_connected = False
            
            # Call disconnect handler
            if self.on_disconnect:
                await self.on_disconnect()
                
        except Exception as e:
            logger.exception(f"WebSocket error: {e}")
            self.is_connected = False
            
            # Call disconnect handler
            if self.on_disconnect:
                await self.on_disconnect()

    async def send_message(self, message_type: str, payload: Dict[str, Any] = None):
        """
        Send a message to the server.
        
        Args:
            message_type: Message type
            payload: Message payload
        """
        if not self.is_connected or not self.websocket:
            logger.warning("Cannot send message: not connected")
            return
            
        if payload is None:
            payload = {}
            
        message = {
            "type": message_type,
            "payload": payload
        }
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.exception(f"Error sending message: {e}")
            await self.disconnect()

    async def send_init(self):
        """
        Send initialization message to the server.
        """
        await self.send_message("init")

    async def send_text(self, text: str, conversation_id: str):
        """
        Send text message to the server.
        
        Args:
            text: Text message
            conversation_id: Conversation ID
        """
        payload = {
            "text": text,
            "conversation_id": conversation_id
        }
        
        await self.send_message("text", payload)

    async def clear_history(self, conversation_id: str):
        """
        Send clear history message to the server.
        
        Args:
            conversation_id: Conversation ID
        """
        payload = {
            "conversation_id": conversation_id
        }
        
        await self.send_message("clear_history", payload)

    async def start_recording(self, conversation_id: str):
        """
        Start recording audio and sending to server.
        
        Args:
            conversation_id: Conversation ID
        """
        if self.is_recording:
            return
            
        self.conversation_id = conversation_id
        self.is_recording = True
        
        # Initialize PyAudio
        if not self.audio:
            self.audio = pyaudio.PyAudio()
            
        # Start recording task
        self.recording_task = asyncio.create_task(self._record_audio())
        
        logger.info("Started audio recording")

    async def stop_recording(self):
        """
        Stop recording audio.
        """
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # Wait for recording task to finish
        if self.recording_task:
            try:
                await self.recording_task
            except asyncio.CancelledError:
                pass
                
            self.recording_task = None
            
        # Stop audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
            
        # Send end stream message
        await self.send_message("end_stream", {"conversation_id": self.conversation_id})
        
        logger.info("Stopped audio recording")

    async def _record_audio(self):
        """
        Record audio and send to server.
        """
        try:
            # Create audio stream
            self.audio_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Start recording
            while self.is_recording:
                # Read audio chunk
                audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Encode as base64
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                
                # Send to server
                payload = {
                    "audio": audio_b64,
                    "conversation_id": self.conversation_id,
                    "is_final": False
                }
                
                await self.send_message("audio", payload)
                
                # Short delay to prevent overwhelming the server
                await asyncio.sleep(0.01)
                
            # Send final chunk
            audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
            audio_b64 = base64.b64encode(audio_data).decode("utf-8")
            
            payload = {
                "audio": audio_b64,
                "conversation_id": self.conversation_id,
                "is_final": True
            }
            
            await self.send_message("audio", payload)
            
        except Exception as e:
            logger.exception(f"Error during audio recording: {e}")
            self.is_recording = False

    async def play_audio(self, audio_data_b64: str):
        """
        Play audio received from server.
        
        Args:
            audio_data_b64: Base64-encoded audio data
        """
        try:
            # Decode base64
            audio_data = base64.b64decode(audio_data_b64)
            
            # Initialize PyAudio if needed
            if not self.audio:
                self.audio = pyaudio.PyAudio()
                
            # Create temporary output stream
            output_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
            
            # Play audio
            output_stream.write(audio_data)
            
            # Close stream
            output_stream.stop_stream()
            output_stream.close()
            
        except Exception as e:
            logger.exception(f"Error playing audio: {e}")

    def close(self):
        """
        Close and clean up resources.
        """
        # Close PyAudio
        if self.audio:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                
            self.audio.terminate()
            self.audio = None
            self.audio_stream = None