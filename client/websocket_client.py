"""
WebSocket client for the voice assistant.

This module provides a WebSocket client for communication with the server.
"""
import asyncio
import base64
import json
import logging
import numpy as np
import time
from typing import Dict, Any, Optional, Callable, List
import uuid

import sounddevice as sd
import soundfile as sf
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
        self.is_recording = False
        self.recording_task = None
        self.conversation_id = None
        self.input_stream = None
        self.output_stream = None
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.dtype = 'int16'  # Equivalent to pyaudio.paInt16
        
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
        if self.input_stream is not None:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
            
        # Send end stream message
        await self.send_message("end_stream", {"conversation_id": self.conversation_id})
        
        logger.info("Stopped audio recording")

    async def _record_audio(self):
        """
        Record audio and send to server using SoundDevice.
        """
        try:
            # Define callback function to process audio data
            queue = asyncio.Queue()
            
            def audio_callback(indata, frames, time, status):
                """Callback function for the input stream"""
                if status:
                    logger.warning(f"Stream status: {status}")
                # Convert the NumPy array to bytes
                audio_data = indata.tobytes()
                # Put the data in the queue
                asyncio.run_coroutine_threadsafe(queue.put(audio_data), asyncio.get_event_loop())
            
            # Open the input stream
            self.input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.chunk_size,
                callback=audio_callback
            )
            
            # Start the stream
            self.input_stream.start()
            
            # Process audio data from the queue
            while self.is_recording:
                try:
                    # Get audio data with timeout
                    audio_data = await asyncio.wait_for(queue.get(), 1.0)
                    
                    # Encode as base64
                    audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                    
                    # Send to server
                    payload = {
                        "audio": audio_b64,
                        "conversation_id": self.conversation_id,
                        "is_final": False
                    }
                    
                    await self.send_message("audio", payload)
                    
                except asyncio.TimeoutError:
                    # Timeout is expected when stopping recording
                    continue
                except Exception as e:
                    logger.exception(f"Error processing audio data: {e}")
                    break
            
            # Get any remaining data in the queue for final chunk
            try:
                # Try to get one more chunk if available
                if not queue.empty():
                    audio_data = queue.get_nowait()
                    audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                    
                    payload = {
                        "audio": audio_b64,
                        "conversation_id": self.conversation_id,
                        "is_final": True
                    }
                    
                    await self.send_message("audio", payload)
            except asyncio.QueueEmpty:
                pass
            
        except Exception as e:
            logger.exception(f"Error during audio recording: {e}")
            self.is_recording = False
        finally:
            # Ensure stream is stopped and closed
            if self.input_stream is not None:
                self.input_stream.stop()
                self.input_stream.close()
                self.input_stream = None

    async def play_audio(self, audio_data_b64: str):
        """
        Play audio received from server.
        
        Args:
            audio_data_b64: Base64-encoded audio data
        """
        try:
            # Decode base64
            audio_data = base64.b64decode(audio_data_b64)
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=self.dtype)
            
            # Play audio using SoundDevice (blocks until audio finishes playing)
            # Use a separate thread to avoid blocking the asyncio event loop
            await asyncio.to_thread(
                sd.play,
                audio_array,
                samplerate=self.sample_rate,
                blocking=True
            )
            
        except Exception as e:
            logger.exception(f"Error playing audio: {e}")

    def close(self):
        """
        Close and clean up resources.
        """
        # Stop and close any open streams
        if self.input_stream is not None:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
            
        if self.output_stream is not None:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
            
        # Reset state
        self.is_recording = False