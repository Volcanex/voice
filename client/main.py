"""
Main client application for the modular voice assistant.

This module sets up the TKinter UI and WebSocket client.
"""
import asyncio
import os
import logging
import sys
import tkinter as tk
from typing import Dict, Any, Optional, Callable

from .ui import VoiceAssistantUI
from .websocket_client import WebSocketClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("voice_assistant_client.log")
    ]
)

logger = logging.getLogger(__name__)

class VoiceAssistantApp:
    """
    Main voice assistant client application.
    """
    def __init__(self, server_url: str = "ws://localhost:8000/ws"):
        """
        Initialize the voice assistant client app.
        
        Args:
            server_url: WebSocket server URL
        """
        self.server_url = server_url
        self.ui = None
        self.ws_client = None
        self.session_id = None
        self.conversation_id = None
        
        # Application state
        self.is_connected = False
        self.is_recording = False
        
        logger.info("Voice Assistant client initialized")

    async def run(self):
        """
        Run the voice assistant client.
        """
        # Create the UI
        root = tk.Tk()
        root.title("Voice Assistant")
        
        # Create the WebSocket client
        self.ws_client = WebSocketClient(self.server_url)
        
        # Create the UI
        self.ui = VoiceAssistantUI(
            root,
            on_connect=self.handle_connect,
            on_send_text=self.handle_send_text,
            on_start_recording=self.handle_start_recording,
            on_stop_recording=self.handle_stop_recording,
            on_clear_history=self.handle_clear_history
        )
        
        # Set up event handlers
        self.ws_client.on_message = self.handle_server_message
        self.ws_client.on_connect = self.handle_server_connect
        self.ws_client.on_disconnect = self.handle_server_disconnect
        
        # Start the UI
        try:
            # Start the asyncio event loop
            while True:
                # Update the UI
                self.ui.update()
                
                # Run asyncio tasks
                await asyncio.sleep(0.01)
                
        except KeyboardInterrupt:
            # Handle graceful shutdown
            await self.shutdown()
            
        except tk.TclError:
            # Window was closed
            await self.shutdown()
            
        except Exception as e:
            logger.exception(f"Error in client: {e}")
            await self.shutdown()

    async def shutdown(self):
        """
        Clean up resources on shutdown.
        """
        logger.info("Shutting down client")
        
        # Close WebSocket connection
        if self.ws_client:
            await self.ws_client.disconnect()
            
        # Close UI
        if self.ui and self.ui.root.winfo_exists():
            self.ui.root.destroy()
            
        logger.info("Client shutdown complete")

    async def handle_connect(self):
        """
        Handle connect button click.
        """
        if self.is_connected:
            await self.ws_client.disconnect()
            self.is_connected = False
            self.ui.set_connected(False)
            self.ui.add_message("System", "Disconnected from server")
            return
            
        # Connect to server
        try:
            await self.ws_client.connect()
            self.ui.set_connecting(True)
            self.ui.add_message("System", "Connecting to server...")
        except Exception as e:
            logger.exception(f"Connection error: {e}")
            self.ui.add_message("Error", f"Failed to connect: {e}")
            self.ui.set_connecting(False)
            self.ui.set_connected(False)

    async def handle_send_text(self, text: str):
        """
        Handle send text button click.
        
        Args:
            text: Text to send
        """
        if not self.is_connected or not self.conversation_id:
            self.ui.add_message("Error", "Not connected to server")
            return
            
        # Add to UI
        self.ui.add_message("You", text)
        self.ui.clear_input()
        
        # Send to server
        await self.ws_client.send_text(text, self.conversation_id)

    async def handle_start_recording(self):
        """
        Handle start recording button click.
        """
        if not self.is_connected:
            self.ui.add_message("Error", "Not connected to server")
            return
            
        if self.is_recording:
            return
            
        self.is_recording = True
        self.ui.set_recording(True)
        self.ui.add_message("System", "Recording started...")
        
        # Start audio capture
        await self.ws_client.start_recording(self.conversation_id)

    async def handle_stop_recording(self):
        """
        Handle stop recording button click.
        """
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.ui.set_recording(False)
        
        # Stop audio capture
        await self.ws_client.stop_recording()

    async def handle_clear_history(self):
        """
        Handle clear history button click.
        """
        if not self.is_connected or not self.conversation_id:
            return
            
        # Clear local UI
        self.ui.clear_messages()
        
        # Send clear history message to server
        await self.ws_client.clear_history(self.conversation_id)
        
        self.ui.add_message("System", "Conversation history cleared")

    async def handle_server_connect(self):
        """
        Handle server connection success.
        """
        self.is_connected = True
        self.ui.set_connecting(False)
        self.ui.set_connected(True)
        
        # Send init message
        await self.ws_client.send_init()

    async def handle_server_disconnect(self):
        """
        Handle server disconnection.
        """
        self.is_connected = False
        self.is_recording = False
        self.session_id = None
        self.conversation_id = None
        
        self.ui.set_connecting(False)
        self.ui.set_connected(False)
        self.ui.set_recording(False)
        self.ui.add_message("System", "Disconnected from server")

    async def handle_server_message(self, message: Dict[str, Any]):
        """
        Handle messages from the server.
        
        Args:
            message: Message from server
        """
        message_type = message.get("type")
        payload = message.get("payload", {})
        
        if message_type == "init":
            # Handle initialization response
            self.session_id = payload.get("session_id")
            self.conversation_id = payload.get("conversation_id")
            self.ui.add_message("System", "Connected to server")
            
        elif message_type == "ready":
            # Handle ready message
            self.ui.add_message("System", "Voice assistant ready")
            
        elif message_type == "transcription":
            # Handle transcription
            text = payload.get("text", "")
            is_final = payload.get("is_final", False)
            
            if is_final:
                self.ui.add_message("You", text)
            else:
                self.ui.update_transcription(text)
                
        elif message_type == "response_token":
            # Handle streaming response token
            token = payload.get("token", "")
            self.ui.update_assistant_response(token)
            
        elif message_type == "response_complete":
            # Handle complete response
            self.ui.finalize_assistant_response()
            
        elif message_type == "audio":
            # Handle audio response
            audio_data = payload.get("audio")
            is_final = payload.get("is_final", False)
            
            # Play audio
            if audio_data:
                await self.ws_client.play_audio(audio_data)
                
        elif message_type == "error":
            # Handle error
            error = payload.get("error", "Unknown error")
            self.ui.add_message("Error", error)
            
        elif message_type == "stream_ended":
            # Handle stream end acknowledgement
            self.ui.set_recording(False)
            
        elif message_type == "history_cleared":
            # Handle history clear acknowledgement
            pass  # Already handled in UI
            
        else:
            logger.warning(f"Unknown message type: {message_type}")


def main():
    """
    Run the voice assistant client.
    """
    # Get server URL from environment variable or use default
    server_url = os.environ.get("VOICE_SERVER_URL", "ws://localhost:8000/ws")
    
    # Create and run the app
    app = VoiceAssistantApp(server_url)
    
    # Set up asyncio event loop
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.exception(f"Client error: {e}")

if __name__ == "__main__":
    main()