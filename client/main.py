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

# Handle both package and direct script execution
if __package__ is None or __package__ == '':
    # Running as a script
    import ui
    import websocket_client
    import connection_manager
    import server_dialog
    from ui import VoiceAssistantUI
    from websocket_client import WebSocketClient
    from connection_manager import ConnectionManager
    from server_dialog import ServerSelectionDialog
else:
    # Running as a package
    from .ui import VoiceAssistantUI
    from .websocket_client import WebSocketClient
    from .connection_manager import ConnectionManager
    from .server_dialog import ServerSelectionDialog

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
            server_url: Default WebSocket server URL
        """
        self.server_url = server_url
        self.ui = None
        self.ws_client = None
        self.session_id = None
        self.conversation_id = None
        
        # Initialize connection manager
        self.connection_manager = ConnectionManager()
        
        # Ensure the local connection exists
        if "Local" not in self.connection_manager.get_connection_names():
            self.connection_manager.add_connection(
                name="Local",
                url=server_url,
                use_ssh=False
            )
        
        # Application state
        self.is_connected = False
        self.is_recording = False
        self.current_connection = None
        
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
        
        # Close any SSH tunnels
        if self.connection_manager:
            self.connection_manager.close_tunnels()
            
        # Close UI
        if self.ui and self.ui.root.winfo_exists():
            self.ui.root.destroy()
            
        logger.info("Client shutdown complete")

    async def handle_connect(self):
        """
        Handle connect button click.
        """
        if self.is_connected:
            # Disconnect from server
            await self.ws_client.disconnect()
            
            # Close any SSH tunnels
            self.connection_manager.close_tunnels()
            
            # Update UI
            self.is_connected = False
            self.current_connection = None
            self.ui.set_connected(False)
            self.ui.set_connection_label("Not connected")
            self.ui.add_message("System", "Disconnected from server")
            return
            
        # Show connection dialog
        ServerSelectionDialog(self.ui.root, self.connection_manager, self.handle_connection_select)
    
    async def handle_connection_select(self, connection_name: str):
        """
        Handle connection selection from the dialog.
        
        Args:
            connection_name: Selected connection name
        """
        self.ui.set_connecting(True)
        self.ui.add_message("System", f"Connecting to {connection_name}...")
        
        logger.info(f"Starting connection process to {connection_name}")
        connection = self.connection_manager.get_connection(connection_name)
        if connection:
            logger.info(f"Connection details: URL={connection.get('url')}, SSH={connection.get('use_ssh', False)}")
        
        # Get connection URL (establishing SSH tunnel if needed)
        logger.info("Setting up connection (and SSH tunnel if configured)...")
        success, result = await self.connection_manager.prepare_connection(connection_name)
        
        if not success:
            # Connection failed
            logger.error(f"Connection error: {result}")
            self.ui.add_message("Error", f"Failed to connect: {result}")
            self.ui.set_connecting(False)
            self.ui.set_connected(False)
            return
            
        # Connect to server
        try:
            # Update server URL
            logger.info(f"Connection established. Using WebSocket URL: {result}")
            self.server_url = result
            
            # Create new WebSocket client with the updated URL
            logger.info("Creating new WebSocket client...")
            self.ws_client = WebSocketClient(self.server_url)
            
            # Set up event handlers
            logger.info("Setting up WebSocket event handlers")
            self.ws_client.on_message = self.handle_server_message
            self.ws_client.on_connect = self.handle_server_connect
            self.ws_client.on_disconnect = self.handle_server_disconnect
            
            # Connect to WebSocket server
            logger.info("Connecting to WebSocket server...")
            await self.ws_client.connect()
            
            # Set current connection
            logger.info(f"Successfully connected to {connection_name}")
            self.current_connection = connection_name
            
            # Update UI
            self.ui.set_connection_label(connection_name)
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            logger.debug(f"Connection error details:", exc_info=True)
            self.ui.add_message("Error", f"Failed to connect: {str(e)}")
            self.ui.set_connecting(False)
            self.ui.set_connected(False)
            
            # Close any SSH tunnels
            logger.info("Cleaning up SSH tunnels due to connection error")
            self.connection_manager.close_tunnels()

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