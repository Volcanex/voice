"""
Integration tests for the voice assistant client.

These tests check the integration between the different client components.
"""
import asyncio
import json
import os
import pytest
import tempfile
import tkinter as tk
from pathlib import Path
import unittest.mock as mock
from unittest.mock import MagicMock, patch, AsyncMock

import sys
import os
# Add the client directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.connection_manager import ConnectionManager
from client.connection_dialog import ConnectionDialog
from client.main import VoiceAssistantApp
from client.websocket_client import WebSocketClient

@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        # Create test configuration
        test_config = {
            "Local": {
                "name": "Local",
                "url": "ws://localhost:8000/ws",
                "use_ssh": False
            },
            "Remote": {
                "name": "Remote",
                "url": "ws://remote:8000/ws",
                "use_ssh": True,
                "ssh_host": "remote",
                "ssh_port": 22,
                "ssh_user": "testuser",
                "remote_host": "localhost",
                "remote_port": 8000,
                "local_port": 8001
            }
        }
        temp_file.write(json.dumps(test_config).encode('utf-8'))
        temp_path = Path(temp_file.name)
    
    yield temp_path
    
    # Clean up
    try:
        os.remove(temp_path)
    except:
        pass

@pytest.fixture
def mock_tk():
    """Mock tkinter to avoid creating actual windows during tests."""
    with patch('tkinter.Tk') as mock_tk:
        mock_instance = mock_tk.return_value
        mock_instance.title = MagicMock()
        mock_instance.update = MagicMock()
        mock_instance.winfo_exists = MagicMock(return_value=True)
        mock_instance.destroy = MagicMock()
        yield mock_instance

class TestVoiceAssistantApp:
    """Test the VoiceAssistantApp class."""
    
    def test_initialization(self):
        """Test app initialization."""
        # Patch ConnectionManager to prevent file operations
        with patch('client.main.ConnectionManager', autospec=True):
            # Create app
            app = VoiceAssistantApp(server_url="ws://test:8000/ws")
            
            # Verify initialization
            assert app.server_url == "ws://test:8000/ws"
            assert app.ui is None
            assert app.ws_client is None
            assert app.session_id is None
            assert app.conversation_id is None
            assert app.is_connected is False
            assert app.is_recording is False
            assert app.current_connection is None
            assert app.connection_manager is not None
            
    @pytest.mark.asyncio
    async def test_run(self, mock_tk):
        """Test running the app."""
        with patch('client.main.WebSocketClient', autospec=True) as mock_ws_client:
            with patch('client.main.VoiceAssistantUI', autospec=True) as mock_ui:
                with patch('client.main.ConnectionManager', autospec=True):
                    # Create app
                    app = VoiceAssistantApp(server_url="ws://test:8000/ws")
                    
                    # Mock asyncio.sleep to raise an exception after first call to end the loop
                    with patch('asyncio.sleep', side_effect=[None, KeyboardInterrupt]):
                        try:
                            await app.run()
                        except KeyboardInterrupt:
                            pass
                        
                        # Verify UI was created
                        mock_ui.assert_called_once()
                        mock_ws_client.assert_called_once_with("ws://test:8000/ws")
                        assert app.ui is not None
                        assert app.ws_client is not None
                        
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test app shutdown."""
        with patch('client.main.WebSocketClient', autospec=True) as mock_ws_client:
            with patch('client.main.VoiceAssistantUI', autospec=True) as mock_ui:
                with patch('client.main.ConnectionManager', autospec=True) as mock_cm:
                    # Create app
                    app = VoiceAssistantApp(server_url="ws://test:8000/ws")
                    
                    # Set up mocks
                    app.ws_client = mock_ws_client.return_value
                    app.ui = mock_ui.return_value
                    app.connection_manager = mock_cm.return_value
                    app.ws_client.disconnect = AsyncMock()
                    
                    # Call shutdown
                    await app.shutdown()
                    
                    # Verify resources were cleaned up
                    app.ws_client.disconnect.assert_called_once()
                    app.connection_manager.disconnect.assert_called_once()
                    app.ui.root.destroy.assert_called_once()
                    
    @pytest.mark.asyncio
    async def test_handle_connect_when_disconnected(self):
        """Test connect button when disconnected."""
        with patch('client.main.ConnectionDialog', autospec=True) as mock_dialog:
            # Create app
            app = VoiceAssistantApp(server_url="ws://test:8000/ws")
            
            # Set up mocks
            app.ui = MagicMock()
            app.is_connected = False
            
            # Call connect handler
            await app.handle_connect()
            
            # Verify dialog was shown
            mock_dialog.assert_called_once_with(app.ui.root, app.connection_manager, app.handle_connection_select)
            
    @pytest.mark.asyncio
    async def test_handle_connect_when_connected(self):
        """Test connect button when already connected."""
        # Create app
        app = VoiceAssistantApp(server_url="ws://test:8000/ws")
        
        # Set up mocks
        app.ui = MagicMock()
        app.ws_client = MagicMock()
        app.ws_client.disconnect = AsyncMock()
        app.connection_manager = MagicMock()
        app.is_connected = True
        app.current_connection = "Test"
        
        # Call connect handler
        await app.handle_connect()
        
        # Verify disconnect was performed
        app.ws_client.disconnect.assert_called_once()
        app.connection_manager.disconnect.assert_called_once()
        app.ui.set_connected.assert_called_with(False)
        app.ui.set_connection_label.assert_called_with("Not connected")
        app.ui.add_message.assert_called_with("System", "Disconnected from server")
        assert app.is_connected is False
        assert app.current_connection is None
        
    @pytest.mark.asyncio
    async def test_handle_connection_select_success(self):
        """Test selecting a connection successfully."""
        # Create app
        app = VoiceAssistantApp(server_url="ws://test:8000/ws")
        
        # Set up mocks
        app.ui = MagicMock()
        app.connection_manager = MagicMock()
        app.connection_manager.connect = AsyncMock(return_value=(True, "ws://new:8000/ws"))
        
        # Mock the WebSocketClient
        with patch('client.main.WebSocketClient', autospec=True) as mock_ws_client:
            ws_client_instance = mock_ws_client.return_value
            ws_client_instance.connect = AsyncMock()
            
            # Call connection select handler
            await app.handle_connection_select("Test")
            
            # Verify connection was established
            app.connection_manager.connect.assert_called_once_with("Test")
            mock_ws_client.assert_called_once_with("ws://new:8000/ws")
            ws_client_instance.connect.assert_called_once()
            app.ui.set_connection_label.assert_called_with("Test")
            assert app.current_connection == "Test"
            assert app.server_url == "ws://new:8000/ws"
            
    @pytest.mark.asyncio
    async def test_handle_connection_select_failure_with_tunnel(self):
        """Test selecting a connection with a failed tunnel."""
        # Create app
        app = VoiceAssistantApp(server_url="ws://test:8000/ws")
        
        # Set up mocks
        app.ui = MagicMock()
        app.connection_manager = MagicMock()
        app.connection_manager.connect = AsyncMock(return_value=(False, "Failed to establish SSH tunnel"))
        
        # Call connection select handler
        await app.handle_connection_select("TestSSH")
        
        # Verify error was displayed
        app.ui.add_message.assert_called_with("Error", "Failed to connect: Failed to establish SSH tunnel")
        app.ui.set_connecting.assert_called_with(False)
        app.ui.set_connected.assert_called_with(False)
        
    @pytest.mark.asyncio
    async def test_handle_connection_select_failure_with_websocket(self):
        """Test selecting a connection with a failed WebSocket connection."""
        # Create app
        app = VoiceAssistantApp(server_url="ws://test:8000/ws")
        
        # Set up mocks
        app.ui = MagicMock()
        app.connection_manager = MagicMock()
        app.connection_manager.connect = AsyncMock(return_value=(True, "ws://new:8000/ws"))
        
        # Mock the WebSocketClient with an error
        with patch('client.main.WebSocketClient', autospec=True) as mock_ws_client:
            ws_client_instance = mock_ws_client.return_value
            ws_client_instance.connect = AsyncMock(side_effect=Exception("WebSocket connection failed"))
            
            # Call connection select handler
            await app.handle_connection_select("Test")
            
            # Verify error was displayed
            app.ui.add_message.assert_called_with("Error", "Failed to connect: WebSocket connection failed")
            app.ui.set_connecting.assert_called_with(False)
            app.ui.set_connected.assert_called_with(False)
            app.connection_manager.disconnect.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_handle_server_connect(self):
        """Test the server connect handler."""
        # Create app
        app = VoiceAssistantApp(server_url="ws://test:8000/ws")
        
        # Set up mocks
        app.ui = MagicMock()
        app.ws_client = MagicMock()
        app.ws_client.send_init = AsyncMock()
        
        # Call server connect handler
        await app.handle_server_connect()
        
        # Verify UI was updated
        assert app.is_connected is True
        app.ui.set_connecting.assert_called_with(False)
        app.ui.set_connected.assert_called_with(True)
        app.ws_client.send_init.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_handle_server_disconnect(self):
        """Test the server disconnect handler."""
        # Create app
        app = VoiceAssistantApp(server_url="ws://test:8000/ws")
        
        # Set up mocks
        app.ui = MagicMock()
        app.is_connected = True
        app.is_recording = True
        app.session_id = "123"
        app.conversation_id = "456"
        
        # Call server disconnect handler
        await app.handle_server_disconnect()
        
        # Verify state was reset
        assert app.is_connected is False
        assert app.is_recording is False
        assert app.session_id is None
        assert app.conversation_id is None
        app.ui.set_connecting.assert_called_with(False)
        app.ui.set_connected.assert_called_with(False)
        app.ui.set_recording.assert_called_with(False)
        app.ui.add_message.assert_called_with("System", "Disconnected from server")
        
    @pytest.mark.asyncio
    async def test_handle_server_message_init(self):
        """Test handling an init message from the server."""
        # Create app
        app = VoiceAssistantApp(server_url="ws://test:8000/ws")
        
        # Set up mocks
        app.ui = MagicMock()
        
        # Create init message
        message = {
            "type": "init",
            "payload": {
                "session_id": "123",
                "conversation_id": "456"
            }
        }
        
        # Call message handler
        await app.handle_server_message(message)
        
        # Verify session was initialized
        assert app.session_id == "123"
        assert app.conversation_id == "456"
        app.ui.add_message.assert_called_with("System", "Connected to server")
        
    @pytest.mark.asyncio
    async def test_handle_server_message_response_token(self):
        """Test handling a response token message from the server."""
        # Create app
        app = VoiceAssistantApp(server_url="ws://test:8000/ws")
        
        # Set up mocks
        app.ui = MagicMock()
        
        # Create response token message
        message = {
            "type": "response_token",
            "payload": {
                "token": "Hello,"
            }
        }
        
        # Call message handler
        await app.handle_server_message(message)
        
        # Verify token was processed
        app.ui.update_assistant_response.assert_called_with("Hello,")