"""
Unit tests for the ConnectionManager class.
"""
import asyncio
import json
import os
import pytest
import tempfile
from pathlib import Path
import unittest.mock as mock

import sys
import os
# Add the client directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.connection_manager import ConnectionManager, SSHTunnel

@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        # Create test configuration
        test_config = {
            "Test": {
                "name": "Test",
                "url": "ws://test:8000/ws",
                "use_ssh": False
            },
            "TestSSH": {
                "name": "TestSSH",
                "url": "ws://test.remote:8000/ws",
                "use_ssh": True,
                "ssh_host": "test.remote",
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

class TestConnectionManager:
    """Test the ConnectionManager class."""
    
    def test_initialization_with_default_config(self):
        """Test initialization with default config path."""
        manager = ConnectionManager()
        assert manager.config_path == ConnectionManager.DEFAULT_CONFIG_PATH
        assert "Local" in manager.connections
        
    def test_initialization_with_custom_config(self, temp_config_file):
        """Test initialization with custom config path."""
        manager = ConnectionManager(temp_config_file)
        assert manager.config_path == temp_config_file
        assert "Test" in manager.connections
        assert "TestSSH" in manager.connections
        
    def test_get_connection_names(self, temp_config_file):
        """Test getting connection names."""
        manager = ConnectionManager(temp_config_file)
        names = manager.get_connection_names()
        assert "Test" in names
        assert "TestSSH" in names
        
    def test_get_connection(self, temp_config_file):
        """Test getting a specific connection."""
        manager = ConnectionManager(temp_config_file)
        connection = manager.get_connection("Test")
        assert connection["name"] == "Test"
        assert connection["url"] == "ws://test:8000/ws"
        assert connection["use_ssh"] is False
        
    def test_add_connection(self, temp_config_file):
        """Test adding a new connection."""
        manager = ConnectionManager(temp_config_file)
        manager.add_connection(
            name="NewTest",
            url="ws://new:8000/ws",
            use_ssh=False
        )
        
        # Verify connection was added
        connection = manager.get_connection("NewTest")
        assert connection["name"] == "NewTest"
        assert connection["url"] == "ws://new:8000/ws"
        assert connection["use_ssh"] is False
        
    def test_add_ssh_connection(self, temp_config_file):
        """Test adding a new SSH connection."""
        manager = ConnectionManager(temp_config_file)
        manager.add_connection(
            name="NewSSH",
            url="ws://newremote:8000/ws",
            use_ssh=True,
            ssh_host="newremote",
            ssh_port=2222,
            ssh_user="newuser",
            remote_host="localhost",
            remote_port=8000,
            local_port=8002
        )
        
        # Verify connection was added
        connection = manager.get_connection("NewSSH")
        assert connection["name"] == "NewSSH"
        assert connection["url"] == "ws://newremote:8000/ws"
        assert connection["use_ssh"] is True
        assert connection["ssh_host"] == "newremote"
        assert connection["ssh_port"] == 2222
        assert connection["ssh_user"] == "newuser"
        assert connection["remote_host"] == "localhost"
        assert connection["remote_port"] == 8000
        assert connection["local_port"] == 8002
        
    def test_update_connection(self, temp_config_file):
        """Test updating an existing connection."""
        manager = ConnectionManager(temp_config_file)
        manager.update_connection("Test", url="ws://updated:8000/ws")
        
        # Verify connection was updated
        connection = manager.get_connection("Test")
        assert connection["url"] == "ws://updated:8000/ws"
        
    def test_remove_connection(self, temp_config_file):
        """Test removing a connection."""
        manager = ConnectionManager(temp_config_file)
        manager.remove_connection("Test")
        
        # Verify connection was removed
        assert "Test" not in manager.get_connection_names()
        
    @pytest.mark.asyncio
    async def test_prepare_connection_direct(self, temp_config_file):
        """Test preparing a direct (non-SSH) connection."""
        manager = ConnectionManager(temp_config_file)
        success, url = await manager.prepare_connection("Test")
        
        assert success is True
        assert url == "ws://test:8000/ws"
        
    @pytest.mark.asyncio
    async def test_prepare_connection_ssh(self, temp_config_file):
        """Test preparing an SSH connection."""
        manager = ConnectionManager(temp_config_file)
        
        # Mock the SSHTunnel start method to return True
        with mock.patch.object(SSHTunnel, 'start', return_value=True):
            success, url = await manager.prepare_connection("TestSSH")
            
            assert success is True
            assert url == "ws://localhost:8001/ws"
            assert manager.active_tunnel is not None
            
    @pytest.mark.asyncio
    async def test_prepare_connection_ssh_failure(self, temp_config_file):
        """Test preparing an SSH connection when tunnel fails."""
        manager = ConnectionManager(temp_config_file)
        
        # Mock the SSHTunnel start method to return False
        with mock.patch.object(SSHTunnel, 'start', return_value=False):
            success, url = await manager.prepare_connection("TestSSH")
            
            assert success is False
            assert "Failed to establish SSH tunnel" in url
            assert manager.active_tunnel is None
            
    def test_close_tunnels(self, temp_config_file):
        """Test closing SSH tunnels."""
        manager = ConnectionManager(temp_config_file)
        
        # Set up a mock SSH tunnel
        mock_tunnel = mock.MagicMock()
        manager.active_tunnel = mock_tunnel
        
        # Call close_tunnels
        manager.close_tunnels()
        
        # Verify tunnel was stopped
        mock_tunnel.stop.assert_called_once()
        assert manager.active_tunnel is None


class TestSSHTunnel:
    """Test the SSHTunnel class."""
    
    def test_initialization(self):
        """Test initialization."""
        tunnel = SSHTunnel(
            ssh_host="test.remote",
            ssh_port=22,
            ssh_user="testuser",
            remote_host="localhost",
            remote_port=8000,
            local_port=8001
        )
        
        assert tunnel.ssh_host == "test.remote"
        assert tunnel.ssh_port == 22
        assert tunnel.ssh_user == "testuser"
        assert tunnel.remote_host == "localhost"
        assert tunnel.remote_port == 8000
        assert tunnel.local_port == 8001
        assert tunnel.process is None
        assert tunnel.is_running is False
        
    def test_start_success(self):
        """Test starting the tunnel with success."""
        tunnel = SSHTunnel(
            ssh_host="test.remote",
            ssh_port=22,
            ssh_user="testuser",
            remote_host="localhost",
            remote_port=8000,
            local_port=8001
        )
        
        # Mock subprocess.Popen
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = None  # Process is running
        
        with mock.patch('subprocess.Popen', return_value=mock_process):
            # Suppress the monitoring thread
            with mock.patch.object(tunnel, '_monitor_tunnel'):
                result = tunnel.start()
                
                assert result is True
                assert tunnel.is_running is True
                assert tunnel.process is mock_process
                
    def test_start_failure(self):
        """Test starting the tunnel with failure."""
        tunnel = SSHTunnel(
            ssh_host="test.remote",
            ssh_port=22,
            ssh_user="testuser",
            remote_host="localhost",
            remote_port=8000,
            local_port=8001
        )
        
        # Mock subprocess.Popen to return a process that has terminated
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = 1  # Process has exited
        mock_process.stderr = mock.MagicMock()
        mock_process.stderr.read.return_value = "SSH connection failed"
        
        with mock.patch('subprocess.Popen', return_value=mock_process):
            result = tunnel.start()
            
            assert result is False
            assert tunnel.is_running is False
            assert tunnel.process is None
                
    def test_stop(self):
        """Test stopping the tunnel."""
        tunnel = SSHTunnel(
            ssh_host="test.remote",
            ssh_port=22,
            ssh_user="testuser",
            remote_host="localhost",
            remote_port=8000,
            local_port=8001
        )
        
        # Set up a mock process
        mock_process = mock.MagicMock()
        tunnel.process = mock_process
        tunnel.is_running = True
        
        # Stop the tunnel
        tunnel.stop()
        
        # Verify process was terminated
        mock_process.terminate.assert_called_once()
        assert tunnel.is_running is False
        assert tunnel.process is None