"""
Unit tests for the ConnectionDialog classes.
"""
import asyncio
import json
import os
import pytest
import tempfile
import tkinter as tk
from pathlib import Path
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import sys
import os
# Add the client directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.connection_manager import ConnectionManager
from client.connection_dialog import ConnectionDialog, ConnectionConfigDialog

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

@pytest.fixture
def mock_tk():
    """Mock tkinter to avoid creating actual windows during tests."""
    with patch('tkinter.Toplevel') as mock_toplevel:
        with patch('tkinter.Tk') as mock_tk:
            with patch('tkinter.ttk.Button') as mock_button:
                with patch('tkinter.ttk.Label') as mock_label:
                    with patch('tkinter.ttk.Entry') as mock_entry:
                    
                        # Configure mocks
                        mock_instance = mock_toplevel.return_value
                        mock_instance.winfo_screenwidth.return_value = 1024
                        mock_instance.winfo_screenheight.return_value = 768
                        mock_instance.winfo_width.return_value = 500
                        mock_instance.winfo_height.return_value = 400
                        
                        # Setup listbox mock
                        mock_listbox = MagicMock()
                        mock_listbox.curselection.return_value = (0,)
                        mock_listbox.get.return_value = "Test"
                        
                        yield {
                            'root': mock_tk,
                            'toplevel': mock_toplevel,
                            'button': mock_button,
                            'label': mock_label,
                            'entry': mock_entry,
                            'listbox': mock_listbox,
                        }

class TestConnectionDialog:
    """Test the ConnectionDialog class."""
    
    @patch('tkinter.Listbox')
    def test_initialization(self, mock_listbox, mock_tk, temp_config_file):
        """Test dialog initialization."""
        # Set up mocks
        mock_listbox.return_value = mock_tk['listbox']
        
        # Create connection manager
        manager = ConnectionManager(temp_config_file)
        
        # Create mock callback
        on_connect = MagicMock()
        
        with patch('client.connection_dialog.ConnectionDialog._populate_connection_list'):
            # Create dialog
            dialog = ConnectionDialog(
                parent=mock_tk['root'],
                connection_manager=manager,
                on_connect=on_connect
            )
            
            # Verify dialog initialization
            assert dialog.parent == mock_tk['root']
            assert dialog.connection_manager == manager
            assert dialog.on_connect == on_connect
            assert dialog.selected_connection is None
            
    @patch('tkinter.Listbox')
    def test_populate_connection_list(self, mock_listbox, mock_tk, temp_config_file):
        """Test populating the connection list."""
        # Set up mocks
        listbox_instance = MagicMock()
        mock_listbox.return_value = listbox_instance
        
        # Create connection manager
        manager = ConnectionManager(temp_config_file)
        
        # Create mock callback
        on_connect = MagicMock()
        
        with patch('client.connection_dialog.ConnectionDialog._on_connection_select'):
            # Create dialog
            dialog = ConnectionDialog(
                parent=mock_tk['root'],
                connection_manager=manager,
                on_connect=on_connect
            )
            
            # Replace the connection listbox
            dialog.connection_listbox = listbox_instance
            
            # Call the method to test
            dialog._populate_connection_list()
            
            # Verify list was populated
            assert listbox_instance.delete.call_count == 1
            assert listbox_instance.insert.call_count == len(manager.get_connection_names())
            
    @patch('tkinter.Listbox')
    def test_on_connection_select(self, mock_listbox, mock_tk, temp_config_file):
        """Test selecting a connection."""
        # Set up mocks
        mock_listbox_instance = MagicMock()
        mock_listbox_instance.curselection.return_value = (0,)
        mock_listbox_instance.get.return_value = "Test"
        mock_listbox.return_value = mock_listbox_instance
        
        # Create connection manager
        manager = ConnectionManager(temp_config_file)
        
        # Create mock callback
        on_connect = MagicMock()
        
        # Create dialog
        dialog = ConnectionDialog(
            parent=mock_tk['root'],
            connection_manager=manager,
            on_connect=on_connect
        )
        
        # Replace the connection listbox
        dialog.connection_listbox = mock_listbox_instance
        
        # Create mock buttons
        dialog.connect_button = MagicMock()
        dialog.edit_button = MagicMock()
        dialog.delete_button = MagicMock()
        
        # Call the method to test
        dialog._on_connection_select(None)
        
        # Verify selection was processed
        assert dialog.selected_connection == "Test"
        dialog.connect_button.config.assert_called_with(state=tk.NORMAL)
        dialog.edit_button.config.assert_called_with(state=tk.NORMAL)
        dialog.delete_button.config.assert_called_with(state=tk.NORMAL)
        
    @patch('tkinter.Listbox')
    def test_on_connect_button(self, mock_listbox, mock_tk, temp_config_file):
        """Test clicking the connect button."""
        # Set up mocks
        mock_listbox_instance = MagicMock()
        mock_listbox.return_value = mock_listbox_instance
        
        # Create connection manager
        manager = ConnectionManager(temp_config_file)
        
        # Create mock callback
        on_connect = MagicMock()
        
        # Create dialog
        dialog = ConnectionDialog(
            parent=mock_tk['root'],
            connection_manager=manager,
            on_connect=on_connect
        )
        
        # Set selected connection
        dialog.selected_connection = "Test"
        
        # Call the method to test
        dialog._on_connect_button()
        
        # Verify callback was called
        on_connect.assert_called_once_with("Test")
        dialog.dialog.destroy.assert_called_once()
        
    @patch('tkinter.Listbox')
    @patch('client.connection_dialog.ConnectionConfigDialog')
    def test_on_edit_button(self, mock_config_dialog, mock_listbox, mock_tk, temp_config_file):
        """Test clicking the edit button."""
        # Set up mocks
        mock_listbox_instance = MagicMock()
        mock_listbox.return_value = mock_listbox_instance
        
        # Create connection manager
        manager = ConnectionManager(temp_config_file)
        
        # Create mock callback
        on_connect = MagicMock()
        
        # Create dialog
        dialog = ConnectionDialog(
            parent=mock_tk['root'],
            connection_manager=manager,
            on_connect=on_connect
        )
        
        # Set selected connection
        dialog.selected_connection = "Test"
        
        # Call the method to test
        dialog._on_edit_button()
        
        # Verify config dialog was created
        mock_config_dialog.assert_called_once_with(
            dialog.dialog,
            manager,
            is_new=False,
            connection_name="Test",
            on_save=dialog._populate_connection_list
        )
        
    @patch('tkinter.Listbox')
    @patch('client.connection_dialog.ConnectionConfigDialog')
    def test_on_new_button(self, mock_config_dialog, mock_listbox, mock_tk, temp_config_file):
        """Test clicking the new button."""
        # Set up mocks
        mock_listbox_instance = MagicMock()
        mock_listbox.return_value = mock_listbox_instance
        
        # Create connection manager
        manager = ConnectionManager(temp_config_file)
        
        # Create mock callback
        on_connect = MagicMock()
        
        # Create dialog
        dialog = ConnectionDialog(
            parent=mock_tk['root'],
            connection_manager=manager,
            on_connect=on_connect
        )
        
        # Call the method to test
        dialog._on_new_button()
        
        # Verify config dialog was created
        mock_config_dialog.assert_called_once_with(
            dialog.dialog,
            manager,
            is_new=True,
            on_save=dialog._populate_connection_list
        )
        
    @patch('tkinter.Listbox')
    @patch('tkinter.messagebox.askyesno')
    def test_on_delete_button(self, mock_askyesno, mock_listbox, mock_tk, temp_config_file):
        """Test clicking the delete button."""
        # Set up mocks
        mock_listbox_instance = MagicMock()
        mock_listbox.return_value = mock_listbox_instance
        mock_askyesno.return_value = True
        
        # Create connection manager
        manager = ConnectionManager(temp_config_file)
        
        # Create mock callback
        on_connect = MagicMock()
        
        # Create dialog
        dialog = ConnectionDialog(
            parent=mock_tk['root'],
            connection_manager=manager,
            on_connect=on_connect
        )
        
        # Set selected connection
        dialog.selected_connection = "Test"
        
        # Mock the populate method
        dialog._populate_connection_list = MagicMock()
        
        # Call the method to test
        dialog._on_delete_button()
        
        # Verify delete was performed
        mock_askyesno.assert_called_once()
        assert "Test" not in manager.get_connection_names()
        dialog._populate_connection_list.assert_called_once()


class TestConnectionConfigDialog:
    """Test the ConnectionConfigDialog class."""
    
    def test_initialization_new(self, mock_tk, temp_config_file):
        """Test dialog initialization for a new connection."""
        # Create connection manager
        manager = ConnectionManager(temp_config_file)
        
        # Create mock callback
        on_save = MagicMock()
        
        with patch('client.connection_dialog.ConnectionConfigDialog._init_ui'):
            with patch('client.connection_dialog.ConnectionConfigDialog._toggle_ssh_fields'):
                # Create dialog
                dialog = ConnectionConfigDialog(
                    parent=mock_tk['root'],
                    connection_manager=manager,
                    is_new=True,
                    on_save=on_save
                )
                
                # Verify dialog initialization
                assert dialog.parent == mock_tk['root']
                assert dialog.connection_manager == manager
                assert dialog.is_new is True
                assert dialog.connection_name is None
                assert dialog.on_save == on_save
                assert dialog.connection is None
                
    def test_initialization_edit(self, mock_tk, temp_config_file):
        """Test dialog initialization for editing a connection."""
        # Create connection manager
        manager = ConnectionManager(temp_config_file)
        
        # Create mock callback
        on_save = MagicMock()
        
        with patch('client.connection_dialog.ConnectionConfigDialog._init_ui'):
            with patch('client.connection_dialog.ConnectionConfigDialog._toggle_ssh_fields'):
                with patch('client.connection_dialog.ConnectionConfigDialog._fill_fields'):
                    # Create dialog
                    dialog = ConnectionConfigDialog(
                        parent=mock_tk['root'],
                        connection_manager=manager,
                        is_new=False,
                        connection_name="Test",
                        on_save=on_save
                    )
                    
                    # Verify dialog initialization
                    assert dialog.parent == mock_tk['root']
                    assert dialog.connection_manager == manager
                    assert dialog.is_new is False
                    assert dialog.connection_name == "Test"
                    assert dialog.on_save == on_save
                    assert dialog.connection is not None
                    
    def test_toggle_ssh_fields(self, mock_tk, temp_config_file):
        """Test toggling SSH fields based on checkbox."""
        # Create connection manager
        manager = ConnectionManager(temp_config_file)
        
        # Create mock callback
        on_save = MagicMock()
        
        with patch('client.connection_dialog.ConnectionConfigDialog._init_ui'):
            with patch('client.connection_dialog.ConnectionConfigDialog._toggle_ssh_fields'):
                # Create dialog
                dialog = ConnectionConfigDialog(
                    parent=mock_tk['root'],
                    connection_manager=manager,
                    is_new=True,
                    on_save=on_save
                )
                
                # Mock the checkbox variable and ssh frame
                dialog.use_ssh_var = MagicMock()
                dialog.ssh_frame = MagicMock()
                dialog.ssh_frame.winfo_children.return_value = [MagicMock(), MagicMock()]
                
                # Test with SSH enabled
                dialog.use_ssh_var.get.return_value = True
                dialog._toggle_ssh_fields()
                
                # Verify all children were configured
                for child in dialog.ssh_frame.winfo_children():
                    child.configure.assert_called_with(state="normal")
                    
                # Test with SSH disabled
                dialog.use_ssh_var.get.return_value = False
                dialog._toggle_ssh_fields()
                
                # Verify all children were configured
                for child in dialog.ssh_frame.winfo_children():
                    child.configure.assert_called_with(state="disabled")
                    
    @patch('tkinter.messagebox.showerror')
    @patch('tkinter.messagebox.askyesno')
    def test_on_save_button_valid(self, mock_askyesno, mock_showerror, mock_tk, temp_config_file):
        """Test saving a valid connection."""
        # Create connection manager
        manager = ConnectionManager(temp_config_file)
        
        # Create mock callback
        on_save = MagicMock()
        
        with patch('client.connection_dialog.ConnectionConfigDialog._init_ui'):
            with patch('client.connection_dialog.ConnectionConfigDialog._toggle_ssh_fields'):
                # Create dialog
                dialog = ConnectionConfigDialog(
                    parent=mock_tk['root'],
                    connection_manager=manager,
                    is_new=True,
                    on_save=on_save
                )
                
                # Mock input fields
                dialog.is_new = True
                dialog.name_entry = MagicMock()
                dialog.name_entry.get.return_value = "NewTest"
                dialog.url_entry = MagicMock()
                dialog.url_entry.get.return_value = "ws://newtest:8000/ws"
                dialog.use_ssh_var = MagicMock()
                dialog.use_ssh_var.get.return_value = False
                
                # Call the method to test
                dialog._on_save_button()
                
                # Verify connection was saved
                assert "NewTest" in manager.get_connection_names()
                on_save.assert_called_once()
                dialog.dialog.destroy.assert_called_once()
                
    @patch('tkinter.messagebox.showerror')
    def test_on_save_button_invalid(self, mock_showerror, mock_tk, temp_config_file):
        """Test saving an invalid connection."""
        # Create connection manager
        manager = ConnectionManager(temp_config_file)
        
        # Create mock callback
        on_save = MagicMock()
        
        with patch('client.connection_dialog.ConnectionConfigDialog._init_ui'):
            with patch('client.connection_dialog.ConnectionConfigDialog._toggle_ssh_fields'):
                # Create dialog
                dialog = ConnectionConfigDialog(
                    parent=mock_tk['root'],
                    connection_manager=manager,
                    is_new=True,
                    on_save=on_save
                )
                
                # Mock input fields - missing name
                dialog.is_new = True
                dialog.name_entry = MagicMock()
                dialog.name_entry.get.return_value = ""  # Empty name
                dialog.url_entry = MagicMock()
                dialog.url_entry.get.return_value = "ws://newtest:8000/ws"
                dialog.use_ssh_var = MagicMock()
                dialog.use_ssh_var.get.return_value = False
                
                # Call the method to test
                dialog._on_save_button()
                
                # Verify error was shown
                mock_showerror.assert_called_once()
                assert not on_save.called
                assert not dialog.dialog.destroy.called