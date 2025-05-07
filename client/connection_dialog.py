"""
Connection Dialog for the Voice Assistant client.

Provides a UI for managing server connections.
"""
import asyncio
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, List, Optional, Callable, Tuple

from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)

class ConnectionDialog:
    """
    Dialog for selecting and configuring server connections.
    """
    def __init__(self, parent: tk.Tk, connection_manager: ConnectionManager, 
                 on_connect: Callable[[str], None]):
        """
        Initialize the connection dialog.
        
        Args:
            parent: Parent window
            connection_manager: Connection manager instance
            on_connect: Callback for when a connection is selected
        """
        self.parent = parent
        self.connection_manager = connection_manager
        self.on_connect = on_connect
        self.selected_connection = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Connect to Server")
        self.dialog.geometry("500x400")
        self.dialog.resizable(True, True)
        self.dialog.minsize(400, 300)
        
        # Make it modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Initialize UI
        self._init_ui()
        
        # Center dialog
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        logger.info("Connection dialog initialized")
    
    def _init_ui(self):
        """
        Initialize the dialog UI.
        """
        # Configure grid layout
        self.dialog.grid_columnconfigure(0, weight=1)
        self.dialog.grid_rowconfigure(0, weight=0)  # Connection list label
        self.dialog.grid_rowconfigure(1, weight=1)  # Connection list
        self.dialog.grid_rowconfigure(2, weight=0)  # Buttons
        
        # Connection list label
        list_label = ttk.Label(
            self.dialog, 
            text="Select a connection:",
            font=("Arial", 12)
        )
        list_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        
        # Connection list frame
        list_frame = ttk.Frame(self.dialog)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)
        
        # Connection listbox with scrollbar
        self.connection_listbox = tk.Listbox(
            list_frame,
            font=("Arial", 11),
            activestyle="dotbox",
            selectmode=tk.SINGLE
        )
        self.connection_listbox.grid(row=0, column=0, sticky="nsew")
        list_scrollbar = ttk.Scrollbar(
            list_frame, 
            orient=tk.VERTICAL, 
            command=self.connection_listbox.yview
        )
        list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.connection_listbox.config(yscrollcommand=list_scrollbar.set)
        
        # Selection event
        self.connection_listbox.bind("<<ListboxSelect>>", self._on_connection_select)
        self.connection_listbox.bind("<Double-1>", self._on_connect_button)
        
        # Button frame
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        # Buttons
        self.connect_button = ttk.Button(
            button_frame,
            text="Connect",
            command=self._on_connect_button,
            state=tk.DISABLED
        )
        self.connect_button.pack(side=tk.RIGHT, padx=5)
        
        self.edit_button = ttk.Button(
            button_frame,
            text="Edit",
            command=self._on_edit_button,
            state=tk.DISABLED
        )
        self.edit_button.pack(side=tk.RIGHT, padx=5)
        
        self.new_button = ttk.Button(
            button_frame,
            text="New",
            command=self._on_new_button
        )
        self.new_button.pack(side=tk.RIGHT, padx=5)
        
        self.delete_button = ttk.Button(
            button_frame,
            text="Delete",
            command=self._on_delete_button,
            state=tk.DISABLED
        )
        self.delete_button.pack(side=tk.RIGHT, padx=5)
        
        # Fill connection list
        self._populate_connection_list()
    
    def _populate_connection_list(self):
        """
        Fill the connection listbox with available connections.
        """
        self.connection_listbox.delete(0, tk.END)
        
        for name in self.connection_manager.get_connection_names():
            self.connection_listbox.insert(tk.END, name)
            
        # Select first item if available
        if self.connection_listbox.size() > 0:
            self.connection_listbox.select_set(0)
            self._on_connection_select(None)
    
    def _on_connection_select(self, event):
        """
        Handle connection selection.
        
        Args:
            event: Selection event
        """
        selection = self.connection_listbox.curselection()
        if selection:
            index = selection[0]
            name = self.connection_listbox.get(index)
            self.selected_connection = name
            
            # Enable buttons
            self.connect_button.config(state=tk.NORMAL)
            self.edit_button.config(state=tk.NORMAL)
            self.delete_button.config(state=tk.NORMAL)
        else:
            self.selected_connection = None
            
            # Disable buttons
            self.connect_button.config(state=tk.DISABLED)
            self.edit_button.config(state=tk.DISABLED)
            self.delete_button.config(state=tk.DISABLED)
    
    def _on_connect_button(self, event=None):
        """
        Handle connect button click.
        
        Args:
            event: Button click event (optional)
        """
        if self.selected_connection:
            self.on_connect(self.selected_connection)
            self.dialog.destroy()
    
    def _on_edit_button(self):
        """
        Handle edit button click.
        """
        if self.selected_connection:
            connection = self.connection_manager.get_connection(self.selected_connection)
            if connection:
                ConnectionConfigDialog(
                    self.dialog,
                    self.connection_manager,
                    is_new=False,
                    connection_name=self.selected_connection,
                    on_save=self._populate_connection_list
                )
    
    def _on_new_button(self):
        """
        Handle new button click.
        """
        ConnectionConfigDialog(
            self.dialog,
            self.connection_manager,
            is_new=True,
            on_save=self._populate_connection_list
        )
    
    def _on_delete_button(self):
        """
        Handle delete button click.
        """
        if self.selected_connection:
            if messagebox.askyesno(
                "Confirm Delete",
                f"Are you sure you want to delete the connection '{self.selected_connection}'?"
            ):
                self.connection_manager.remove_connection(self.selected_connection)
                self._populate_connection_list()

class ConnectionConfigDialog:
    """
    Dialog for configuring a server connection.
    """
    def __init__(self, parent, connection_manager: ConnectionManager, 
                 is_new: bool = True, connection_name: str = None,
                 on_save: Callable[[], None] = None):
        """
        Initialize the connection configuration dialog.
        
        Args:
            parent: Parent window
            connection_manager: Connection manager instance
            is_new: Whether this is a new connection
            connection_name: Name of the connection to edit (if is_new is False)
            on_save: Callback for when the connection is saved
        """
        self.parent = parent
        self.connection_manager = connection_manager
        self.is_new = is_new
        self.connection_name = connection_name
        self.on_save = on_save
        
        # If editing, get the connection
        self.connection = None
        if not is_new and connection_name:
            self.connection = self.connection_manager.get_connection(connection_name)
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("New Connection" if is_new else "Edit Connection")
        self.dialog.geometry("500x450")
        self.dialog.resizable(True, False)
        
        # Make it modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Initialize UI
        self._init_ui()
        
        # Center dialog
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        logger.info(f"Connection config dialog initialized (is_new={is_new})")
    
    def _init_ui(self):
        """
        Initialize the dialog UI.
        """
        # Configure grid layout
        self.dialog.grid_columnconfigure(0, weight=0)  # Labels
        self.dialog.grid_columnconfigure(1, weight=1)  # Inputs
        
        # Add form fields
        
        # Connection name
        ttk.Label(self.dialog, text="Connection Name:").grid(
            row=0, column=0, sticky="w", padx=10, pady=10
        )
        self.name_entry = ttk.Entry(self.dialog, width=40)
        self.name_entry.grid(
            row=0, column=1, sticky="ew", padx=10, pady=10
        )
        
        # Server URL
        ttk.Label(self.dialog, text="Server URL:").grid(
            row=1, column=0, sticky="w", padx=10, pady=10
        )
        self.url_entry = ttk.Entry(self.dialog, width=40)
        self.url_entry.grid(
            row=1, column=1, sticky="ew", padx=10, pady=10
        )
        
        # Use SSH tunnel
        ttk.Label(self.dialog, text="Use SSH Tunnel:").grid(
            row=2, column=0, sticky="w", padx=10, pady=10
        )
        self.use_ssh_var = tk.BooleanVar(value=False)
        self.use_ssh_check = ttk.Checkbutton(
            self.dialog,
            variable=self.use_ssh_var,
            command=self._toggle_ssh_fields
        )
        self.use_ssh_check.grid(
            row=2, column=1, sticky="w", padx=10, pady=10
        )
        
        # SSH settings frame
        self.ssh_frame = ttk.LabelFrame(self.dialog, text="SSH Tunnel Settings")
        self.ssh_frame.grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=10
        )
        
        # Configure ssh_frame grid
        self.ssh_frame.grid_columnconfigure(0, weight=0)  # Labels
        self.ssh_frame.grid_columnconfigure(1, weight=1)  # Inputs
        
        # SSH Host
        ttk.Label(self.ssh_frame, text="SSH Host:").grid(
            row=0, column=0, sticky="w", padx=10, pady=5
        )
        self.ssh_host_entry = ttk.Entry(self.ssh_frame, width=30)
        self.ssh_host_entry.grid(
            row=0, column=1, sticky="ew", padx=10, pady=5
        )
        
        # SSH Port
        ttk.Label(self.ssh_frame, text="SSH Port:").grid(
            row=1, column=0, sticky="w", padx=10, pady=5
        )
        self.ssh_port_entry = ttk.Entry(self.ssh_frame, width=10)
        self.ssh_port_entry.grid(
            row=1, column=1, sticky="w", padx=10, pady=5
        )
        self.ssh_port_entry.insert(0, "22")
        
        # SSH Username
        ttk.Label(self.ssh_frame, text="SSH Username:").grid(
            row=2, column=0, sticky="w", padx=10, pady=5
        )
        self.ssh_user_entry = ttk.Entry(self.ssh_frame, width=20)
        self.ssh_user_entry.grid(
            row=2, column=1, sticky="w", padx=10, pady=5
        )
        
        # Remote Host
        ttk.Label(self.ssh_frame, text="Remote Host:").grid(
            row=3, column=0, sticky="w", padx=10, pady=5
        )
        self.remote_host_entry = ttk.Entry(self.ssh_frame, width=30)
        self.remote_host_entry.grid(
            row=3, column=1, sticky="ew", padx=10, pady=5
        )
        self.remote_host_entry.insert(0, "localhost")
        
        # Remote Port
        ttk.Label(self.ssh_frame, text="Remote Port:").grid(
            row=4, column=0, sticky="w", padx=10, pady=5
        )
        self.remote_port_entry = ttk.Entry(self.ssh_frame, width=10)
        self.remote_port_entry.grid(
            row=4, column=1, sticky="w", padx=10, pady=5
        )
        self.remote_port_entry.insert(0, "8000")
        
        # Local Port
        ttk.Label(self.ssh_frame, text="Local Port:").grid(
            row=5, column=0, sticky="w", padx=10, pady=5
        )
        self.local_port_entry = ttk.Entry(self.ssh_frame, width=10)
        self.local_port_entry.grid(
            row=5, column=1, sticky="w", padx=10, pady=5
        )
        self.local_port_entry.insert(0, "8000")
        
        # Button frame
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=10
        )
        
        # Save button
        save_button = ttk.Button(
            button_frame,
            text="Save",
            command=self._on_save_button
        )
        save_button.pack(side=tk.RIGHT, padx=5)
        
        # Cancel button
        cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.dialog.destroy
        )
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Initial state
        self._toggle_ssh_fields()
        
        # Fill fields if editing
        if self.connection:
            self._fill_fields()
    
    def _toggle_ssh_fields(self):
        """
        Toggle SSH tunnel fields based on checkbox.
        """
        if self.use_ssh_var.get():
            for child in self.ssh_frame.winfo_children():
                child.configure(state="normal")
        else:
            for child in self.ssh_frame.winfo_children():
                if isinstance(child, ttk.Entry) or isinstance(child, ttk.Checkbutton):
                    child.configure(state="disabled")
    
    def _fill_fields(self):
        """
        Fill fields with connection data.
        """
        if not self.connection:
            return
            
        self.name_entry.insert(0, self.connection.get("name", ""))
        self.url_entry.insert(0, self.connection.get("url", ""))
        
        use_ssh = self.connection.get("use_ssh", False)
        self.use_ssh_var.set(use_ssh)
        self._toggle_ssh_fields()
        
        if use_ssh:
            self.ssh_host_entry.delete(0, tk.END)
            self.ssh_host_entry.insert(0, self.connection.get("ssh_host", ""))
            
            self.ssh_port_entry.delete(0, tk.END)
            self.ssh_port_entry.insert(0, str(self.connection.get("ssh_port", 22)))
            
            self.ssh_user_entry.delete(0, tk.END)
            self.ssh_user_entry.insert(0, self.connection.get("ssh_user", ""))
            
            self.remote_host_entry.delete(0, tk.END)
            self.remote_host_entry.insert(0, self.connection.get("remote_host", "localhost"))
            
            self.remote_port_entry.delete(0, tk.END)
            self.remote_port_entry.insert(0, str(self.connection.get("remote_port", 8000)))
            
            self.local_port_entry.delete(0, tk.END)
            self.local_port_entry.insert(0, str(self.connection.get("local_port", 8000)))
    
    def _on_save_button(self):
        """
        Handle save button click.
        """
        # Validate fields
        name = self.name_entry.get().strip()
        url = self.url_entry.get().strip()
        
        if not name:
            messagebox.showerror("Error", "Connection name is required")
            return
            
        if not url:
            messagebox.showerror("Error", "Server URL is required")
            return
            
        # Validate URL format
        if not url.startswith("ws://") and not url.startswith("wss://"):
            if not messagebox.askyesno(
                "Warning",
                "Server URL should start with 'ws://' or 'wss://'. Would you like to add the prefix?"
            ):
                return
            else:
                url = "ws://" + url
        
        # Prepare connection data
        connection_data = {
            "name": name,
            "url": url,
            "use_ssh": self.use_ssh_var.get()
        }
        
        if self.use_ssh_var.get():
            try:
                ssh_port = int(self.ssh_port_entry.get().strip())
                remote_port = int(self.remote_port_entry.get().strip())
                local_port = int(self.local_port_entry.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Port numbers must be integers")
                return
                
            ssh_host = self.ssh_host_entry.get().strip()
            ssh_user = self.ssh_user_entry.get().strip()
            remote_host = self.remote_host_entry.get().strip()
            
            if not ssh_host:
                messagebox.showerror("Error", "SSH host is required")
                return
                
            if not ssh_user:
                messagebox.showerror("Error", "SSH username is required")
                return
            
            connection_data.update({
                "ssh_host": ssh_host,
                "ssh_port": ssh_port,
                "ssh_user": ssh_user,
                "remote_host": remote_host,
                "remote_port": remote_port,
                "local_port": local_port
            })
        
        # Check if name already exists for new connections
        if self.is_new and name in self.connection_manager.get_connection_names():
            if not messagebox.askyesno(
                "Warning",
                f"A connection named '{name}' already exists. Do you want to replace it?"
            ):
                return
        
        # Save connection
        if self.is_new:
            self.connection_manager.add_connection(**connection_data)
        else:
            # If name changed, remove old connection
            if self.connection_name != name:
                self.connection_manager.remove_connection(self.connection_name)
                self.connection_manager.add_connection(**connection_data)
            else:
                self.connection_manager.update_connection(name, **connection_data)
        
        # Call on_save callback
        if self.on_save:
            self.on_save()
            
        # Close dialog
        self.dialog.destroy()