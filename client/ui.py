"""
UI module for the voice assistant client.

This module provides a TKinter-based UI for the voice assistant.
"""
import asyncio
import logging
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Dict, Any, Optional, Callable, List
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class VoiceAssistantUI:
    """
    TKinter UI for the voice assistant client.
    """
    def __init__(
        self,
        root: tk.Tk,
        on_connect: Callable[[], None],
        on_send_text: Callable[[str], None],
        on_start_recording: Callable[[], None],
        on_stop_recording: Callable[[], None],
        on_clear_history: Callable[[], None]
    ):
        """
        Initialize the voice assistant UI.
        
        Args:
            root: TKinter root window
            on_connect: Callback for connect button
            on_send_text: Callback for send text button
            on_start_recording: Callback for start recording button
            on_stop_recording: Callback for stop recording button
            on_clear_history: Callback for clear history button
        """
        self.root = root
        self.on_connect = self._wrap_callback(on_connect)
        self.on_send_text = self._wrap_callback(on_send_text)
        self.on_start_recording = self._wrap_callback(on_start_recording)
        self.on_stop_recording = self._wrap_callback(on_stop_recording)
        self.on_clear_history = self._wrap_callback(on_clear_history)
        
        # Set window size and properties
        self.root.geometry("800x600")
        self.root.minsize(400, 300)
        
        # Initialize state
        self.is_connected = False
        self.is_connecting = False
        self.is_recording = False
        
        # Current assistant response being built
        self.current_assistant_response = ""
        
        # Initialize UI components
        self._init_ui()
        
        logger.info("Voice Assistant UI initialized")

    def _init_ui(self):
        """
        Initialize the UI components.
        """
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=0)
        
        # Create header frame
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.grid(row=0, column=0, sticky="ew")
        
        # Title
        title_label = ttk.Label(
            header_frame, 
            text="Voice Assistant", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.LEFT, padx=5)
        
        # Connect button
        self.connect_button = ttk.Button(
            header_frame, 
            text="Connect", 
            command=self.on_connect
        )
        self.connect_button.pack(side=tk.RIGHT, padx=5)
        
        # Status indicator
        self.status_label = ttk.Label(
            header_frame, 
            text="Disconnected", 
            foreground="red"
        )
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Connection label
        self.connection_label = ttk.Label(
            header_frame,
            text="Not connected",
            font=("Arial", 10, "italic")
        )
        self.connection_label.pack(side=tk.RIGHT, padx=5)
        
        # Create conversation frame
        conv_frame = ttk.Frame(self.root, padding="10")
        conv_frame.grid(row=1, column=0, sticky="nsew")
        conv_frame.grid_columnconfigure(0, weight=1)
        conv_frame.grid_rowconfigure(0, weight=1)
        
        # Conversation display
        self.conversation_text = scrolledtext.ScrolledText(
            conv_frame, 
            wrap=tk.WORD, 
            state=tk.DISABLED,
            font=("Arial", 11)
        )
        self.conversation_text.grid(row=0, column=0, sticky="nsew")
        
        # Create input frame
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.grid(row=2, column=0, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Transcription display
        self.transcription_label = ttk.Label(
            input_frame, 
            text="", 
            foreground="gray",
            wraplength=600
        )
        self.transcription_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 5))
        
        # Text input
        self.text_input = ttk.Entry(input_frame, font=("Arial", 11))
        self.text_input.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        self.text_input.bind("<Return>", self._on_enter_key)
        
        # Send button
        self.send_button = ttk.Button(
            input_frame, 
            text="Send", 
            command=self._on_send_button
        )
        self.send_button.grid(row=1, column=1, padx=(0, 5))
        
        # Voice button
        self.voice_button = ttk.Button(
            input_frame, 
            text="Record", 
            command=self._on_voice_button
        )
        self.voice_button.grid(row=1, column=2, padx=(0, 5))
        
        # Clear button
        self.clear_button = ttk.Button(
            input_frame, 
            text="Clear", 
            command=self._on_clear_button
        )
        self.clear_button.grid(row=1, column=3)
        
        # Initial UI state
        self.set_connected(False)

    def _wrap_callback(self, callback: Callable) -> Callable:
        """
        Wrap a callback to ensure it returns a future.
        
        Args:
            callback: Callback function
            
        Returns:
            Wrapped callback
        """
        def wrapper(*args, **kwargs):
            result = callback(*args, **kwargs)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
            return result
        return wrapper

    def update(self):
        """
        Update the UI, needs to be called periodically.
        """
        self.root.update()

    def set_connected(self, is_connected: bool):
        """
        Update UI to reflect connection state.
        
        Args:
            is_connected: Whether connected to server
        """
        self.is_connected = is_connected
        
        if is_connected:
            self.status_label.config(text="Connected", foreground="green")
            self.connect_button.config(text="Disconnect")
            self.send_button.config(state=tk.NORMAL)
            self.voice_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL)
            self.text_input.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="Disconnected", foreground="red")
            self.connect_button.config(text="Connect")
            self.send_button.config(state=tk.DISABLED)
            self.voice_button.config(state=tk.DISABLED)
            self.clear_button.config(state=tk.DISABLED)
            self.text_input.config(state=tk.DISABLED)
            
            # Reset recording state
            self.set_recording(False)

    def set_connecting(self, is_connecting: bool):
        """
        Update UI to reflect connecting state.
        
        Args:
            is_connecting: Whether connecting to server
        """
        self.is_connecting = is_connecting
        
        if is_connecting:
            self.status_label.config(text="Connecting...", foreground="orange")
            self.connect_button.config(state=tk.DISABLED)
        else:
            self.connect_button.config(state=tk.NORMAL)
            
    def set_connection_label(self, name: str):
        """
        Update the connection label with the server name.
        
        Args:
            name: Connection name
        """
        self.connection_label.config(text=name)

    def set_recording(self, is_recording: bool):
        """
        Update UI to reflect recording state.
        
        Args:
            is_recording: Whether currently recording
        """
        self.is_recording = is_recording
        
        if is_recording:
            self.voice_button.config(text="Stop", style="Accent.TButton")
            self.send_button.config(state=tk.DISABLED)
            self.text_input.config(state=tk.DISABLED)
        else:
            self.voice_button.config(text="Record", style="TButton")
            if self.is_connected:
                self.send_button.config(state=tk.NORMAL)
                self.text_input.config(state=tk.NORMAL)
            self.transcription_label.config(text="")

    def add_message(self, sender: str, message: str):
        """
        Add a message to the conversation display.
        
        Args:
            sender: Message sender (e.g., "You", "Assistant")
            message: Message content
        """
        # Enable editing
        self.conversation_text.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Insert message with appropriate formatting
        if self.conversation_text.index('end-1c') != '1.0':
            self.conversation_text.insert(tk.END, "\n\n")
            
        # Insert with tag based on sender
        tag = f"tag_{sender.lower()}"
        
        self.conversation_text.insert(tk.END, f"[{timestamp}] {sender}: ", tag)
        self.conversation_text.insert(tk.END, message)
        
        # Configure tags for different senders
        if sender == "You":
            self.conversation_text.tag_configure(tag, foreground="blue", font=("Arial", 11, "bold"))
        elif sender == "Assistant":
            self.conversation_text.tag_configure(tag, foreground="green", font=("Arial", 11, "bold"))
        elif sender == "System":
            self.conversation_text.tag_configure(tag, foreground="gray", font=("Arial", 11, "italic"))
        elif sender == "Error":
            self.conversation_text.tag_configure(tag, foreground="red", font=("Arial", 11, "bold"))
        
        # Scroll to bottom
        self.conversation_text.see(tk.END)
        
        # Disable editing
        self.conversation_text.config(state=tk.DISABLED)

    def update_transcription(self, text: str):
        """
        Update the transcription display.
        
        Args:
            text: Transcription text
        """
        self.transcription_label.config(text=text)

    def update_assistant_response(self, token: str):
        """
        Update the assistant's response in the conversation display.
        
        Args:
            token: New token to append
        """
        # If this is the first token, add the message header
        if not self.current_assistant_response:
            self.conversation_text.config(state=tk.NORMAL)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Insert message with appropriate formatting
            if self.conversation_text.index('end-1c') != '1.0':
                self.conversation_text.insert(tk.END, "\n\n")
                
            # Insert header with tag
            self.conversation_text.insert(tk.END, f"[{timestamp}] Assistant: ", "tag_assistant")
            self.conversation_text.tag_configure(
                "tag_assistant", 
                foreground="green", 
                font=("Arial", 11, "bold")
            )
            
            self.conversation_text.config(state=tk.DISABLED)
        
        # Append the token
        self.current_assistant_response += token
        
        # Update the displayed message
        self.conversation_text.config(state=tk.NORMAL)
        
        # Find the last assistant message and replace it
        last_pos = self.conversation_text.search(
            "Assistant: ", 
            "1.0", 
            tk.END, 
            backwards=True
        )
        
        if last_pos:
            # Get the line and column
            line, col = map(int, last_pos.split('.'))
            
            # Calculate position after "Assistant: "
            start_pos = f"{line}.{col + 11}"
            
            # Delete existing content
            self.conversation_text.delete(start_pos, tk.END)
            
            # Insert updated content
            self.conversation_text.insert(tk.END, self.current_assistant_response)
            
        # Scroll to bottom
        self.conversation_text.see(tk.END)
        
        # Disable editing
        self.conversation_text.config(state=tk.DISABLED)

    def finalize_assistant_response(self):
        """
        Finalize the current assistant response.
        """
        # Reset current response
        self.current_assistant_response = ""

    def clear_input(self):
        """
        Clear the text input field.
        """
        self.text_input.delete(0, tk.END)

    def clear_messages(self):
        """
        Clear the conversation display.
        """
        self.conversation_text.config(state=tk.NORMAL)
        self.conversation_text.delete(1.0, tk.END)
        self.conversation_text.config(state=tk.DISABLED)
        self.current_assistant_response = ""

    def _on_send_button(self):
        """
        Handle send button click.
        """
        text = self.text_input.get().strip()
        if text:
            # Call the callback
            self.on_send_text(text)

    def _on_voice_button(self):
        """
        Handle voice button click.
        """
        if self.is_recording:
            self.on_stop_recording()
        else:
            self.on_start_recording()

    def _on_clear_button(self):
        """
        Handle clear button click.
        """
        self.on_clear_history()

    def _on_enter_key(self, event):
        """
        Handle Enter key in text input.
        
        Args:
            event: Key event
        """
        self._on_send_button()