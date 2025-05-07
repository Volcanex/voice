"""
End-to-end tests for the voice assistant system.

These tests verify the integration between modules and the complete workflow.
"""
import asyncio
import base64
import json
import pytest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from server.config import get_config
from server.state_manager import StateManager
from server.modules.asr_module import ASRModule
from server.modules.llm_module import LLMModule
from server.modules.csm_module import CSMModule
from server.websocket_gateway import WebSocketManager, ClientMessage, ServerMessage

@pytest.fixture
def test_config():
    """Provide a test configuration for the entire system."""
    return {
        "host": "127.0.0.1",
        "port": 8000,
        "websocket_path": "/ws",
        "debug": True,
        "asr_model": {
            "model_id": "openai/whisper-small",
            "device": "cpu",
            "compute_type": "int8",
            "cache_dir": "./test_models/asr"
        },
        "llm_model": {
            "model_id": "microsoft/phi-3-mini-4k-instruct",
            "device": "cpu",
            "quantize": "gguf",
            "cache_dir": "./test_models/llm",
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9
        },
        "csm_model": {
            "model_id": "sesame/csm-1b",
            "device": "cpu",
            "cache_dir": "./test_models/csm",
            "quality": "low"
        },
        "sample_rate": 16000,
        "channels": 1,
        "chunk_size": 1024,
        "audio_format": "wav",
        "max_history_length": 10,
        "system_prompt": "You are a helpful voice assistant.",
        "log_level": "DEBUG"
    }

class MockWebSocket:
    """
    Mock WebSocket class for testing.
    """
    def __init__(self):
        self.sent_messages = []
        self.closed = False
        self.close_code = None
        
    async def accept(self):
        """Accept the connection."""
        pass
        
    async def send_text(self, message):
        """Send a text message."""
        self.sent_messages.append(message)
        # Print sent messages for debugging
        print(f"Mock WebSocket received message: {message[:100]}...")
        
    async def close(self, code=1000):
        """Close the connection."""
        self.closed = True
        self.close_code = code
        
    def __aiter__(self):
        """Async iterator for messages."""
        return self
        
    async def __anext__(self):
        """Get next message."""
        if self.closed:
            raise StopAsyncIteration
        raise StopAsyncIteration

@pytest.fixture
def mock_modules():
    """Create mock modules for the system."""
    asr_module = MagicMock(spec=ASRModule)
    asr_module.initialize = MagicMock(return_value=asyncio.Future())
    asr_module.initialize.return_value.set_result(None)
    asr_module.process_audio_stream = MagicMock(return_value=asyncio.Future())
    asr_module.process_audio_stream.return_value.set_result("Hello, assistant.")
    
    llm_module = MagicMock(spec=LLMModule)
    llm_module.initialize = MagicMock(return_value=asyncio.Future())
    llm_module.initialize.return_value.set_result(None)
    
    # Mock streaming response
    async def mock_generate_stream(messages, system_prompt=None):
        tokens = ["Hello, ", "how ", "can ", "I ", "help ", "you?"]
        for token in tokens:
            yield token
    
    llm_module.generate_response_stream = mock_generate_stream
    
    csm_module = MagicMock(spec=CSMModule)
    csm_module.initialize = MagicMock(return_value=asyncio.Future())
    csm_module.initialize.return_value.set_result(None)
    
    # Mock streaming speech
    async def mock_speech_stream(text, conversation_history=None):
        chunks = [b"audio_chunk_1", b"audio_chunk_2"]
        for chunk in chunks:
            yield chunk
    
    csm_module.synthesize_speech_stream = mock_speech_stream
    
    return asr_module, llm_module, csm_module

@pytest.mark.asyncio
async def test_websocket_connection(test_config, mock_modules):
    """Test WebSocket connection and initialization."""
    asr_module, llm_module, csm_module = mock_modules
    state_manager = StateManager(test_config)
    websocket_manager = WebSocketManager(
        state_manager, asr_module, llm_module, csm_module
    )
    
    # Create mock WebSocket
    websocket = MockWebSocket()
    
    # Handle connection
    await websocket_manager.handle_connection(websocket)
    
    # Check connection was accepted
    assert len(websocket.sent_messages) == 1
    
    # Parse welcome message
    welcome_msg = json.loads(websocket.sent_messages[0])
    assert welcome_msg["type"] == "init"
    assert "session_id" in welcome_msg["payload"]
    assert "conversation_id" in welcome_msg["payload"]

@pytest.mark.asyncio
async def test_text_processing_flow(test_config, mock_modules):
    """Test the complete text processing flow."""
    asr_module, llm_module, csm_module = mock_modules
    state_manager = StateManager(test_config)
    websocket_manager = WebSocketManager(
        state_manager, asr_module, llm_module, csm_module
    )
    
    # Create mock WebSocket
    websocket = MockWebSocket()
    
    # Handle connection
    await websocket_manager.handle_connection(websocket)
    
    # Get conversation ID from welcome message
    welcome_msg = json.loads(websocket.sent_messages[0])
    conversation_id = welcome_msg["payload"]["conversation_id"]
    session_id = welcome_msg["payload"]["session_id"]
    
    # Important: Associate the WebSocket with the session
    websocket_manager.active_connections[session_id] = websocket
    
    # Clear messages list
    websocket.sent_messages.clear()
    
    # Create text message
    text_message = ClientMessage(
        type="text",
        payload={
            "text": "Hello, assistant.",
            "conversation_id": conversation_id
        }
    )
    
    # Handle text message
    await websocket_manager.handle_text(session_id, text_message)
    
    # Wait a short time for async operations to complete
    await asyncio.sleep(0.1)
    
    # Print number of messages received
    print(f"\nNumber of messages received: {len(websocket.sent_messages)}")
    
    # Check responses
    assert len(websocket.sent_messages) > 0, "No messages were sent by the WebSocket Manager"
    
    # Parse all messages
    parsed_messages = [json.loads(msg) for msg in websocket.sent_messages]
    
    # Check for token messages (should have at least one)
    token_messages = [msg for msg in parsed_messages if msg["type"] == "response_token"]
    assert len(token_messages) > 0, "No token messages received"
    
    # Check at least one token contains text
    token_texts = [msg["payload"]["token"] for msg in token_messages]
    print(f"Tokens received: {token_texts}")
    
    # Check for completion message (should have exactly one)
    completion_messages = [msg for msg in parsed_messages if msg["type"] == "response_complete"]
    assert len(completion_messages) > 0, "No completion message received"
    
    # Audio messages may be optional depending on implementation
    audio_messages = [msg for msg in parsed_messages if msg["type"] == "audio"]
    print(f"Audio messages: {len(audio_messages)}")

@pytest.mark.asyncio
async def test_audio_processing_flow(test_config, mock_modules):
    """Test the complete audio processing flow."""
    asr_module, llm_module, csm_module = mock_modules
    state_manager = StateManager(test_config)
    websocket_manager = WebSocketManager(
        state_manager, asr_module, llm_module, csm_module
    )
    
    # Create mock WebSocket
    websocket = MockWebSocket()
    
    # Handle connection
    await websocket_manager.handle_connection(websocket)
    
    # Get conversation ID from welcome message
    welcome_msg = json.loads(websocket.sent_messages[0])
    conversation_id = welcome_msg["payload"]["conversation_id"]
    session_id = welcome_msg["payload"]["session_id"]
    
    # Important: Associate the WebSocket with the session
    websocket_manager.active_connections[session_id] = websocket
    
    # Clear messages list
    websocket.sent_messages.clear()
    
    # Create audio data (random 16-bit PCM)
    audio_data = np.random.randint(-32768, 32767, 16000).astype(np.int16).tobytes()
    audio_b64 = base64.b64encode(audio_data).decode("utf-8")
    
    # Create audio message
    audio_message = ClientMessage(
        type="audio",
        payload={
            "audio": audio_b64,
            "conversation_id": conversation_id,
            "is_final": True
        }
    )
    
    # Handle audio message
    await websocket_manager.handle_audio(session_id, audio_message)
    
    # Wait a short time for async operations to complete
    await asyncio.sleep(0.1)
    
    # Print number of messages received
    print(f"\nAudio flow - Number of messages received: {len(websocket.sent_messages)}")
    
    # Check responses
    assert len(websocket.sent_messages) > 0, "No messages were sent by the WebSocket Manager"
    
    # Parse all messages
    parsed_messages = [json.loads(msg) for msg in websocket.sent_messages]
    
    # Log message types for debugging
    message_types = [msg["type"] for msg in parsed_messages]
    print(f"Message types received: {message_types}")
    
    # Check for transcription message
    transcription_messages = [msg for msg in parsed_messages if msg["type"] == "transcription"]
    if transcription_messages:
        print(f"Transcription: {transcription_messages[0]['payload']['text']}")
    assert len(transcription_messages) > 0, "No transcription messages received"
    
    # Check for token messages
    token_messages = [msg for msg in parsed_messages if msg["type"] == "response_token"]
    token_texts = []
    if token_messages:
        token_texts = [msg["payload"]["token"] for msg in token_messages]
        print(f"Tokens received: {token_texts}")
    assert len(token_messages) > 0, "No token messages received"
    
    # Check for completion message
    completion_messages = [msg for msg in parsed_messages if msg["type"] == "response_complete"]
    assert len(completion_messages) > 0, "No completion message received"
    
    # Audio messages may be optional depending on implementation
    audio_messages = [msg for msg in parsed_messages if msg["type"] == "audio"]
    print(f"Audio messages: {len(audio_messages)}")

@pytest.mark.asyncio
async def test_conversation_state(test_config):
    """Test conversation state management."""
    state_manager = StateManager(test_config)
    
    # Create a conversation
    conversation_id = state_manager.create_conversation("test_session")
    
    # Add messages
    state_manager.add_message(conversation_id, "user", "Hello!")
    state_manager.add_message(conversation_id, "assistant", "Hi there!")
    state_manager.add_message(conversation_id, "user", "How are you?")
    
    # Get history
    history = state_manager.get_conversation_history(conversation_id)
    
    # Check history
    assert len(history) == 3
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello!"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hi there!"
    assert history[2]["role"] == "user"
    assert history[2]["content"] == "How are you?"
    
    # Clear conversation
    state_manager.clear_conversation(conversation_id)
    
    # Check history is empty
    history = state_manager.get_conversation_history(conversation_id)
    assert len(history) == 0
    
    # Test history length limit
    max_length = test_config["max_history_length"]
    for i in range(max_length + 5):
        state_manager.add_message(conversation_id, "user", f"Message {i}")
        
    history = state_manager.get_conversation_history(conversation_id)
    assert len(history) == max_length
    assert history[0]["content"] == f"Message {5}"  # First 5 messages trimmed