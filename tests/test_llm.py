"""
Tests for the LLM module.
"""
import asyncio
import importlib
import pytest
import sys
from unittest.mock import MagicMock, patch

import torch

# Import after adding the mock
from server.modules.llm_module import LLMModule

@pytest.fixture
def llm_config():
    """Provide a test configuration for the LLM module."""
    return {
        "model_id": "microsoft/phi-3-mini-4k-instruct",
        "device": "cpu",  # Use CPU for testing
        "quantize": "gguf",
        "cache_dir": "./test_models/llm",
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9
    }

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    tokenizer.decode.return_value = "This is a test response"
    return tokenizer

@pytest.fixture
def mock_model():
    """Create a mock language model."""
    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    return model

@pytest.fixture
def mock_streamer():
    """Create a mock text streamer."""
    streamer = MagicMock()
    streamer.__iter__.return_value = ["This ", "is ", "a ", "test ", "response"]
    return streamer

@pytest.mark.asyncio
async def test_llm_initialization(llm_config):
    """Test LLM module initialization."""
    llm = LLMModule(llm_config)
    
    assert llm.model_id == llm_config["model_id"]
    assert llm.device == llm_config["device"]
    assert llm.cache_dir == llm_config["cache_dir"]
    assert llm.quantize == llm_config["quantize"]
    assert llm.max_new_tokens == llm_config["max_new_tokens"]
    assert llm.temperature == llm_config["temperature"]
    assert llm.top_p == llm_config["top_p"]
    assert not llm.is_initialized

@pytest.mark.asyncio
@patch("server.modules.llm_module.AutoTokenizer")
@patch("server.modules.llm_module.AutoModelForCausalLM")
async def test_llm_initialize(mock_model_class, mock_tokenizer_class, llm_config):
    """Test LLM module model loading."""
    mock_tokenizer_class.from_pretrained.return_value = MagicMock()
    mock_model_class.from_pretrained.return_value = MagicMock()
    
    llm = LLMModule(llm_config)
    await llm.initialize()
    
    assert llm.is_initialized
    mock_tokenizer_class.from_pretrained.assert_called_once_with(
        llm_config["model_id"], 
        cache_dir=llm_config["cache_dir"],
        trust_remote_code=True
    )
    mock_model_class.from_pretrained.assert_called_once()

@pytest.mark.asyncio
async def test_format_conversation(llm_config):
    """Test conversation formatting."""
    llm = LLMModule(llm_config)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "How can I help you?"},
        {"role": "user", "content": "What's the weather like?"}
    ]
    
    formatted = llm.format_conversation(messages)
    
    assert "<|system|>" in formatted
    assert "<|user|>" in formatted
    assert "<|assistant|>" in formatted
    assert "You are a helpful assistant" in formatted
    assert "Hello!" in formatted
    assert "How can I help you?" in formatted
    assert "What's the weather like?" in formatted

@pytest.mark.asyncio
async def test_generate_response(llm_config, mock_tokenizer, mock_model):
    """Test response generation."""
    llm = LLMModule(llm_config)
    llm.tokenizer = mock_tokenizer
    llm.model = mock_model
    llm.is_initialized = True
    
    messages = [
        {"role": "user", "content": "Hello!"}
    ]
    
    response = await llm.generate_response(messages, "You are a helpful assistant.")
    
    # Check that the model was called
    assert mock_model.generate.called
    assert isinstance(response, str)
    assert response == "This is a test response"

@pytest.mark.asyncio
@patch("server.modules.llm_module.TextIteratorStreamer")
@patch("server.modules.llm_module.Thread")
async def test_generate_response_stream(
    mock_thread, mock_streamer_class, llm_config, mock_tokenizer, mock_model, mock_streamer
):
    """Test streaming response generation."""
    llm = LLMModule(llm_config)
    llm.tokenizer = mock_tokenizer
    llm.model = mock_model
    llm.is_initialized = True
    
    # Set up streamer mock
    mock_streamer_class.return_value = mock_streamer
    
    messages = [
        {"role": "user", "content": "Hello!"}
    ]
    
    # Test streaming
    collected_tokens = []
    async for token in llm.generate_response_stream(messages, "You are a helpful assistant."):
        collected_tokens.append(token)
    
    # Check that the model was called via thread
    assert mock_thread.called
    assert collected_tokens == ["This ", "is ", "a ", "test ", "response"]

@pytest.mark.asyncio
@patch("server.modules.llm_module.is_package_available")
@patch("server.modules.llm_module.AutoTokenizer")
@patch("server.modules.llm_module.AutoModelForCausalLM")
async def test_missing_dependencies_fallback(
    mock_model_class, mock_tokenizer_class, mock_is_package_available, llm_config
):
    """Test LLM module gracefully handles missing dependencies."""
    # Mock that dependencies are missing
    def is_package_mock(package_name):
        # Return False for bitsandbytes and flash_attn to simulate missing dependencies
        if package_name in ["bitsandbytes", "flash_attn"]:
            return False
        return True
    
    mock_is_package_available.side_effect = is_package_mock
    
    # Mock tokenizer and model
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    
    # First attempt raises the bitsandbytes import error, second succeeds
    mock_model_class.from_pretrained.side_effect = [
        ImportError("No package metadata was found for bitsandbytes"),
        mock_model
    ]
    
    # Run the test
    llm = LLMModule(llm_config)
    await llm.initialize()
    
    # Verify the model was loaded correctly despite missing dependencies
    assert llm.is_initialized
    assert llm.model == mock_model
    
    # Verify it attempted loading twice - first with quantization, then without
    assert mock_model_class.from_pretrained.call_count == 2
    
    # Check the second call used eager attention
    second_call_args = mock_model_class.from_pretrained.call_args_list[1][1]
    assert "attn_implementation" in second_call_args
    assert second_call_args["attn_implementation"] == "eager"