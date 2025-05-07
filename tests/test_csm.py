"""
Tests for the CSM module.
"""
import asyncio
import io
import tempfile
import pytest
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import torch

from server.modules.csm_module import CSMModule

@pytest.fixture
def csm_config():
    """Provide a test configuration for the CSM module."""
    return {
        "model_id": "sesame/csm-1b",
        "device": "cpu",  # Use CPU for testing
        "cache_dir": "./test_models/csm",
        "quality": "medium"
    }

@pytest.fixture
def mock_processor():
    """Create a mock CSM processor."""
    processor = MagicMock()
    processor.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    return processor

@pytest.fixture
def mock_model():
    """Create a mock CSM model."""
    model = MagicMock()
    model.generate.return_value = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    return model

@pytest.mark.asyncio
async def test_csm_initialization(csm_config):
    """Test CSM module initialization."""
    csm = CSMModule(csm_config)
    
    assert csm.model_id == csm_config["model_id"]
    assert csm.device == csm_config["device"]
    assert csm.cache_dir == csm_config["cache_dir"]
    assert csm.quality == csm_config["quality"]
    assert not csm.is_initialized
    
    # Check quality settings mapping
    assert csm.chunk_size == 30  # medium quality
    assert csm.overlap == 10  # medium quality

@pytest.mark.asyncio
@patch("server.modules.csm_module.AutoProcessor")
@patch("server.modules.csm_module.AutoModel")
async def test_csm_initialize(mock_model_class, mock_processor_class, csm_config):
    """Test CSM module model loading."""
    mock_processor_class.from_pretrained.return_value = MagicMock()
    mock_model_class.from_pretrained.return_value = MagicMock()
    
    csm = CSMModule(csm_config)
    await csm.initialize()
    
    assert csm.is_initialized
    mock_processor_class.from_pretrained.assert_called_once_with(
        csm_config["model_id"], cache_dir=csm_config["cache_dir"]
    )
    mock_model_class.from_pretrained.assert_called_once_with(
        csm_config["model_id"], cache_dir=csm_config["cache_dir"]
    )

@pytest.mark.asyncio
@patch("server.modules.csm_module.tempfile.NamedTemporaryFile")
@patch("server.modules.csm_module.sf")
async def test_synthesize_speech(
    mock_soundfile, mock_tempfile, csm_config, mock_processor, mock_model
):
    """Test speech synthesis."""
    csm = CSMModule(csm_config)
    csm.processor = mock_processor
    csm.model = mock_model
    csm.is_initialized = True
    
    # Set up temp file mock
    mock_temp_file = MagicMock()
    mock_temp_file.name = "test.wav"
    mock_temp_file.read.return_value = b"test audio data"
    mock_tempfile.return_value.__enter__.return_value = mock_temp_file
    
    # Test synthesis
    text = "This is a test sentence."
    audio_bytes = await csm.synthesize_speech(text)
    
    # Check that the model was called
    assert mock_model.generate.called
    assert mock_soundfile.write.called
    assert audio_bytes == b"test audio data"

@pytest.mark.asyncio
async def test_synthesize_speech_with_history(
    csm_config, mock_processor, mock_model
):
    """Test speech synthesis with conversation history."""
    with patch("server.modules.csm_module.tempfile.NamedTemporaryFile") as mock_tempfile, \
         patch("server.modules.csm_module.sf") as mock_soundfile:
        
        # Set up temp file mock
        mock_temp_file = MagicMock()
        mock_temp_file.name = "test.wav"
        mock_temp_file.read.return_value = b"test audio data"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file
        
        csm = CSMModule(csm_config)
        csm.processor = mock_processor
        csm.model = mock_model
        csm.is_initialized = True
        
        # Test with conversation history
        text = "This is a response."
        history = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "How can I help you?"},
            {"role": "user", "content": "Tell me something."}
        ]
        
        audio_bytes = await csm.synthesize_speech(text, history)
        
        # Check that the processor was called with history
        processor_calls = mock_processor.call_args_list
        assert len(processor_calls) == 1
        assert "history_text" in processor_calls[0][1]
        assert audio_bytes == b"test audio data"

@pytest.mark.asyncio
async def test_synthesize_speech_stream(
    csm_config, mock_processor, mock_model
):
    """Test streaming speech synthesis."""
    with patch("server.modules.csm_module.tempfile.NamedTemporaryFile") as mock_tempfile, \
         patch("server.modules.csm_module.sf") as mock_soundfile:
        
        # Set up temp file mock
        mock_temp_file = MagicMock()
        mock_temp_file.name = "test.wav"
        mock_temp_file.read.return_value = b"test audio data"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file
        
        # Create the CSM module with modified settings for testing
        csm = CSMModule(csm_config)
        csm.processor = mock_processor
        csm.model = mock_model
        csm.is_initialized = True
        
        # Modify the chunk_size to be very small to force multiple chunks in our test
        # This is crucial - the default value (30 for medium quality) is too large
        # for our test sentences to split into multiple chunks
        original_chunk_size = csm.chunk_size
        csm.chunk_size = 2  # Make it smaller than our test sentences
        
        # Track when sentences are processed
        processed_sentences = []
        
        # Mock _split_into_sentences to return test sentences
        def mock_split_sentences(text):
            sentences = ["This is sentence one. ", "This is sentence two. "]
            print(f"\nSplitting text into sentences: {sentences}")
            return sentences
            
        csm._split_into_sentences = mock_split_sentences
        
        # Fix: Replace the internal _synthesize method with our controlled mock
        original_synthesize = csm._synthesize
        
        # Counter to track calls to _synthesize and return unique data for each chunk
        call_count = 0
        
        def mock_synthesize(text, history=None):
            nonlocal call_count
            call_count += 1
            processed_sentences.append(text)
            print(f"Synthesizing chunk {call_count}: '{text}'")
            # Return different data for each call to verify we're getting multiple chunks
            return f"test audio data for chunk {call_count}".encode()
            
        # Replace with our mock
        csm._synthesize = mock_synthesize
        
        # Test streaming
        text = "This is sentence one. This is sentence two."
        
        try:
            chunks = []
            async for chunk in csm.synthesize_speech_stream(text):
                print(f"Received chunk: {chunk.decode()}")
                chunks.append(chunk)
            
            print(f"\nProcessed sentences: {processed_sentences}")
            print(f"Streamed audio chunks: {len(chunks)}")
            print(f"Chunks: {[chunk.decode() for chunk in chunks]}")
            
            # Should have 2 chunks for 2 sentences
            assert len(chunks) == 2, f"Expected 2 chunks but got {len(chunks)}"
            assert chunks[0] == b"test audio data for chunk 1"
            assert chunks[1] == b"test audio data for chunk 2"
            
        finally:
            # Restore original implementation and settings
            csm._synthesize = original_synthesize
            csm.chunk_size = original_chunk_size

@pytest.mark.asyncio
async def test_format_conversation_history(csm_config):
    """Test conversation history formatting."""
    csm = CSMModule(csm_config)
    
    history = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "How can I help you?"},
        {"role": "user", "content": "Tell me something."},
        {"role": "system", "content": "This should be ignored."}
    ]
    
    formatted = csm._format_conversation_history(history)
    
    assert "User: Hello!" in formatted
    assert "Assistant: How can I help you?" in formatted
    assert "User: Tell me something." in formatted
    assert "system" not in formatted.lower()

@pytest.mark.asyncio
async def test_split_into_sentences(csm_config):
    """Test sentence splitting."""
    csm = CSMModule(csm_config)
    
    text = "Hello! This is a test. How are you doing? This is great."
    sentences = csm._split_into_sentences(text)
    
    assert len(sentences) == 4
    assert sentences[0] == "Hello! "
    assert sentences[1] == "This is a test. "
    assert sentences[2] == "How are you doing? "
    assert sentences[3] == "This is great. "