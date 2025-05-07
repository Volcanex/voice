"""
Tests for the ASR module.
"""
import asyncio
import os
import pytest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from server.modules.asr_module import ASRModule

@pytest.fixture
def asr_config():
    """Provide a test configuration for the ASR module."""
    return {
        "model_id": "openai/whisper-small",
        "device": "cpu",  # Use CPU for testing
        "compute_type": "int8",
        "cache_dir": "./test_models/asr"
    }

@pytest.fixture
def mock_processor():
    """Create a mock Whisper processor."""
    processor = MagicMock()
    processor.return_value.input_features = torch.rand(1, 80, 3000)
    processor.batch_decode.return_value = ["This is a test transcription"]
    return processor

@pytest.fixture
def mock_model():
    """Create a mock Whisper model."""
    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    return model

@pytest.mark.asyncio
async def test_asr_initialization(asr_config):
    """Test ASR module initialization."""
    asr = ASRModule(asr_config)
    
    assert asr.model_id == asr_config["model_id"]
    assert asr.device == asr_config["device"]
    assert asr.cache_dir == asr_config["cache_dir"]
    assert asr.compute_type == asr_config["compute_type"]
    assert not asr.is_initialized

@pytest.mark.asyncio
@patch("server.modules.asr_module.WhisperProcessor")
@patch("server.modules.asr_module.WhisperForConditionalGeneration")
async def test_asr_initialize(mock_whisper_model, mock_whisper_processor, asr_config):
    """Test ASR module model loading."""
    mock_whisper_processor.from_pretrained.return_value = MagicMock()
    mock_whisper_model.from_pretrained.return_value = MagicMock()
    
    asr = ASRModule(asr_config)
    await asr.initialize()
    
    assert asr.is_initialized
    mock_whisper_processor.from_pretrained.assert_called_once_with(
        asr_config["model_id"], cache_dir=asr_config["cache_dir"]
    )
    mock_whisper_model.from_pretrained.assert_called_once()

@pytest.mark.asyncio
async def test_process_audio(asr_config, mock_processor, mock_model):
    """Test audio processing."""
    asr = ASRModule(asr_config)
    asr.processor = mock_processor
    asr.model = mock_model
    asr.is_initialized = True
    
    # Create some dummy audio data
    audio_data = np.random.randint(-32768, 32767, 16000).astype(np.int16).tobytes()
    
    # Process the audio
    transcription = await asr.process_audio(audio_data)
    
    # Check that the model was called
    assert mock_model.generate.called
    assert isinstance(transcription, str)

@pytest.mark.asyncio
async def test_process_audio_stream(asr_config, mock_processor, mock_model):
    """Test streaming audio processing."""
    asr = ASRModule(asr_config)
    asr.processor = mock_processor
    asr.model = mock_model
    asr.is_initialized = True
    
    # Create some dummy audio data
    audio_data = np.random.randint(-32768, 32767, 32000).astype(np.int16).tobytes()
    
    # First chunk, not final
    transcription = await asr.process_audio_stream(audio_data, False)
    assert transcription is not None
    
    # Second chunk, final
    transcription = await asr.process_audio_stream(audio_data, True)
    assert transcription is not None
    
    # Check that the model was called
    assert mock_model.generate.called

@pytest.mark.asyncio
async def test_reset_stream(asr_config):
    """Test stream reset."""
    asr = ASRModule(asr_config)
    
    # Add some data to the buffer
    asr.audio_buffer = np.ones(1000, dtype=np.float32)
    
    # Reset the stream
    asr.reset_stream()
    
    # Check that the buffer is empty
    assert len(asr.audio_buffer) == 0