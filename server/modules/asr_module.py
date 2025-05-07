"""
Speech Recognition Module using Whisper Model.

This module provides a wrapper around the OpenAI Whisper model for 
automatic speech recognition, optimized for streaming audio input.
"""
import asyncio
import io
import logging
from typing import Dict, Any, Optional, List, Tuple, AsyncGenerator

import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

logger = logging.getLogger(__name__)

class ASRModule:
    """
    Speech Recognition Module using Whisper model with streaming capabilities.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ASR module with the given configuration.
        
        Args:
            config: Dict containing ASR module configuration
        """
        self.config = config
        self.device = config["device"]
        self.model_id = config["model_id"]
        self.cache_dir = config["cache_dir"]
        self.compute_type = config["compute_type"]
        
        self.processor = None
        self.model = None
        self.is_initialized = False
        
        # Streaming settings
        self.sample_rate = 16000  # Whisper requires 16kHz audio
        self.chunk_duration = 2.0  # Process audio in 2-second chunks
        self.stride_duration = 1.0  # Overlap chunks by 1 second
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        self.stride_samples = int(self.stride_duration * self.sample_rate)
        
        # Buffer for streaming audio
        self.audio_buffer = np.array([], dtype=np.float32)
        
        logger.info(f"ASR Module initialized with {self.model_id}")

    async def initialize(self) -> None:
        """
        Load the ASR model and processor asynchronously.
        """
        if self.is_initialized:
            return
            
        # Run model loading in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
        self.is_initialized = True
        logger.info(f"ASR model loaded on {self.device}")

    def _load_model(self) -> None:
        """
        Load the Whisper model and processor.
        """
        try:
            self.processor = WhisperProcessor.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id, 
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.compute_type == "float16" else torch.float32,
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                logger.warning("CUDA not available, using CPU for ASR")
                self.device = "cpu"
                
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            raise

    async def process_audio(self, audio_data: bytes) -> str:
        """
        Process a complete audio file and return the transcription.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Transcribed text
        """
        if not self.is_initialized:
            await self.initialize()
            
        # Run transcription in a separate thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_audio, audio_data)

    def _transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe a complete audio file.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Transcribed text
        """
        try:
            # Convert bytes to numpy array
            audio_np = self._bytes_to_array(audio_data)
            
            # Process with Whisper
            input_features = self.processor(
                audio_np, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).input_features
            
            if self.device == "cuda":
                input_features = input_features.to("cuda")
                
            # Generate token ids
            predicted_ids = self.model.generate(input_features)
            
            # Decode the token ids to text
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""

    async def process_audio_stream(
        self, audio_chunk: bytes, is_final: bool = False
    ) -> Optional[str]:
        """
        Process a chunk of streaming audio and return partial transcriptions.
        
        Args:
            audio_chunk: Raw audio bytes for the current chunk
            is_final: Whether this is the final chunk in the stream
            
        Returns:
            Partial transcription or None if not enough audio
        """
        if not self.is_initialized:
            await self.initialize()
            
        # Convert bytes to numpy array and add to buffer
        chunk_np = self._bytes_to_array(audio_chunk)
        self.audio_buffer = np.append(self.audio_buffer, chunk_np)
        
        # If we don't have enough audio yet and this isn't the final chunk, return None
        if len(self.audio_buffer) < self.chunk_samples and not is_final:
            return None
            
        # Process the buffer
        loop = asyncio.get_event_loop()
        transcription = await loop.run_in_executor(None, self._process_buffer, is_final)
        
        return transcription

    def _process_buffer(self, is_final: bool) -> Optional[str]:
        """
        Process the current audio buffer and return a transcription.
        
        Args:
            is_final: Whether this is the final chunk in the stream
            
        Returns:
            Partial transcription or None
        """
        # If final chunk or buffer large enough
        if is_final or len(self.audio_buffer) >= self.chunk_samples:
            # Take a chunk from the buffer
            if is_final:
                # Use all remaining audio if this is the final chunk
                audio_chunk = self.audio_buffer
                self.audio_buffer = np.array([], dtype=np.float32)
            else:
                # Use a fixed-size chunk and keep the overlap in the buffer
                audio_chunk = self.audio_buffer[:self.chunk_samples]
                self.audio_buffer = self.audio_buffer[self.stride_samples:]
            
            # Process with Whisper
            input_features = self.processor(
                audio_chunk, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).input_features
            
            if self.device == "cuda":
                input_features = input_features.to("cuda")
                
            # Generate token ids
            predicted_ids = self.model.generate(input_features)
            
            # Decode the token ids to text
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
            
            return transcription
            
        return None

    def _bytes_to_array(self, audio_bytes: bytes) -> np.ndarray:
        """
        Convert raw audio bytes to a numpy array suitable for Whisper.
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            Audio as numpy array with values in [-1, 1]
        """
        # Convert bytes to numpy array (assumes 16-bit PCM)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        
        # Normalize to [-1, 1]
        audio_np = audio_np / 32768.0
        
        return audio_np

    def reset_stream(self) -> None:
        """
        Reset the streaming state.
        """
        self.audio_buffer = np.array([], dtype=np.float32)

    async def shutdown(self) -> None:
        """
        Clean up resources used by the ASR module.
        """
        if not self.is_initialized:
            return
            
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.is_initialized = False
        logger.info("ASR module shutdown completed")