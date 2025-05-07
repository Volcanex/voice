"""
Controllable Speech Model (CSM) Module.

This module provides a wrapper around Sesame AI's CSM-1B model for
generating natural-sounding speech with appropriate prosody and pauses.
"""
import asyncio
import io
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, AsyncGenerator, Optional, Tuple, BinaryIO

import numpy as np
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModel

logger = logging.getLogger(__name__)

class CSMModule:
    """
    Controllable Speech Model Module for text-to-speech synthesis.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CSM module with the given configuration.
        
        Args:
            config: Dict containing CSM module configuration
        """
        self.config = config
        self.device = config["device"]
        self.model_id = config["model_id"]
        self.cache_dir = config["cache_dir"]
        self.quality = config.get("quality", "high")
        
        # Map quality setting to model parameters
        self.quality_settings = {
            "low": {"chunk_size": 20, "overlap": 5},
            "medium": {"chunk_size": 30, "overlap": 10},
            "high": {"chunk_size": 50, "overlap": 15}
        }
        
        # Get settings based on quality
        settings = self.quality_settings.get(
            self.quality, self.quality_settings["medium"]
        )
        self.chunk_size = settings["chunk_size"]
        self.overlap = settings["overlap"]
        
        self.processor = None
        self.model = None
        self.is_initialized = False
        
        # Audio settings
        self.sample_rate = 24000  # CSM uses 24kHz audio
        
        logger.info(f"CSM Module initialized with {self.model_id}")

    async def initialize(self) -> None:
        """
        Load the CSM model and processor asynchronously.
        """
        if self.is_initialized:
            return
            
        # Run model loading in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
        self.is_initialized = True
        logger.info(f"CSM model loaded on {self.device}")

    def _load_model(self) -> None:
        """
        Load the CSM model and processor.
        """
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                logger.warning("CUDA not available, using CPU for CSM")
                self.device = "cpu"
                
        except Exception as e:
            logger.error(f"Failed to load CSM model: {e}")
            raise

    async def synthesize_speech(
        self, text: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> bytes:
        """
        Synthesize speech from text, considering conversation history.
        
        Args:
            text: Text to synthesize
            conversation_history: Optional conversation history for context
            
        Returns:
            Synthesized speech as WAV bytes
        """
        if not self.is_initialized:
            await self.initialize()
            
        # Run synthesis in a separate thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._synthesize, text, conversation_history
        )

    def _synthesize(
        self, text: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> bytes:
        """
        Synthesize speech from text using the CSM model.
        
        Args:
            text: Text to synthesize
            conversation_history: Optional conversation history for context
            
        Returns:
            Synthesized speech as WAV bytes
        """
        try:
            # Process the input with conversation history if available
            if conversation_history:
                # Format conversation history for the CSM model
                history_text = self._format_conversation_history(conversation_history)
                inputs = self.processor(
                    text=text,
                    history_text=history_text,
                    return_tensors="pt"
                )
            else:
                inputs = self.processor(text=text, return_tensors="pt")
                
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            # Generate speech
            with torch.no_grad():
                speech_output = self.model.generate(**inputs)
                
            # Convert to numpy and then to WAV bytes
            speech_array = speech_output.cpu().numpy().squeeze()
            
            # Use a temporary file to write the WAV data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                sf.write(
                    tmp_file.name, 
                    speech_array, 
                    self.sample_rate, 
                    format='WAV', 
                    subtype='PCM_16'
                )
                tmp_file.flush()
                tmp_file.seek(0)
                wav_bytes = tmp_file.read()
                
            return wav_bytes
            
        except Exception as e:
            logger.error(f"Error during speech synthesis: {e}")
            # Return empty bytes on error
            return b""

    async def synthesize_speech_stream(
        self, 
        text: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized speech by processing the text in chunks.
        
        Args:
            text: Text to synthesize
            conversation_history: Optional conversation history for context
            
        Yields:
            Chunks of synthesized speech as WAV bytes
        """
        if not self.is_initialized:
            await self.initialize()
            
        # Split the text into sentences for streaming
        sentences = self._split_into_sentences(text)
        
        # Process each sentence or chunk of sentences
        chunk_text = ""
        for sentence in sentences:
            chunk_text += sentence
            
            # Process when we have enough text or at the last sentence
            if len(chunk_text.split()) >= self.chunk_size or sentence == sentences[-1]:
                # Run synthesis in a separate thread
                loop = asyncio.get_event_loop()
                audio_bytes = await loop.run_in_executor(
                    None, self._synthesize, chunk_text, conversation_history
                )
                
                # Yield the audio chunk
                if audio_bytes:
                    yield audio_bytes
                    
                # Reset with overlap
                words = chunk_text.split()
                if len(words) > self.overlap:
                    chunk_text = " ".join(words[-self.overlap:])
                else:
                    chunk_text = ""
                    
                # Short pause to allow other async tasks to run
                await asyncio.sleep(0.01)

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better synthesis.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting based on common punctuation
        import re
        
        # Define sentence ending punctuation
        sentence_end = r'(?<=[.!?])\s+'
        
        # Split the text
        sentences = re.split(sentence_end, text)
        
        # Make sure each sentence ends with punctuation
        result = []
        for sentence in sentences:
            if sentence:
                if not sentence[-1] in '.!?':
                    sentence += '.'
                result.append(sentence + ' ')
                
        return result

    def _format_conversation_history(
        self, conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Format conversation history for the CSM model.
        
        Args:
            conversation_history: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Formatted conversation history string
        """
        formatted_history = ""
        
        for message in conversation_history:
            if message["role"] == "user":
                formatted_history += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                formatted_history += f"Assistant: {message['content']}\n"
                
        return formatted_history.strip()

    async def shutdown(self) -> None:
        """
        Clean up resources used by the CSM module.
        """
        if not self.is_initialized:
            return
            
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.is_initialized = False
        logger.info("CSM module shutdown completed")