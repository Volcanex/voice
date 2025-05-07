"""
Language Model Module using Phi-3-mini.

This module provides a wrapper around Microsoft's Phi-3-mini model
for generating text responses in a conversational context.
"""
import asyncio
import importlib.util
import logging
import sys
from typing import Dict, Any, List, AsyncGenerator, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread

# Check for optional dependencies
def is_package_available(package_name):
    """Check if a package is available without importing it."""
    return importlib.util.find_spec(package_name) is not None

# Check for quantization and acceleration libraries
HAS_BITSANDBYTES = is_package_available("bitsandbytes")
HAS_FLASH_ATTN = is_package_available("flash_attn")

logger = logging.getLogger(__name__)

class LLMModule:
    """
    Language Model Module using Phi-3-mini with streaming capabilities.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM module with the given configuration.
        
        Args:
            config: Dict containing LLM module configuration
        """
        self.config = config
        self.device = config["device"]
        self.model_id = config["model_id"]
        self.cache_dir = config["cache_dir"]
        self.quantize = config["quantize"]
        
        # Memory management options
        self.low_memory = config.get("low_memory", False)
        self.device_map = config.get("device_map")
        
        # Generation parameters
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        
        self.tokenizer = None
        self.model = None
        self.is_initialized = False
        
        # Conversation history format
        self.message_template = {
            "user": "<|user|>\n{message}\n",
            "assistant": "<|assistant|>\n{message}\n",
            "system": "<|system|>\n{message}\n"
        }
        
        logger.info(f"LLM Module initialized with {self.model_id}")

    async def initialize(self) -> None:
        """
        Load the LLM model and tokenizer asynchronously.
        """
        if self.is_initialized:
            return
            
        # Run model loading in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
        self.is_initialized = True
        logger.info(f"LLM model loaded on {self.device}")

    def _load_model(self) -> None:
        """
        Load the model and tokenizer.
        """
        # Add memory usage logging
        try:
            import resource
            import psutil
            has_monitoring = True
            process = psutil.Process()
        except ImportError:
            has_monitoring = False
            logger.warning("psutil not installed, memory monitoring disabled")
        
        # Log initial memory usage
        if has_monitoring:
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Memory usage before model loading: {mem_before:.2f} MB")
            
            # Report system memory
            virtual_memory = psutil.virtual_memory()
            logger.info(f"System memory: {virtual_memory.total / 1024 / 1024:.2f} MB total, "
                        f"{virtual_memory.available / 1024 / 1024:.2f} MB available")
            
            # Check if CUDA is available and get GPU memory info
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024  # MB
                    logger.info(f"GPU {i} total memory: {gpu_mem:.2f} MB")
        
        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Set model loading options based on device and quantization
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
            }
            
            # Determine if we can use GPU
            use_gpu = self.device == "cuda" and torch.cuda.is_available()
            
            # Apply low memory mode settings
            if self.low_memory:
                logger.info("Low memory mode is enabled")
                model_kwargs["low_cpu_mem_usage"] = True
                
                # Use device map if provided
                if self.device_map:
                    logger.info(f"Using device map: {self.device_map}")
                    model_kwargs["device_map"] = self.device_map
            
            # Add quantization options if dependencies are available
            if use_gpu and self.quantize == "gptq":
                if HAS_BITSANDBYTES and not self.low_memory:
                    logger.info("Using 4-bit quantization with bitsandbytes")
                    model_kwargs["load_in_4bit"] = True
                    model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                else:
                    logger.warning("bitsandbytes not available or low memory mode, falling back to standard loading")
            elif self.quantize == "gguf" and HAS_BITSANDBYTES:
                logger.info("Using 8-bit quantization for CPU")
                model_kwargs["load_in_8bit"] = True
            else:
                logger.info("Quantization not available or not requested")
                
            # Handle flash attention
            if use_gpu and HAS_FLASH_ATTN:
                logger.info("Using Flash Attention for faster inference")
                model_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                logger.info("Flash Attention not available, using eager implementation")
                model_kwargs["attn_implementation"] = "eager"
                
            # Set low_cpu_mem_usage to True to reduce memory overhead (always a good idea)
            model_kwargs["low_cpu_mem_usage"] = True
            
            # Enable offloading to disk for large models
            model_kwargs["offload_folder"] = self.cache_dir
                
            # Load model
            logger.info(f"Loading LLM with options: {model_kwargs}")
            
            # Log memory usage before model loading
            if has_monitoring:
                mem_before_load = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(f"Memory usage before model loading: {mem_before_load:.2f} MB")
            
            # Set a conservative memory limit for loading - if exceeded, retry with more options
            memory_limited = False
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    **model_kwargs
                )
            except (RuntimeError, MemoryError) as mem_error:
                logger.warning(f"Memory error during model loading: {mem_error}")
                logger.warning("Trying again with device_map='auto' to spread across CPU/GPU")
                memory_limited = True
                
                # Add auto device map for memory-efficient loading
                model_kwargs["device_map"] = "auto"
                model_kwargs["max_memory"] = None  # Let the library determine optimal allocation
                
                if has_monitoring:
                    # Force garbage collection
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Try loading with auto device map
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    **model_kwargs
                )
            
            # Log memory usage after model loading
            if has_monitoring:
                mem_after_load = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(f"Memory usage after model loading: {mem_after_load:.2f} MB")
                if mem_before_load > 0:
                    logger.info(f"Model loading used approximately {mem_after_load - mem_before_load:.2f} MB")
            
            # Move to device if not using device_map='auto' and not memory limited
            if use_gpu and not memory_limited:
                if self.quantize != "gptq" or not HAS_BITSANDBYTES:
                    logger.info("Moving model to CUDA")
                    self.model = self.model.to("cuda")
            elif not memory_limited:
                logger.warning("CUDA not available, using CPU for LLM")
                self.device = "cpu"
                
            logger.info("LLM model loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            
            # Log memory at time of failure
            if has_monitoring:
                mem_at_failure = process.memory_info().rss / 1024 / 1024  # MB
                logger.error(f"Memory usage at failure: {mem_at_failure:.2f} MB")
                
                # Check if it's likely an OOM issue
                if mem_at_failure > virtual_memory.total * 0.8 / 1024 / 1024:  # If using >80% of RAM
                    logger.error("Memory usage is very high - likely an out-of-memory condition")
                
            # Continue with reduced functionality rather than crashing
            if "No package metadata was found for bitsandbytes" in str(e):
                logger.warning("Attempting to load model without quantization")
                try:
                    # Try again without quantization, with minimal memory footprint
                    logger.info("Retrying with minimal config and device_map='auto'")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True,
                        attn_implementation="eager",  # Always use eager as fallback
                        device_map="auto",           # Distribute across devices
                        low_cpu_mem_usage=True,      # Minimize memory usage
                        offload_folder=self.cache_dir # Allow offloading
                    )
                    
                    logger.info("Successfully loaded model without quantization")
                    return
                except Exception as inner_e:
                    logger.error(f"Failed to load model without quantization: {inner_e}")
            
            logger.error("Exhausted all loading options, model initialization failed")
            
            # If we can't load the model at all, raise the exception
            raise

    def format_conversation(
        self, messages: List[Dict[str, str]], system_prompt: str = None
    ) -> str:
        """
        Format the conversation history for the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt to include
            
        Returns:
            Formatted conversation string
        """
        conversation = ""
        
        # Add system prompt if provided
        if system_prompt:
            conversation += self.message_template["system"].format(message=system_prompt)
        
        # Add conversation history
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role in self.message_template:
                conversation += self.message_template[role].format(message=content)
            else:
                logger.warning(f"Unknown role: {role}")
                
        # Add assistant prefix for the response
        conversation += "<|assistant|>\n"
        
        return conversation

    async def generate_response(
        self, messages: List[Dict[str, str]], system_prompt: str = None
    ) -> str:
        """
        Generate a complete response for the given conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt to include
            
        Returns:
            Generated response text
        """
        if not self.is_initialized:
            await self.initialize()
            
        # Format conversation
        conversation = self.format_conversation(messages, system_prompt)
        
        # Run generation in a separate thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate, conversation)

    def _generate(self, prompt: str) -> str:
        """
        Generate a complete response for the given prompt.
        
        Args:
            prompt: Formatted conversation prompt
            
        Returns:
            Generated response text
        """
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                )
                
            # Decode the generated tokens, skipping the prompt
            prompt_length = inputs["input_ids"].shape[1]
            generated_text = self.tokenizer.decode(
                outputs[0][prompt_length:], skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return "I encountered an error while processing your request."

    async def generate_response_stream(
        self, messages: List[Dict[str, str]], system_prompt: str = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream the generated response token by token.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt to include
            
        Yields:
            Generated text tokens as they become available
        """
        if not self.is_initialized:
            await self.initialize()
            
        # Format conversation
        conversation = self.format_conversation(messages, system_prompt)
        
        # Tokenize the prompt
        inputs = self.tokenizer(conversation, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        # Create a streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Set up generation parameters
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
        }
        
        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they become available
        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            yield new_text
            
            # Short pause to allow other async tasks to run
            await asyncio.sleep(0.01)
            
        thread.join()

    async def shutdown(self) -> None:
        """
        Clean up resources used by the LLM module.
        """
        if not self.is_initialized:
            return
            
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.is_initialized = False
        logger.info("LLM module shutdown completed")