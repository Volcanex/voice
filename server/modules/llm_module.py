"""
Language Model Module using Phi-3-mini.

This module provides a wrapper around Microsoft's Phi-3-mini model
for generating text responses in a conversational context.
"""
import asyncio
import logging
from typing import Dict, Any, List, AsyncGenerator, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread

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
        try:
            # Load tokenizer
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
            
            # Add quantization options if needed
            if self.quantize == "gptq" and self.device == "cuda":
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            elif self.quantize == "gguf" and self.device == "cpu":
                model_kwargs["load_in_8bit"] = True
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                **model_kwargs
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                if self.quantize != "gptq":  # Only move to GPU if not already quantized
                    self.model = self.model.to("cuda")
            else:
                logger.warning("CUDA not available, using CPU for LLM")
                self.device = "cpu"
                
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
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