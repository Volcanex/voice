"""
Model downloading script for the voice assistant.

This script downloads all required models for the voice assistant.
"""
import os
import logging
import sys
from pathlib import Path

import torch
import torchaudio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModel
)

from .config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("model_download.log")
    ]
)

logger = logging.getLogger(__name__)

def download_asr_model():
    """Download and cache the ASR model."""
    config = get_config()
    model_config = config["asr_model"]
    
    logger.info(f"Downloading ASR model: {model_config['model_id']}")
    
    try:
        # Download processor
        WhisperProcessor.from_pretrained(
            model_config["model_id"], 
            cache_dir=model_config["cache_dir"]
        )
        
        # Download model
        WhisperForConditionalGeneration.from_pretrained(
            model_config["model_id"], 
            cache_dir=model_config["cache_dir"]
        )
        
        logger.info(f"ASR model downloaded successfully to {model_config['cache_dir']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download ASR model: {e}")
        return False

def download_llm_model():
    """Download and cache the LLM model."""
    config = get_config()
    model_config = config["llm_model"]
    
    logger.info(f"Downloading LLM model: {model_config['model_id']}")
    
    try:
        # Download tokenizer
        AutoTokenizer.from_pretrained(
            model_config["model_id"], 
            cache_dir=model_config["cache_dir"],
            trust_remote_code=True
        )
        
        # Download model
        model_kwargs = {
            "cache_dir": model_config["cache_dir"],
            "trust_remote_code": True,
        }
        
        # Add quantization options if needed
        if model_config["quantize"] == "gptq" and model_config["device"] == "cuda":
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
        elif model_config["quantize"] == "gguf" and model_config["device"] == "cpu":
            model_kwargs["load_in_8bit"] = True
        
        AutoModelForCausalLM.from_pretrained(
            model_config["model_id"],
            **model_kwargs
        )
        
        logger.info(f"LLM model downloaded successfully to {model_config['cache_dir']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download LLM model: {e}")
        return False

def download_csm_model():
    """Download and cache the CSM model."""
    config = get_config()
    model_config = config["csm_model"]
    
    logger.info(f"Downloading CSM model: {model_config['model_id']}")
    
    try:
        # Download processor
        AutoProcessor.from_pretrained(
            model_config["model_id"], 
            cache_dir=model_config["cache_dir"]
        )
        
        # Download model
        AutoModel.from_pretrained(
            model_config["model_id"], 
            cache_dir=model_config["cache_dir"]
        )
        
        logger.info(f"CSM model downloaded successfully to {model_config['cache_dir']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download CSM model: {e}")
        return False

def main():
    """Download all models."""
    logger.info("Starting model download")
    
    # Create directory structure if not exists
    config = get_config()
    os.makedirs(config["asr_model"]["cache_dir"], exist_ok=True)
    os.makedirs(config["llm_model"]["cache_dir"], exist_ok=True)
    os.makedirs(config["csm_model"]["cache_dir"], exist_ok=True)
    
    # Download models
    asr_success = download_asr_model()
    llm_success = download_llm_model()
    csm_success = download_csm_model()
    
    # Check all downloads
    if asr_success and llm_success and csm_success:
        logger.info("All models downloaded successfully")
    else:
        logger.error("Some models failed to download")
        sys.exit(1)

if __name__ == "__main__":
    main()