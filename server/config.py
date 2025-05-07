"""
Configuration settings for the modular voice assistant system.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent.absolute()
MODEL_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"

# Server settings
HOST = os.getenv("VOICE_HOST", "0.0.0.0")
PORT = int(os.getenv("VOICE_PORT", "8000"))
WEBSOCKET_PATH = "/ws"
DEBUG = os.getenv("VOICE_DEBUG", "0") == "1"

# Model settings
ASR_MODEL = {
    "model_id": "openai/whisper-small",
    "device": "cuda" if os.getenv("USE_CUDA", "1") == "1" else "cpu",
    "compute_type": "float16" if os.getenv("USE_CUDA", "1") == "1" else "int8",
    "cache_dir": str(MODEL_DIR / "asr"),
}

LLM_MODEL = {
    "model_id": "microsoft/phi-3-mini-4k-instruct",
    "device": "cuda" if os.getenv("USE_CUDA", "1") == "1" else "cpu",
    "quantize": "gptq" if os.getenv("USE_CUDA", "1") == "1" else "gguf",
    "cache_dir": str(MODEL_DIR / "llm"),
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
}

CSM_MODEL = {
    "model_id": "sesame/csm-1b",
    "device": "cuda" if os.getenv("USE_CUDA", "1") == "1" else "cpu",
    "cache_dir": str(MODEL_DIR / "csm"),
    "quality": "high",  # Options: low, medium, high
}

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
AUDIO_FORMAT = "wav"  # Supported: wav, mp3, ogg

# Conversation settings
MAX_HISTORY_LENGTH = 10  # Number of conversation turns to keep
SYSTEM_PROMPT = """You are a helpful voice assistant. 
Keep your responses concise and conversational.
Respond to the user's queries in a natural, helpful way."""

# Logging settings
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"
LOG_DIR = BASE_DIR / "logs"

# Create required directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR / "asr", exist_ok=True)
os.makedirs(MODEL_DIR / "llm", exist_ok=True)
os.makedirs(MODEL_DIR / "csm", exist_ok=True)

def get_config() -> Dict[str, Any]:
    """Return the full configuration as a dictionary."""
    return {
        "host": HOST,
        "port": PORT,
        "websocket_path": WEBSOCKET_PATH,
        "debug": DEBUG,
        "asr_model": ASR_MODEL,
        "llm_model": LLM_MODEL,
        "csm_model": CSM_MODEL,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "chunk_size": CHUNK_SIZE,
        "audio_format": AUDIO_FORMAT,
        "max_history_length": MAX_HISTORY_LENGTH,
        "system_prompt": SYSTEM_PROMPT,
        "log_level": LOG_LEVEL,
    }