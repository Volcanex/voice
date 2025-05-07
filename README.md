# Modular CSM-LLM Voice Assistant System

A modular, low-latency voice assistant system that combines Sesame AI's Controllable Speech Model (CSM) with Phi-3-mini for natural language processing.

## Architecture

The system consists of these key components:

- **Speech-to-text module (ASR)**: Using OpenAI's Whisper-small
- **Language model**: Microsoft's Phi-3-mini-4k-instruct for text generation
- **CSM**: Sesame AI's CSM-1B for natural speech synthesis
- **WebSocket server**: For real-time client communication
- **Python client**: With TKinter UI for user interaction

## Directory Structure

```
VOICE/
├── README.md
├── requirements.txt
├── server/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── websocket_gateway.py
│   ├── state_manager.py
│   └── modules/
│       ├── __init__.py
│       ├── asr_module.py
│       ├── llm_module.py
│       └── csm_module.py
├── client/
│   ├── __init__.py
│   ├── main.py
│   ├── ui.py
│   └── websocket_client.py
├── tests/
│   ├── __init__.py
│   ├── test_asr.py
│   ├── test_llm.py
│   ├── test_csm.py
│   └── test_e2e.py
└── Dockerfile
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd VOICE

# Use the setup script (Python 3.10 recommended)
./setup.sh

# Alternative manual setup:
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup Sesame CSM
./setup_csm.sh

# Download models (this will take some time)
python -m server.download_models
```

### Requirements Note

This project has a few special dependencies:

1. **asynctkinter**: The original project required `tkinter-async>=0.2.0` which is not available on PyPI. We've replaced it with `asynctkinter` which provides similar functionality.

2. **sounddevice**: We use `sounddevice` instead of `pyaudio` for audio capture and playback. While both require PortAudio as a system-level dependency, sounddevice has a cleaner API and better error handling.

3. **sesame-csm**: This is a private package not available on PyPI. The setup script will clone it from GitHub and set up the module for use in the project.

### System Dependencies

Before running the application, you need to install some system-level dependencies:

```bash
# For Debian/Ubuntu systems
sudo apt-get install portaudio19-dev python3-tk

# OR use our included script
./install_deps.sh
```

These are required for:
- **PortAudio**: Backend for audio input/output (required by sounddevice)
- **Tkinter**: For the GUI interface

### Installation Notes

This project uses a controlled installation process to handle dependencies properly:

1. Run `./setup.sh` to create the virtual environment and install dependencies.
2. For manual installation, use `./install_requirements.sh` after activating your virtual environment.
3. GPU acceleration packages (bitsandbytes, flash-attn) are optional and will be skipped if installation fails.
4. The system will automatically fall back to CPU operation if GPU libraries are not available.

### Python Version Requirement

This project is designed to work with Python 3.10. While it may work with Python 3.9 or 3.11, using Python 3.12+ might cause compatibility issues with some dependencies.

### CUDA/GPU Support

To use GPU acceleration:

1. Install the NVIDIA CUDA Toolkit (version 11.8 or higher) on your system
2. Install PyTorch with CUDA support matching your installed CUDA version
3. The system will automatically use GPU if available

For CPU-only operation, no additional setup is required - the system will fall back to CPU processing.

## Usage

### Running the Server

```bash
# Start the server (using the wrapper script)
./run_server.py

# Alternative method (as a module)
python -m server.main
```

### Running the Client

```bash
# Start the client (using the wrapper script)
./run_client.py

# Alternative method (as a module)
python -m client.main
```

## Docker Deployment

```bash
# Build the Docker image
docker build -t voice-assistant .

# Run the container
docker run -p 8000:8000 --gpus all voice-assistant
```

## Performance Metrics

- End-to-end latency: Under 1 second from end of speech to first audio response
- Module independence: Each module can run and be tested standalone
- Container size: Less than 10GB for the complete system
- Resource usage: Runs on a single GPU with 16GB VRAM

## Model Details

- **ASR**: [OpenAI Whisper-small](https://huggingface.co/openai/whisper-small) (244M parameters)
- **LLM**: [Microsoft Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/phi-3-mini-4k-instruct) (3.8B parameters)
- **CSM**: [Sesame AI CSM-1B](https://huggingface.co/sesame/csm-1b)