# Multi-stage build for the modular voice assistant

# Build stage
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3-dev \
    python3-pip \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download models (this can take some time)
RUN mkdir -p /app/models/asr /app/models/llm /app/models/csm \
    && python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
        model = WhisperProcessor.from_pretrained('openai/whisper-small', cache_dir='/app/models/asr'); \
        model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small', cache_dir='/app/models/asr')"

RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
        tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-3-mini-4k-instruct', cache_dir='/app/models/llm', trust_remote_code=True); \
        model = AutoModelForCausalLM.from_pretrained('microsoft/phi-3-mini-4k-instruct', cache_dir='/app/models/llm', \
            trust_remote_code=True, load_in_4bit=True)"

RUN python -c "from transformers import AutoProcessor, AutoModel; \
        processor = AutoProcessor.from_pretrained('sesame/csm-1b', cache_dir='/app/models/csm'); \
        model = AutoModel.from_pretrained('sesame/csm-1b', cache_dir='/app/models/csm')"

# Optimize models (quantize)
RUN python -c "import os; \
        import torch; \
        from transformers import WhisperForConditionalGeneration; \
        model = WhisperForConditionalGeneration.from_pretrained('/app/models/asr/models--openai--whisper-small', \
            torch_dtype=torch.float16); \
        os.makedirs('/app/models/asr/optimized', exist_ok=True); \
        model.save_pretrained('/app/models/asr/optimized')"

# Final stage
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy models from builder stage
COPY --from=builder /app/models /app/models

# Set environment variables
ENV USE_CUDA=1
ENV VOICE_HOST=0.0.0.0
ENV VOICE_PORT=8000
ENV VOICE_DEBUG=0
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Create volume for logs
VOLUME /app/logs

# Run the server
CMD ["python", "-m", "server.main"]