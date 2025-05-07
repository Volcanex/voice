#!/bin/bash
# Script to run the voice assistant server in low memory mode

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Running Voice Assistant Server (Low Memory Mode) =====${NC}"

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo -e "${BLUE}Activating virtual environment...${NC}"
        source .venv/bin/activate
    else
        echo -e "${RED}Virtual environment not found. Please run setup.sh first.${NC}"
        exit 1
    fi
fi

# Force CPU mode and low memory settings for constrained environments
export USE_CUDA=0
export LOW_MEMORY=1

# Clean up memory before starting
echo -e "${BLUE}Clearing memory caches...${NC}"
sync  # Flush file system buffers
if [ $(id -u) -eq 0 ]; then
    # If running as root, free pagecache, dentries and inodes
    echo 3 > /proc/sys/vm/drop_caches
else
    echo -e "${BLUE}Not running as root, skipping system cache clearing${NC}"
    # Try forcing garbage collection in Python
    python -c "import gc; gc.collect()" || true
fi

# Set optimal server parameters for low memory
echo -e "${YELLOW}Starting server in low memory mode...${NC}"
echo -e "${BLUE}Voice assistant will use CPU-only mode with memory optimizations${NC}"
echo -e "${BLUE}Models will be loaded with minimal memory footprint${NC}"

# Launch server with memory monitoring
python -m server.main

# This point will only be reached if the server exits
echo -e "${YELLOW}Server has stopped${NC}"