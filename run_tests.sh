#!/bin/bash
# Script to run all tests for the voice assistant

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Running Voice Assistant Tests =====${NC}"

# Ensure working directory is the project root
cd "$(dirname "$0")"

# Ensure test directories exist
mkdir -p ./test_models/asr
mkdir -p ./test_models/llm
mkdir -p ./test_models/csm

# Check if virtual environment exists, if not create it
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install pytest pytest-asyncio pytest-mock
else
    source venv/bin/activate
fi

# Run the tests
echo -e "${YELLOW}Running ASR module tests...${NC}"
python -m pytest tests/test_asr.py -v

echo -e "${YELLOW}Running LLM module tests...${NC}"
python -m pytest tests/test_llm.py -v

echo -e "${YELLOW}Running CSM module tests...${NC}"
python -m pytest tests/test_csm.py -v

echo -e "${YELLOW}Running end-to-end tests...${NC}"
python -m pytest tests/test_e2e.py -v

echo -e "${GREEN}===== All tests completed successfully! =====${NC}"