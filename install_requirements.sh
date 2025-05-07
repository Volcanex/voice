#!/bin/bash
# Script to install requirements with fallbacks for optional dependencies

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Installing Voice Assistant Requirements =====${NC}"

# Ensure working directory is the project root
cd "$(dirname "$0")"

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}Error: No virtual environment activated${NC}"
    echo -e "${YELLOW}Please activate your virtual environment first:${NC}"
    echo -e "    source .venv/bin/activate"
    exit 1
fi

# Ensure pip is up to date
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install core dependencies first
echo -e "${YELLOW}Installing core dependencies...${NC}"
pip install numpy scipy pydantic python-dotenv loguru

# Install PyTorch
echo -e "${YELLOW}Installing PyTorch...${NC}"
pip install torch torchaudio transformers

# Install the rest of the required packages
echo -e "${YELLOW}Installing required packages...${NC}"
pip install -r requirements.txt

# Try to install optional GPU acceleration packages
echo -e "${YELLOW}Attempting to install optional GPU acceleration packages...${NC}"
echo -e "${BLUE}Note: These may fail if hardware or system libraries are missing.${NC}"
echo -e "${BLUE}The system will fall back to CPU operation if these fail.${NC}"

# Create a temporary file with the optional requirements
cat > /tmp/optional_requirements.txt << EOL
bitsandbytes>=0.41.0
flash-attn>=2.3.0
EOL

# Try to install each optional package individually
while read -r package; do
    echo -e "${YELLOW}Trying to install: $package${NC}"
    pip install $package 2>/dev/null || echo -e "${BLUE}Skipping optional package: $package${NC}"
done < /tmp/optional_requirements.txt

# Remove the temporary file
rm /tmp/optional_requirements.txt

echo -e "${GREEN}===== Installation Complete =====${NC}"
echo -e "${YELLOW}If you encountered errors, the system should still function using CPU-only mode.${NC}"