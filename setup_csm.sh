#!/bin/bash
# Setup script for Sesame CSM

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Setting up Sesame CSM =====${NC}"

# Create a directory for external repositories
EXTERNAL_DIR="./external"
mkdir -p "$EXTERNAL_DIR"
cd "$EXTERNAL_DIR"

# Check if the repository already exists
if [ -d "csm" ]; then
    echo -e "${YELLOW}CSM repository already exists.${NC}"
    echo -e "${YELLOW}Updating repository...${NC}"
    cd csm
    git pull
else
    echo -e "${YELLOW}Cloning Sesame CSM repository...${NC}"
    git clone https://github.com/SesameAILabs/csm.git
    cd csm
fi

echo -e "${GREEN}CSM repository setup completed!${NC}"
echo
echo -e "${YELLOW}To use the CSM module, add the following line to your import paths:${NC}"
echo -e "${GREEN}import sys; sys.path.append('$(cd ../../ && pwd)/external/csm')${NC}"
echo
echo -e "${YELLOW}Then you can import the module:${NC}"
echo -e "${GREEN}from csm import ...${NC}"