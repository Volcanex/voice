#!/bin/bash
# Quick script to install required system dependencies

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Installing Voice Assistant System Dependencies =====${NC}"

# Check if we're on a Debian/Ubuntu system
if command -v apt-get &> /dev/null; then
    echo -e "${YELLOW}Installing PortAudio and Tkinter...${NC}"
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-tk
    echo -e "${GREEN}Dependencies installed successfully!${NC}"
else
    echo -e "${RED}Not a Debian/Ubuntu system. Please install these dependencies manually:${NC}"
    echo -e "- PortAudio development headers"
    echo -e "- Tkinter for Python"
    exit 1
fi