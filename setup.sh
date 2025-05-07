#!/bin/bash
# Setup script for the voice assistant project
# Creates a virtual environment with Python 3.10

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Setting up Voice Assistant Environment =====${NC}"

# Ensure working directory is the project root
cd "$(dirname "$0")"

# Check if Python 3.10 is available
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo -e "${GREEN}Found Python 3.10${NC}"
else
    echo -e "${YELLOW}Python 3.10 not found, checking for alternatives...${NC}"
    
    # Check if Python 3.9 or 3.11 is available as fallback
    if command -v python3.9 &> /dev/null; then
        PYTHON_CMD="python3.9"
        echo -e "${YELLOW}Using Python 3.9 instead${NC}"
    elif command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        echo -e "${YELLOW}Using Python 3.11 instead${NC}"
    elif command -v python3 &> /dev/null; then
        # Check Python 3 version
        PY_VERSION=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
        echo -e "${YELLOW}Found Python ${PY_VERSION}${NC}"
        
        if [[ "$PY_VERSION" == "3.10" ]]; then
            PYTHON_CMD="python3"
        elif [[ "$PY_VERSION" == "3.9" || "$PY_VERSION" == "3.11" ]]; then
            PYTHON_CMD="python3"
            echo -e "${YELLOW}Python 3.10 is recommended, but Python ${PY_VERSION} should work${NC}"
        else
            echo -e "${RED}Warning: Python 3.10 is recommended, Python ${PY_VERSION} may have compatibility issues${NC}"
            read -p "Continue with Python ${PY_VERSION}? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                PYTHON_CMD="python3"
            else
                echo -e "${RED}Setup aborted. Please install Python 3.10${NC}"
                exit 1
            fi
        fi
    else
        echo -e "${RED}Error: Python 3.x not found. Please install Python 3.10${NC}"
        exit 1
    fi
fi

# Check for pip
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo -e "${RED}Error: pip not found for $PYTHON_CMD${NC}"
    echo -e "${YELLOW}Try installing pip: sudo apt install python3-pip${NC}"
    exit 1
fi

# Check for venv module
if ! $PYTHON_CMD -c "import venv" &> /dev/null; then
    echo -e "${RED}Error: venv module not found for $PYTHON_CMD${NC}"
    echo -e "${YELLOW}Try installing venv: sudo apt install python3-venv${NC}"
    exit 1
fi

# If a previous venv exists, ask if it should be replaced
if [ -d ".venv" ]; then
    read -p "Virtual environment already exists. Replace it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf .venv
    else
        echo -e "${YELLOW}Using existing virtual environment${NC}"
        # Activate the environment
        source .venv/bin/activate
        echo -e "${GREEN}Virtual environment activated!${NC}"
        echo -e "${YELLOW}Installing/updating dependencies...${NC}"
        pip install -r requirements.txt
        echo -e "${GREEN}Setup complete!${NC}"
        echo
        echo -e "To activate the virtual environment:"
        echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
        exit 0
    fi
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment with $PYTHON_CMD...${NC}"
$PYTHON_CMD -m venv .venv

# Activate the environment
source .venv/bin/activate

# Verify Python version
VENV_PYTHON_VERSION=$(python -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
echo -e "${GREEN}Virtual environment created with Python ${VENV_PYTHON_VERSION}${NC}"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Setup Sesame CSM
echo -e "${YELLOW}Setting up Sesame CSM...${NC}"
./setup_csm.sh

# Install development dependencies
echo -e "${YELLOW}Installing development dependencies...${NC}"
pip install pytest pytest-asyncio pytest-mock

echo -e "${GREEN}Setup complete!${NC}"
echo
echo -e "To activate the virtual environment:"
echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
echo
echo -e "To run the tests:"
echo -e "  ${YELLOW}./run_tests.sh${NC}"