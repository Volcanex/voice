#!/usr/bin/env python3
"""
Wrapper script to run the voice assistant client.
"""
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the client
from client.main import main

if __name__ == "__main__":
    main()