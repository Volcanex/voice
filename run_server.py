#!/usr/bin/env python3
"""
Wrapper script to run the voice assistant server.
"""
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the server
from server.main import main

if __name__ == "__main__":
    main()