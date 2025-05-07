#!/usr/bin/env python3
"""
Wrapper script for running the Voice Assistant server.

This script provides a more detailed startup banner and ensures proper imports.
"""
import os
import sys
import logging
from datetime import datetime

def print_banner():
    """Print a nice banner with server information."""
    banner = f"""
{'='*70}
           VOICE ASSISTANT SERVER
           
Version: 0.1.0
Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

The server will check for open ports and display connection URLs
when it starts. Use these URLs to connect with the client.

For more detailed logs, check voice_assistant.log
{'='*70}
"""
    print(banner)

def main():
    """Run the Voice Assistant server."""
    # Ensure working directory is the project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Print banner
    print_banner()
    
    # Configure logging to show all INFO messages in console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("voice_assistant.log")
        ]
    )
    
    # Import and run server
    from server.main import main as server_main
    
    # Run server
    server_main()

if __name__ == "__main__":
    main()