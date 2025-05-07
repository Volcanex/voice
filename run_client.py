#!/usr/bin/env python3
"""
Wrapper script for running the Voice Assistant client.

This script provides a more detailed startup banner and ensures proper imports.
"""
import asyncio
import os
import sys
import logging
from datetime import datetime

def print_banner():
    """Print a nice banner with connection information."""
    banner = f"""
{'='*70}
           VOICE ASSISTANT CLIENT
           
Version: 0.1.0
Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

To connect to a server:
1. Click the "Connect" button in the top right
2. Select a connection profile or create a new one
3. For SSH tunneled connections, ensure SSH is properly configured

For more detailed logs, check voice_assistant_client.log
{'='*70}
"""
    print(banner)

def main():
    """Run the Voice Assistant client."""
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
            logging.FileHandler("voice_assistant_client.log")
        ]
    )
    
    # Import and run client
    from client.main import main as client_main
    
    # Run client
    client_main()

if __name__ == "__main__":
    main()