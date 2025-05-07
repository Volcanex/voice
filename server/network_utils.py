"""
Network utilities for the voice assistant server.

This module provides functions to check network connectivity, get IP addresses,
and check if ports are open.
"""
import logging
import socket
import subprocess
import requests
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

def get_local_ip() -> str:
    """
    Get the local IP address of the machine.
    
    Returns:
        Local IP address as a string
    """
    try:
        # Create a socket to connect to an external service
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Use a public DNS server - doesn't actually send traffic
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.warning(f"Failed to get local IP: {e}")
        # Fallback to localhost
        return "127.0.0.1"

def get_public_ip() -> Optional[str]:
    """
    Get the public IP address of the machine.
    
    Returns:
        Public IP address as a string or None if failed
    """
    try:
        # Use a public service to get the IP
        response = requests.get("https://api.ipify.org", timeout=5)
        if response.status_code == 200:
            return response.text
        return None
    except Exception as e:
        logger.warning(f"Failed to get public IP: {e}")
        return None

def is_port_open(host: str, port: int) -> bool:
    """
    Check if a port is open on a host.
    
    Args:
        host: Host to check
        port: Port to check
        
    Returns:
        True if port is open, False otherwise
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex((host, port))
        s.close()
        
        # Log more details about the result
        if result == 0:
            logger.info(f"Port {port} on {host} is open (result: {result})")
        else:
            logger.warning(f"Port {port} on {host} appears to be closed (result: {result})")
            
        return result == 0
    except Exception as e:
        logger.warning(f"Failed to check if port {port} is open: {e}")
        return False

def check_port_accessible(port: int) -> Tuple[bool, bool, bool]:
    """
    Check if a port is accessible locally, on local network, and publicly.
    
    Args:
        port: Port to check
        
    Returns:
        Tuple of (local, local_network, public) booleans
    """
    # Check if port is open locally
    local = is_port_open("127.0.0.1", port)
    
    # Check if port is open on local network
    local_ip = get_local_ip()
    local_network = is_port_open(local_ip, port)
    
    # Check if port is open publicly
    # This is likely to be False unless the machine has a public IP
    # or port forwarding is set up
    public = False
    public_ip = get_public_ip()
    if public_ip:
        public = is_port_open(public_ip, port)
    
    return (local, local_network, public)

def get_connection_urls(port: int, path: str = "/ws") -> List[str]:
    """
    Get a list of potential connection URLs.
    
    Args:
        port: Port to use
        path: WebSocket path
        
    Returns:
        List of connection URLs
    """
    urls = []
    
    # Local URL
    urls.append(f"ws://localhost:{port}{path}")
    
    # Local network URL
    local_ip = get_local_ip()
    if local_ip != "127.0.0.1":
        urls.append(f"ws://{local_ip}:{port}{path}")
    
    # Public URL
    public_ip = get_public_ip()
    if public_ip:
        urls.append(f"ws://{public_ip}:{port}{path}")
    
    return urls

def is_port_forwarded(port: int) -> bool:
    """
    Check if a port is forwarded to the public internet.
    
    Args:
        port: Port to check
        
    Returns:
        True if port is forwarded, False otherwise
    """
    public_ip = get_public_ip()
    if not public_ip:
        return False
    
    try:
        # Use an external service to check if port is open from outside
        response = requests.get(f"https://www.canyouseeme.org/?port={port}", timeout=5)
        # This is a basic check, as the response would need to be parsed
        # For a more accurate check, a proper port checking service should be used
        return "Success" in response.text
    except Exception as e:
        logger.warning(f"Failed to check if port {port} is forwarded: {e}")
        return False