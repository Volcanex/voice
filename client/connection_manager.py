"""
Connection Manager for the Voice Assistant client.

Handles server connection profiles, SSH tunneling, and connection management.
"""
import asyncio
import json
import logging
import os
import subprocess
import threading
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Default connection config file path
DEFAULT_CONFIG_PATH = Path.home() / ".voice_assistant" / "connections.json"

class SSHTunnel:
    """
    Manages SSH tunneling for remote connections.
    """
    def __init__(self, ssh_host: str, ssh_port: int, ssh_user: str, 
                 remote_host: str, remote_port: int, local_port: int):
        """
        Initialize SSH tunnel.
        
        Args:
            ssh_host: SSH host to connect to
            ssh_port: SSH port to use
            ssh_user: SSH username
            remote_host: Remote host for port forwarding (usually localhost)
            remote_port: Remote port to forward
            local_port: Local port to forward to
        """
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_port = local_port
        self.process = None
        self.is_running = False
        self.monitoring_thread = None
        
        logger.info(f"SSH tunnel initialized for {ssh_user}@{ssh_host}:{ssh_port}")
    
    def start(self) -> bool:
        """
        Start the SSH tunnel.
        
        Returns:
            True if tunnel started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("SSH tunnel already running")
            return True
            
        try:
            # Prepare the command
            cmd = [
                "ssh",
                "-N",  # No command execution
                "-L", f"{self.local_port}:{self.remote_host}:{self.remote_port}",
                f"{self.ssh_user}@{self.ssh_host}",
                "-p", str(self.ssh_port)
            ]
            
            cmd_str = ' '.join(cmd)
            logger.info(f"Starting SSH tunnel with command: {cmd_str}")
            logger.info(f"This will forward {self.remote_host}:{self.remote_port} to localhost:{self.local_port}")
            
            # Start the process
            start_time = time.time()
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Give it a moment to establish
            logger.info("Waiting for SSH tunnel to establish...")
            time.sleep(2)
            
            # Check if process is still running
            if self.process.poll() is not None:
                stderr = self.process.stderr.read()
                logger.error(f"SSH tunnel failed to start: {stderr}")
                self.process = None
                return False
                
            tunnel_startup_time = time.time() - start_time
            logger.info(f"SSH tunnel process started in {tunnel_startup_time:.2f} seconds")
            
            self.is_running = True
            
            # Start monitoring thread
            logger.info("Starting SSH tunnel monitoring thread")
            self.monitoring_thread = threading.Thread(
                target=self._monitor_tunnel,
                daemon=True
            )
            self.monitoring_thread.start()
            
            logger.info(f"SSH tunnel established successfully from localhost:{self.local_port} to "
                       f"{self.ssh_user}@{self.ssh_host}:{self.remote_port}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting SSH tunnel: {str(e)}")
            logger.debug("SSH tunnel error details:", exc_info=True)
            self.stop()
            return False
    
    def stop(self) -> None:
        """
        Stop the SSH tunnel.
        """
        if self.process:
            try:
                self.process.terminate()
                # Wait briefly for graceful termination
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                logger.exception(f"Error stopping SSH tunnel: {e}")
            
            self.process = None
            
        self.is_running = False
        logger.info("SSH tunnel stopped")
    
    def _monitor_tunnel(self) -> None:
        """
        Monitor the SSH tunnel to make sure it's still running.
        """
        while self.is_running and self.process:
            # Check if process is still running
            if self.process.poll() is not None:
                stderr = self.process.stderr.read()
                logger.error(f"SSH tunnel stopped unexpectedly: {stderr}")
                self.is_running = False
                break
                
            # Sleep for a bit
            time.sleep(5)

class ConnectionManager:
    """
    Manages server connection profiles for the Voice Assistant client.
    """
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the connection manager.
        
        Args:
            config_path: Path to connection config file (optional)
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.connections = {}
        self.active_tunnel = None
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Load connections
        self.load_connections()
        
        logger.info(f"Connection manager initialized with {len(self.connections)} connections")
    
    def load_connections(self) -> None:
        """
        Load connections from config file.
        """
        if not os.path.exists(self.config_path):
            # Create default connections
            self.connections = {
                "Local": {
                    "name": "Local",
                    "url": "ws://localhost:8000/ws",
                    "use_ssh": False
                }
            }
            self.save_connections()
            return
            
        try:
            with open(self.config_path, "r") as f:
                self.connections = json.load(f)
                
            logger.info(f"Loaded {len(self.connections)} connections from {self.config_path}")
            
        except Exception as e:
            logger.exception(f"Error loading connections: {e}")
            # Use default connections
            self.connections = {
                "Local": {
                    "name": "Local",
                    "url": "ws://localhost:8000/ws",
                    "use_ssh": False
                }
            }
    
    def save_connections(self) -> None:
        """
        Save connections to config file.
        """
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.connections, f, indent=2)
                
            logger.info(f"Saved {len(self.connections)} connections to {self.config_path}")
            
        except Exception as e:
            logger.exception(f"Error saving connections: {e}")
    
    def add_connection(self, 
                      name: str, 
                      url: str, 
                      use_ssh: bool = False,
                      ssh_host: str = "",
                      ssh_port: int = 22,
                      ssh_user: str = "",
                      remote_host: str = "localhost",
                      remote_port: int = 8000,
                      local_port: int = 8000) -> None:
        """
        Add a new connection.
        
        Args:
            name: Connection name
            url: WebSocket URL
            use_ssh: Whether to use SSH tunneling
            ssh_host: SSH host (if use_ssh is True)
            ssh_port: SSH port (if use_ssh is True)
            ssh_user: SSH username (if use_ssh is True)
            remote_host: Remote host for port forwarding (if use_ssh is True)
            remote_port: Remote port for port forwarding (if use_ssh is True)
            local_port: Local port for port forwarding (if use_ssh is True)
        """
        self.connections[name] = {
            "name": name,
            "url": url,
            "use_ssh": use_ssh
        }
        
        if use_ssh:
            self.connections[name].update({
                "ssh_host": ssh_host,
                "ssh_port": ssh_port,
                "ssh_user": ssh_user,
                "remote_host": remote_host,
                "remote_port": remote_port,
                "local_port": local_port
            })
            
        self.save_connections()
        logger.info(f"Added connection: {name}")
    
    def update_connection(self, name: str, **kwargs) -> None:
        """
        Update an existing connection.
        
        Args:
            name: Connection name
            **kwargs: Connection parameters to update
        """
        if name not in self.connections:
            logger.warning(f"Connection not found: {name}")
            return
            
        self.connections[name].update(kwargs)
        self.save_connections()
        logger.info(f"Updated connection: {name}")
    
    def remove_connection(self, name: str) -> None:
        """
        Remove a connection.
        
        Args:
            name: Connection name
        """
        if name not in self.connections:
            logger.warning(f"Connection not found: {name}")
            return
            
        del self.connections[name]
        self.save_connections()
        logger.info(f"Removed connection: {name}")
    
    def get_connection(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a connection by name.
        
        Args:
            name: Connection name
            
        Returns:
            Connection details or None if not found
        """
        return self.connections.get(name)
    
    def get_connection_names(self) -> List[str]:
        """
        Get a list of all connection names.
        
        Returns:
            List of connection names
        """
        return list(self.connections.keys())
    
    async def connect(self, name: str) -> Tuple[bool, str]:
        """
        Connect to a server.
        
        Args:
            name: Connection name
            
        Returns:
            Tuple of (success, url_or_error)
        """
        logger.info(f"Attempting to connect to: {name}")
        
        connection = self.get_connection(name)
        if not connection:
            logger.error(f"Connection profile not found: {name}")
            return False, f"Connection not found: {name}"
            
        # Stop any existing tunnel
        if self.active_tunnel:
            logger.info("Closing existing SSH tunnel before establishing new connection")
            self.disconnect()
            
        if connection.get("use_ssh", False):
            # Start SSH tunnel
            logger.info(f"Setting up SSH tunnel to {connection['ssh_host']}:{connection['ssh_port']} "
                       f"as user {connection['ssh_user']}")
            logger.info(f"Forwarding remote {connection['remote_host']}:{connection['remote_port']} "
                       f"to local port {connection['local_port']}")
            
            tunnel = SSHTunnel(
                connection["ssh_host"],
                connection["ssh_port"],
                connection["ssh_user"],
                connection["remote_host"],
                connection["remote_port"],
                connection["local_port"]
            )
            
            logger.info("Attempting to establish SSH tunnel...")
            if not tunnel.start():
                logger.error("Failed to establish SSH tunnel")
                return False, "Failed to establish SSH tunnel"
                
            logger.info("SSH tunnel established successfully")
            self.active_tunnel = tunnel
            
            # Use local port for connection
            ws_url = f"ws://localhost:{connection['local_port']}/ws"
            logger.info(f"Using tunneled connection URL: {ws_url}")
            return True, ws_url
        else:
            # Use direct URL
            logger.info(f"Using direct connection URL: {connection['url']}")
            return True, connection["url"]
    
    def disconnect(self) -> None:
        """
        Disconnect from server and close any tunnels.
        """
        if self.active_tunnel:
            self.active_tunnel.stop()
            self.active_tunnel = None
            logger.info("Closed SSH tunnel")