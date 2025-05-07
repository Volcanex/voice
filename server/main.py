"""
Main server application for the modular voice assistant.

This module sets up the FastAPI application with WebSocket support.
"""
import asyncio
import logging
import os
import signal
import sys
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Handle both package and direct script execution
if __package__ is None or __package__ == '':
    # Running as a script
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Now we can import from server package
    from server.config import get_config
    from server.state_manager import StateManager
    from server.modules.asr_module import ASRModule
    from server.modules.llm_module import LLMModule
    from server.modules.csm_module import CSMModule
    from server.websocket_gateway import WebSocketManager
else:
    # Running as a package
    from .config import get_config
    from .state_manager import StateManager
    from .modules.asr_module import ASRModule
    from .modules.llm_module import LLMModule
    from .modules.csm_module import CSMModule
    from .websocket_gateway import WebSocketManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("voice_assistant.log")
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Voice Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application state
app_state = {
    "config": get_config(),
    "state_manager": None,
    "asr_module": None,
    "llm_module": None,
    "csm_module": None,
    "websocket_manager": None,
    "shutdown_event": asyncio.Event()
}

@app.on_event("startup")
async def startup_event():
    """Initialize application state on startup."""
    config = app_state["config"]
    
    # Initialize modules
    app_state["state_manager"] = StateManager(config)
    app_state["asr_module"] = ASRModule(config["asr_model"])
    app_state["llm_module"] = LLMModule(config["llm_model"])
    app_state["csm_module"] = CSMModule(config["csm_model"])
    
    # Initialize WebSocket manager
    app_state["websocket_manager"] = WebSocketManager(
        app_state["state_manager"],
        app_state["asr_module"],
        app_state["llm_module"],
        app_state["csm_module"]
    )
    
    # Log server startup
    logger.info("Voice Assistant server started")
    
    # Check ports and display connection URLs
    if __package__ is None or __package__ == '':
        # Only when running directly, not when imported by uvicorn
        from server.network_utils import get_connection_urls
        from server.network_utils import check_port_accessible
    else:
        from .network_utils import get_connection_urls
        from .network_utils import check_port_accessible
    
    port = config["port"]
    path = config["websocket_path"]
    
    # Check port accessibility
    local, local_network, public = check_port_accessible(port)
    
    # Log connection URLs with status
    logger.info("=" * 50)
    logger.info("VOICE ASSISTANT SERVER CONNECTION INFORMATION")
    logger.info("=" * 50)
    
    # Get and display connection URLs
    urls = get_connection_urls(port, path)
    statuses = [local, local_network, public]
    
    for i, url in enumerate(urls):
        status = "OPEN" if i < len(statuses) and statuses[i] else "UNKNOWN"
        logger.info(f"Connection URL {i+1}: {url} - Status: {status}")
    
    logger.info("=" * 50)
    logger.info("Use one of these URLs to connect with the client")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Voice Assistant server")
    
    # Set shutdown event
    app_state["shutdown_event"].set()
    
    # Shutdown WebSocket manager
    if app_state["websocket_manager"]:
        await app_state["websocket_manager"].shutdown()
        
    # Shutdown modules
    if app_state["asr_module"]:
        await app_state["asr_module"].shutdown()
        
    if app_state["llm_module"]:
        await app_state["llm_module"].shutdown()
        
    if app_state["csm_module"]:
        await app_state["csm_module"].shutdown()
        
    logger.info("Voice Assistant server shutdown complete")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for client connections."""
    websocket_manager = app_state["websocket_manager"]
    if not websocket_manager:
        await websocket.close(code=1011)
        return
        
    await websocket_manager.handle_connection(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "app": "Voice Assistant Server",
        "version": "0.1.0",
        "websocket": "/ws",
        "health": "/health"
    }

def handle_sigterm(signum, frame):
    """Handle SIGTERM signal for graceful shutdown."""
    logger.info("Received SIGTERM signal")
    raise SystemExit(0)

def main():
    """Run the server directly."""
    config = get_config()
    host = config["host"]
    port = config["port"]
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    # Run server with the correct app import path
    # Determine module path based on how it's being run
    if __package__ is None or __package__ == '':
        # Running directly
        app_path = "main:app"
    else:
        # Running as a module
        app_path = "server.main:app"
        
    # Run server
    uvicorn.run(
        app_path,
        host=host,
        port=port,
        reload=config["debug"],
        log_level=config["log_level"].lower()
    )

if __name__ == "__main__":
    main()