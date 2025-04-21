# promptos_backend/app/websocket/connection_manager.py
from loguru import logger

class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections = []

    def connect(self, websocket):
        logger.info("Adding WebSocket connection")
        self.active_connections.append(websocket)

    def disconnect(self, websocket):
        logger.info("Removing WebSocket connection")
        self.active_connections.remove(websocket)
