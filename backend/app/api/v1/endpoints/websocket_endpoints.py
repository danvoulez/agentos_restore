# promptos_backend/app/api/v1/endpoints/websocket_endpoints.py
from fastapi import APIRouter, WebSocket
from loguru import logger

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    # Add WebSocket logic here
    pass
