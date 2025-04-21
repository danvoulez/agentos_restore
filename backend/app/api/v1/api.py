# promptos_backend/app/api/v1/api.py
from fastapi import APIRouter
from .endpoints import auth, files, websocket_endpoints

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["Auth"])
router.include_router(files.router, prefix="/files", tags=["Files"])
router.include_router(websocket_endpoints.router, prefix="/ws", tags=["WebSocket"])
