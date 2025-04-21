# agentos-pessoas/routes/roles.py
from fastapi import APIRouter, Depends, HTTPException, status
from schemas.role import RoleCreate, RoleRead
from schemas.base import MsgDetail
from services.role_service import RoleService
# Import updated auth dependency
from utils.auth import require_role
from loguru import logger  # Import logger
from typing import List  # Import List

router = APIRouter()

# Inject service using Depends
def get_role_service(service: RoleService = Depends(RoleService)):
    return service

@router.post(
    "/",
    response_model=RoleRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create new role",
    # Use role checker dependency directly
    dependencies=[Depends(require_role(["admin"]))]
)
async def create_role(
    role_in: RoleCreate,
    service: RoleService = Depends(get_role_service)  # Get injected service
):
    try:
