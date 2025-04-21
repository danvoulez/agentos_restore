# agentos-pessoas/routes/profiles.py
from fastapi import APIRouter, Depends, HTTPException, status, Request
from schemas.profile import ProfileCreate, ProfileUpdate, ProfileRead, ProfileFilter  # Ensure ProfileFilter is defined
from schemas.base import MsgDetail
from services.profile_service import ProfileService
# Import dependencies from updated auth utils
from utils.auth import get_current_active_user, require_role, require_authentication
from loguru import logger  # Import logger
from typing import List  # Import List

router = APIRouter()

# Inject service using Depends
def get_profile_service(service: ProfileService = Depends(ProfileService)):
    return service

@router.post(
    "/",
    dependencies=[Depends(require_role(["admin"]))]
)
async def create_profile(
    profile_in: ProfileCreate,
    service: ProfileService = Depends(get_profile_service)  # Get injected service
):
    try:
