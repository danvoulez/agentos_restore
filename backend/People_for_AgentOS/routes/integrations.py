# agentos-pessoas/routes/integrations.py
from fastapi import APIRouter, Depends, HTTPException, status, Request
from loguru import logger
from schemas.profile import ProfileRead
from schemas.base import MsgDetail, settings  # Import settings too
from services.profile_service import ProfileService
# Import specific dependencies from the updated auth utils
from utils.auth import require_role  # Use role check for internal auth for now
import uuid

router = APIRouter(tags=["Internal Integrations"])

# Define dependency for checking access (Admin/System JWT Role)
InternalAccessDependency = Depends(require_role(["system", "admin"]))

@router.get(
    "/profile/{profile_id}",
    response_model=ProfileRead,
    summary="[Internal] Get profile data by ID",
    # Use require_role directly as the dependency for checking JWT roles
    dependencies=[InternalAccessDependency],
    responses={
        401: {"description": "Authentication required"},
        404: {"model": MsgDetail, "description": "Profile not found"}
    }
)
async def get_profile_for_internal_use(
    profile_id: str,
    request: Request,
    # Inject service using Depends
    service: ProfileService = Depends(ProfileService)
):
    # Auth check is handled by the dependency above
    trace_id = request.state.trace_id if hasattr(request.state, 'trace_id') else str(uuid.uuid4())
    log = logger.bind(trace_id=trace_id)
    log.info(f"Internal request to fetch profile {profile_id}")

    try:
        pessoa = await service.get_profile_by_id(profile_id, internal_call=True)
        return pessoa
    except HTTPException as e:
        raise e  # Re-raise known HTTP errors
    except Exception as e:
