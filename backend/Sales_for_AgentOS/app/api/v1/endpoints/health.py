from fastapi import APIRouter

router = APIRouter()

@router.get("/health", summary="Healthcheck")
async def health():
    return {"status": "ok"}