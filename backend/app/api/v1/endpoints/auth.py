# promptos_backend/app/api/v1/endpoints/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

router = APIRouter()

@router.post("/login")
async def login():
    logger.info("Login endpoint called")
    # Add login logic here
    pass
