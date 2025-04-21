# agentos-pessoas/services/role_service.py
from schemas.role import RoleCreate, RoleRead
from typing import List
from loguru import logger

class RoleService:
    """Service for managing roles."""

    async def create_role(self, role_data: RoleCreate) -> RoleRead:
        logger.info(f"Creating role: {role_data}")
        # Add logic to create a role in the database
        pass

    async def list_roles(self) -> List[RoleRead]:
        logger.info("Listing all roles")
        # Add logic to list all roles
        pass
