# agentos-pessoas/schemas/role.py
from pydantic import BaseModel, Field
from typing import Optional

class RoleBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=50, description="Nome único da role (e.g., 'admin', 'vendedor')")
    description: Optional[str] = Field(None, max_length=255)

class RoleCreate(RoleBase):
    pass

class RoleRead(RoleBase):
    id: str = Field(..., description="ID único da role", alias="_id")
