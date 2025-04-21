# agentos-pessoas/schemas/profile.py
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from enum import Enum

class ProfileType(str, Enum):
    CLIENTE = "cliente"
    VENDEDOR = "vendedor"
    REVENDEDOR = "revendedor"
    ESTAFETA = "estafeta"
    ADMIN = "admin"
    SYSTEM = "system"  # Added system role

class ProfileBase(BaseModel):
    email: EmailStr = Field(..., description="Email único do usuário")
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    phone_number: Optional[str] = Field(None, max_length=30)
    profile_type: ProfileType = Field(..., description="Tipo principal de perfil")
    is_active: bool = Field(True, description="Indica se o perfil está ativo")
    roles: List[str] = Field(default_factory=list, description="Lista de roles associadas ao perfil")

class ProfileCreate(ProfileBase):
    pass

class ProfileUpdate(BaseModel):
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    profile_type: Optional[ProfileType] = None
    is_active: Optional[bool] = None

class ProfileRead(ProfileBase):
    id: str = Field(..., description="ID único do perfil", alias="_id")
    created_at: datetime
    updated_at: Optional[datetime] = None

class ProfileFilter(BaseModel):
    email: Optional[str] = None
    name: Optional[str] = None
    profile_type: Optional[ProfileType] = None
    is_active: Optional[bool] = None
    role: Optional[str] = None
