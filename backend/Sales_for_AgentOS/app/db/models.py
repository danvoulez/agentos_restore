from datetime import datetime, timezone
from bson import ObjectId
from pydantic import BaseModel, Field, computed_field
from enum import Enum

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls): yield cls.validate
    @classmethod
    def validate(cls, v): return ObjectId(str(v))

class SaleStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

class IntegrationStatusEnum(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"

class IntegrationStatus(BaseModel):
    status: IntegrationStatusEnum = IntegrationStatusEnum.PENDING
    external_id: str | None = None
    retry_count: int = 0
    last_error: str | None = None
    last_attempted_at: datetime | None = None

class SaleItem(BaseModel):
    product_id: str
    sku: str
    name: str
    quantity: int
    unit_price: float
    total_price: float

class SaleDoc(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    client_id: str
    agent_id: str
    items: list[SaleItem]
    total_amount: float
    currency: str = "BRL"
    status: SaleStatus = SaleStatus.PROCESSING
    payment_info: IntegrationStatus = IntegrationStatus()
    delivery_info: IntegrationStatus = IntegrationStatus()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field  # type: ignore[misc]
    @property
    def status_history(self) -> list[dict]:  # fake field for docs
        return []