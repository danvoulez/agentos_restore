from app.db.repositories.base_repository import BaseRepository
from app.db.models import SaleDoc, SaleStatus
from app.core.logging_config import logger
from bson import ObjectId
from datetime import datetime, timezone

class SaleRepository(BaseRepository):
    collection_name = "sales"
    model = SaleDoc

    async def update_sale_status(self, sale_id: str, status: SaleStatus):
        res = await self.collection.update_one(
            {"_id": ObjectId(sale_id)},
            {"$set": {"status": status, "updated_at": datetime.now(timezone.utc)}}
        )
        return res.modified_count == 1