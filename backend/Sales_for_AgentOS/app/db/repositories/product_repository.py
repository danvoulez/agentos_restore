from app.db.repositories.base_repository import BaseRepository
from app.db.models import PyObjectId
from app.core.exceptions import RepositoryError
from bson import ObjectId

class ProductRepository(BaseRepository):
    collection_name = "products"
    model = dict  # simplificado

    async def allocate_stock(self, sku: str, quantity: int):
        res = await self.collection.update_one(
            {"sku": sku, "stock": {"$gte": quantity}},
            {"$inc": {"stock": -quantity}}
        )
        if res.modified_count != 1:
            raise RepositoryError(f"Stock insufficient for {sku}")