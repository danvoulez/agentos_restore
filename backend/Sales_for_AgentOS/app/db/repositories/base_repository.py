from typing import Any, Dict, List
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from bson import ObjectId
from app.core.logging_config import logger

class BaseRepository:
    model: type
    collection_name: str

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db[self.collection_name]
        self.log = logger.bind(repo=self.collection_name)

    async def get_by_id(self, _id: str):
        doc = await self.collection.find_one({"_id": ObjectId(_id)})
        return self.model.model_validate(doc) if doc else None

    async def list(self, query: Dict, skip: int, limit: int, sort=None):
        cursor = self.collection.find(query).skip(skip).limit(limit)
        if sort: cursor = cursor.sort(sort)
        return [self.model.model_validate(d) async for d in cursor]

    async def create(self, data: Dict) -> Any:
        res = await self.collection.insert_one(data)
        return await self.get_by_id(res.inserted_id)

    async def update(self, _id: str, payload: Dict) -> bool:
        res = await self.collection.update_one({"_id": ObjectId(_id)}, {"$set": payload})
        return res.modified_count == 1