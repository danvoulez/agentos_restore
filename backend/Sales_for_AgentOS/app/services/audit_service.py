from datetime import datetime, timezone
from app.db.repositories.sale_repository import SaleRepository
from app.core.logging_config import logger

class AuditService:
    def __init__(self, sale_repo: SaleRepository):
        self.repo = sale_repo
        self.log = logger.bind(service="AuditService")

    async def log_sale_action(self, sale_id: str, actor_id: str, action: str, detail: str | None = None):
        entry = {
            "timestamp": datetime.now(timezone.utc),
            "actor_id": actor_id,
            "action": action,
            "detail": detail
        }
        await self.repo.collection.update_one(
            {"_id": sale_id},
            {"$push": {"audit_log": entry}}
        )
        self.log.info(f"audit logged {action} by {actor_id} on {sale_id}")