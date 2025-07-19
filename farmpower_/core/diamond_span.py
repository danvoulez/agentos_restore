import uuid, hashlib, os
from datetime import datetime

class DiamondSpan:
    def __init__(self, id=None, parent_ids=None, content=None, metadata=None):
        self.id = id or str(uuid.uuid4())
        self.parent_ids = parent_ids or []
        self.content = content or {}
        self.metadata = metadata or {}
        self.energy = self.metadata.get("energy", 0)
        self.signature = self._sign()
        self.created_at = datetime.utcnow()
        self.decay_rate = self.metadata.get("decay_rate", 0.01)
    def _sign(self):
        base = f"{self.id}{str(self.content)}"
        return hashlib.sha256(base.encode()).hexdigest()
    def validate(self, ledger):
        return all(pid in ledger for pid in self.parent_ids)
    def decay(self):
        if not self.is_exempt():
            self.energy *= (1 - self.decay_rate)
    def is_exempt(self):
        return self.metadata.get("creator") == os.getenv("GOD_KEY")