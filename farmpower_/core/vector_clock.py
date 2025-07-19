"""
Vector Clock implementation for LogLineOS/DiamondSpan
Provides causal ordering and distributed timestamps
"""
import time
import json
import uuid
from typing import Dict, Any, List, Optional

class VectorClock:
    """
    Vector Clock for causal ordering of spans
    """
    def __init__(self, clock_dict: Dict[str, int] = None, node_id: str = None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.clock = clock_dict or {self.node_id: 0}
        self.timestamp = time.time()
    
    def tick(self):
        """Increment local logical clock"""
        self.clock[self.node_id] = self.clock.get(self.node_id, 0) + 1
        self.timestamp = time.time()
        return self
    
    def update(self, other_clock: Dict[str, int]):
        """Update with another vector clock"""
        for node, count in other_clock.items():
            self.clock[node] = max(self.clock.get(node, 0), count)
        return self
    
    def merge(self, other_clock: 'VectorClock'):
        """Merge with another VectorClock"""
        self.update(other_clock.clock)
        self.timestamp = max(self.timestamp, other_clock.timestamp)
        return self
    
    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if this clock happens before another"""
        if self == other:
            return False
        
        for node, count in self.clock.items():
            if node in other.clock and count > other.clock[node]:
                return False
        
        return True
    
    def concurrent_with(self, other: 'VectorClock') -> bool:
        """Check if this clock is concurrent with another"""
        return not (self.happens_before(other) or other.happens_before(self))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "node_id": self.node_id,
            "clock": self.clock,
            "timestamp": self.timestamp
        }
    
    def __eq__(self, other):
        if not isinstance(other, VectorClock):
            return False
        return self.clock == other.clock
    
    def __str__(self):
        return f"VectorClock({self.node_id}, {self.clock}, {self.timestamp})"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorClock':
        """Create from dictionary"""
        return cls(
            clock_dict=data["clock"],
            node_id=data["node_id"]
        )