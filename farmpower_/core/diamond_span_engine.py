"""
Diamond Span Engine for LogLineOS
Unified engine for Diamond Span operations
Created: 2025-07-19 05:47:12 UTC
User: danvoulez
"""
import os
import json
import time
import logging
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

try:
    from core.diamond_span import DiamondSpan
except ImportError:
    # Create a minimal version if not available
    from dataclasses import dataclass, field
    
    @dataclass
    class DiamondSpan:
        id: str = None
        parent_ids: List[str] = field(default_factory=list)
        content: Dict[str, Any] = field(default_factory=dict)
        metadata: Dict[str, Any] = field(default_factory=dict)
        energy: float = 0.0
        created_at: datetime = field(default_factory=datetime.now)
        
        def is_exempt(self) -> bool:
            return self.metadata.get("is_exempt", False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/diamond_span_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DiamondSpanEngine")

class DiamondSpanEngine:
    """
    Unified engine for Diamond Span operations
    """
    
    def __init__(self):
        self.spans: Dict[str, DiamondSpan] = {}
        self.index_by_kind: Dict[str, List[str]] = {}
        self.index_by_actor: Dict[str, List[str]] = {}
        self.causal_graph: Dict[str, List[str]] = {}  # parent -> children
        self.last_operation_time = time.time()
        self.stats = {
            "total_spans": 0,
            "total_energy": 0.0,
            "span_kinds": {},
            "causal_connections": 0,
            "operations": {
                "create": 0,
                "read": 0,
                "update": 0,
                "delete": 0,
                "query": 0
            }
        }
    
    async def create_span(self, span: Union[DiamondSpan, Dict[str, Any]]) -> DiamondSpan:
        """
        Create a new Diamond Span
        
        Args:
            span: DiamondSpan object or dictionary representation
            
        Returns:
            The created DiamondSpan
        """
        # Convert dict to DiamondSpan if needed
        if isinstance(span, dict):
            span_obj = self._dict_to_span(span)
        else:
            span_obj = span
        
        # Generate ID if not provided
        if not span_obj.id:
            span_obj.id = self._generate_id(span_obj)
        
        # Calculate energy if not provided
        if not span_obj.energy:
            span_obj.energy = self._calculate_energy(span_obj)
        
        # Set creation time if not provided
        if not span_obj.created_at:
            span_obj.created_at = datetime.now()
        
        # Store the span
        self.spans[span_obj.id] = span_obj
        
        # Update indexes
        self._update_indexes(span_obj)
        
        # Update causal graph
        self._update_causal_graph(span_obj)
        
        # Update statistics
        self._update_stats_on_create(span_obj)
        
        logger.info(f"Created span: {span_obj.id} ({self._get_span_kind(span_obj)})")
        return span_obj
    
    async def get_span(self, span_id: str) -> Optional[DiamondSpan]:
        """
        Get a span by ID
        
        Args:
            span_id: ID of the span to retrieve
            
        Returns:
            The DiamondSpan or None if not found
        """
        # Update read stats
        self.stats["operations"]["read"] += 1
        self.last_operation_time = time.time()
        
        return self.spans.get(span_id)
    
    async def update_span(self, span_id: str, updates: Dict[str, Any]) -> Optional[DiamondSpan]:
        """
        Update a span
        
        Args:
            span_id: ID of the span to update
            updates: Dictionary of updates to apply
            
        Returns:
            The updated DiamondSpan or None if not found
        """
        if span_id not in self.spans:
            return None
        
        span = self.spans[span_id]
        
        # Apply updates to metadata (only metadata can be updated)
        if "metadata" in updates:
            span.metadata.update(updates["metadata"])
        
        # Update stats
        self.stats["operations"]["update"] += 1
        self.last_operation_time = time.time()
        
        logger.info(f"Updated span: {span_id}")
        return span
    
    async def delete_span(self, span_id: str) -> bool:
        """
        Delete a span
        
        Args:
            span_id: ID of the span to delete
            
        Returns:
            True if deleted, False if not found
        """
        if span_id not in self.spans:
            return False
        
        span = self.spans[span_id]
        
        # Remove from indexes
        kind = self._get_span_kind(span)
        actor = self._get_span_actor(span)
        
        if kind in self.index_by_kind and span_id in self.index_by_kind[kind]:
            self.index_by_kind[kind].remove(span_id)
        
        if actor in self.index_by_actor and span_id in self.index_by_actor[actor]:
            self.index_by_actor[actor].remove(span_id)
        
        # Remove from causal graph
        for parent_id in span.parent_ids:
            if parent_id in self.causal_graph and span_id in self.causal_graph[parent_id]:
                self.causal_graph[parent_id].remove(span_id)
        
        # Delete children references
        if span_id in self.causal_graph:
            del self.causal_graph[span_id]
        
        # Update stats
        self.stats["total_spans"] -= 1
        self.stats["total_energy"] -= span.energy
        if kind in self.stats["span_kinds"]:
            self.stats["span_kinds"][kind] -= 1
        
        self.stats["operations"]["delete"] += 1
        self.last_operation_time = time.time()
        
        # Remove from spans
        del self.spans[span_id]
        
        logger.info(f"Deleted span: {span_id}")
        return True
    
    async def query_spans(self, 
                        kinds: List[str] = None,
                        actors: List[str] = None,
                        parent_ids: List[str] = None,
                        limit: int = 100,
                        offset: int = 0,
                        sort_by: str = "created_at",
                        descending: bool = True) -> Tuple[List[DiamondSpan], int]:
        """
        Query spans with filters
        
        Args:
            kinds: Filter by span kinds
            actors: Filter by actors
            parent_ids: Filter by parent IDs
            limit: Maximum number of results
            offset: Pagination offset
            sort_by: Field to sort by
            descending: Sort in descending order
            
        Returns:
            Tuple of (matching spans, total count)
        """
        # Update query stats
        self.stats["operations"]["query"] += 1
        self.last_operation_time = time.time()
        
        # Start with all spans
        all_span_ids = set(self.spans.keys())
        
        # Apply kind filter
        if kinds:
            kind_span_ids = set()
            for kind in kinds:
                if kind in self.index_by_kind:
                    kind_span_ids.update(self.index_by_kind[kind])
            all_span_ids &= kind_span_ids
        
        # Apply actor filter
        if actors:
            actor_span_ids = set()
            for actor in actors:
                if actor in self.index_by_actor:
                    actor_span_ids.update(self.index_by_actor[actor])
            all_span_ids &= actor_span_ids
        
        # Apply parent filter
        if parent_ids:
            parent_span_ids = set()
            for parent_id in parent_ids:
                if parent_id in self.causal_graph:
                    parent_span_ids.update(self.causal_graph[parent_id])
            all_span_ids &= parent_span_ids
        
        # Get matching spans
        matching_spans = [self.spans[span_id] for span_id in all_span_ids]
        
        # Sort results
        if sort_by == "energy":
            matching_spans.sort(key=lambda s: s.energy, reverse=descending)
        elif sort_by == "created_at":
            matching_spans.sort(key=lambda s: s.created_at, reverse=descending)
        else:
            # Default to created_at
            matching_spans.sort(key=lambda s: s.created_at, reverse=descending)
        
        # Apply pagination
        total_count = len(matching_spans)
        paginated_spans = matching_spans[offset:offset+limit]
        
        return paginated_spans, total_count
    
    async def get_causal_children(self, span_id: str) -> List[DiamondSpan]:
        """
        Get all spans that have this span as a parent
        
        Args:
            span_id: ID of the parent span
            
        Returns:
            List of child spans
        """
        if span_id not in self.causal_graph:
            return []
        
        child_ids = self.causal_graph[span_id]
        return [self.spans[cid] for cid in child_ids if cid in self.spans]
    
    async def get_causal_ancestors(self, span_id: str, max_depth: int = 10) -> List[DiamondSpan]:
        """
        Get all ancestors of a span up to max_depth
        
        Args:
            span_id: ID of the span
            max_depth: Maximum recursion depth
            
        Returns:
            List of ancestor spans
        """
        if span_id not in self.spans:
            return []
        
        # Use breadth-first search to find ancestors
        visited = set()
        queue = list(self.spans[span_id].parent_ids)
        ancestors = []
        depth = 0
        
        while queue and depth < max_depth:
            depth += 1
            level_size = len(queue)
            
            for _ in range(level_size):
                parent_id = queue.pop(0)
                
                if parent_id in visited:
                    continue
                
                visited.add(parent_id)
                
                if parent_id in self.spans:
                    ancestors.append(self.spans[parent_id])
                    # Add this parent's parents to the queue
                    queue.extend([pid for pid in self.spans[parent_id].parent_ids if pid not in visited])
        
        return ancestors
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        stats = {**self.stats}
        stats["timestamp"] = datetime.now().isoformat()
        stats["uptime"] = time.time() - self.last_operation_time
        return stats
    
    def _dict_to_span(self, span_dict: Dict[str, Any]) -> DiamondSpan:
        """Convert dictionary to DiamondSpan object"""
        # Extract standard fields
        span_id = span_dict.get("id")
        parent_ids = span_dict.get("parent_ids", [])
        
        # Handle different content formats
        if "content" in span_dict:
            content = span_dict["content"]
        elif "payload" in span_dict:
            content = {"payload": span_dict["payload"]}
            # If we have verb/actor/object, include them in content
            if "verb" in span_dict:
                content["verb"] = span_dict["verb"]
            if "actor" in span_dict or "who" in span_dict:
                content["actor"] = span_dict.get("actor") or span_dict.get("who")
            if "object" in span_dict:
                content["object"] = span_dict["object"]
        else:
            content = {}
        
        # Extract metadata
        metadata = span_dict.get("metadata", {})
        
        # Extract energy if present
        energy = span_dict.get("energy", 0.0)
        
        # Handle created_at
        created_at_str = span_dict.get("created_at")
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except:
                created_at = datetime.now()
        else:
            created_at = datetime.now()
        
        return DiamondSpan(
            id=span_id,
            parent_ids=parent_ids,
            content=content,
            metadata=metadata,
            energy=energy,
            created_at=created_at
        )
    
    def _generate_id(self, span: DiamondSpan) -> str:
        """Generate a unique ID for a span"""
        # Get kind for prefix
        kind = self._get_span_kind(span)
        kind_prefix = kind[:4].lower() if kind else "span"
        
        # Create timestamp component
        timestamp = int(time.time() * 1000)
        
        # Create hash component based on content
        content_str = json.dumps(span.content, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:8]
        
        return f"{kind_prefix}-{timestamp}-{content_hash}"
    
    def _calculate_energy(self, span: DiamondSpan) -> float:
        """Calculate energy for a span"""
        # Base energy
        base = 10.0
        
        # Content complexity factor
        content_str = json.dumps(span.content)
        complexity = len(content_str) / 1000
        complexity_factor = min(3.0, 1.0 + complexity)
        
        # Parent connections factor
        parent_factor = 1.0 + (len(span.parent_ids) * 0.15)
        
        # Kind-specific factors
        kind = self._get_span_kind(span)
        kind_factor = {
            "genesis": 5.0,
            "constitution": 4.0,
            "governance_policy": 3.0,
            "diamond": 2.0,
            "train": 1.8,
            "fine_tune": 1.5,
            "simulate": 1.2
        }.get(kind, 1.0)
        
        # Calculate final energy
        energy = base * complexity_factor * parent_factor * kind_factor
        
        # Cap at reasonable value
        return min(100.0, energy)
    
    def _get_span_kind(self, span: DiamondSpan) -> str:
        """Extract kind from a span"""
        # Try different field locations
        if isinstance(span.content, dict):
            if "kind" in span.content:
                return span.content["kind"]
            if "payload" in span.content and isinstance(span.content["payload"], dict):
                if "kind" in span.content["payload"]:
                    return span.content["payload"]["kind"]
        
        # Check metadata
        if "kind" in span.metadata:
            return span.metadata["kind"]
        
        # Default kind
        return "diamond"
    
    def _get_span_actor(self, span: DiamondSpan) -> str:
        """Extract actor from a span"""
        # Try different field locations
        if isinstance(span.content, dict):
            if "actor" in span.content:
                return span.content["actor"]
            if "who" in span.content:
                return span.content["who"]
        
        # Check metadata
        if "actor" in span.metadata:
            return span.metadata["actor"]
        if "creator" in span.metadata:
            return span.metadata["creator"]
        
        # Default actor
        return "system"
    
    def _update_indexes(self, span: DiamondSpan) -> None:
        """Update index structures for a span"""
        # Kind index
        kind = self._get_span_kind(span)
        if kind not in self.index_by_kind:
            self.index_by_kind[kind] = []
        if span.id not in self.index_by_kind[kind]:
            self.index_by_kind[kind].append(span.id)
        
        # Actor index
        actor = self._get_span_actor(span)
        if actor not in self.index_by_actor:
            self.index_by_actor[actor] = []
        if span.id not in self.index_by_actor[actor]:
            self.index_by_actor[actor].append(span.id)
    
    def _update_causal_graph(self, span: DiamondSpan) -> None:
        """Update causal graph with a span"""
        # Add connections from parents to this span
        for parent_id in span.parent_ids:
            if parent_id not in self.causal_graph:
                self.causal_graph[parent_id] = []
            if span.id not in self.causal_graph[parent_id]:
                self.causal_graph[parent_id].append(span.id)
                self.stats["causal_connections"] += 1
    
    def _update_stats_on_create(self, span: DiamondSpan) -> None:
        """Update statistics when a new span is created"""
        self.stats["total_spans"] += 1
        self.stats["total_energy"] += span.energy
        
        kind = self._get_span_kind(span)
        if kind not in self.stats["span_kinds"]:
            self.stats["span_kinds"][kind] = 0
        self.stats["span_kinds"][kind] += 1
        
        self.stats["operations"]["create"] += 1
        self.last_operation_time = time.time()