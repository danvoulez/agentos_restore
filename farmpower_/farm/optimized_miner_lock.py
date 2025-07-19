"""
Optimized Locking System for Diamond Span Miner
Improves concurrency control and reduces lock contention
Created: 2025-07-19 06:33:24 UTC
User: danvoulez
"""
import asyncio
import threading
import time
import logging
from typing import Dict, Any, Set, Optional
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/miner_locks.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MinerLocks")

class OptimizedMinerLock:
    """
    Optimized locking system for diamond span miner
    Provides fine-grained locking to reduce contention
    """
    
    def __init__(self):
        """Initialize the locking system"""
        # Different lock groups for different resources
        self.pending_locks = {}      # Locks for pending spans queue
        self.mined_locks = {}        # Locks for mined spans queue
        self.span_locks = {}         # Locks for individual spans
        self.parent_locks = {}       # Locks for parent span relationships
        
        # Global lock for the lock registries themselves
        self.meta_lock = threading.RLock()
        
        # Stats
        self.stats = {
            "pending_locks_created": 0,
            "mined_locks_created": 0,
            "span_locks_created": 0,
            "parent_locks_created": 0,
            "lock_acquisitions": 0,
            "lock_contentions": 0
        }
        
        logger.info("OptimizedMinerLock initialized")
    
    def get_pending_lock(self, priority_range: str) -> threading.Lock:
        """
        Get a lock for a specific priority range in the pending queue
        This allows concurrent operations on different priority ranges
        
        Args:
            priority_range: Priority range identifier (e.g., "high", "medium", "low")
            
        Returns:
            Lock for that priority range
        """
        with self.meta_lock:
            if priority_range not in self.pending_locks:
                self.pending_locks[priority_range] = threading.RLock()
                self.stats["pending_locks_created"] += 1
            
            return self.pending_locks[priority_range]
    
    def get_mined_lock(self, batch_id: str) -> threading.Lock:
        """
        Get a lock for a specific batch in the mined queue
        This allows concurrent operations on different batches
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Lock for that batch
        """
        with self.meta_lock:
            if batch_id not in self.mined_locks:
                self.mined_locks[batch_id] = threading.RLock()
                self.stats["mined_locks_created"] += 1
            
            return self.mined_locks[batch_id]
    
    def get_span_lock(self, span_id: str) -> threading.Lock:
        """
        Get a lock for a specific span
        This allows concurrent operations on different spans
        
        Args:
            span_id: Span identifier
            
        Returns:
            Lock for that span
        """
        with self.meta_lock:
            if span_id not in self.span_locks:
                self.span_locks[span_id] = threading.RLock()
                self.stats["span_locks_created"] += 1
            
            return self.span_locks[span_id]
    
    def get_parent_lock(self, parent_id: str) -> threading.Lock:
        """
        Get a lock for a specific parent relationship
        This allows concurrent operations on different parent spans
        
        Args:
            parent_id: Parent span identifier
            
        Returns:
            Lock for that parent relationship
        """
        with self.meta_lock:
            if parent_id not in self.parent_locks:
                self.parent_locks[parent_id] = threading.RLock()
                self.stats["parent_locks_created"] += 1
            
            return self.parent_locks[parent_id]
    
    def acquire_span_lock(self, span_id: str, timeout: float = None) -> bool:
        """
        Acquire a lock for a specific span
        
        Args:
            span_id: Span identifier
            timeout: Optional timeout in seconds
            
        Returns:
            True if lock acquired, False if timeout
        """
        lock = self.get_span_lock(span_id)
        
        start_time = time.time()
        acquired = lock.acquire(timeout=timeout)
        
        if acquired:
            with self.meta_lock:
                self.stats["lock_acquisitions"] += 1
        else:
            with self.meta_lock:
                self.stats["lock_contentions"] += 1
        
        return acquired
    
    def release_span_lock(self, span_id: str):
        """
        Release a lock for a specific span
        
        Args:
            span_id: Span identifier
        """
        with self.meta_lock:
            if span_id in self.span_locks:
                try:
                    self.span_locks[span_id].release()
                except RuntimeError:
                    # Not holding the lock
                    pass
    
    def acquire_multi_locks(self, lock_ids: Set[str], lock_type: str, timeout: float = None) -> bool:
        """
        Acquire multiple locks in a deadlock-safe manner
        
        Args:
            lock_ids: Set of lock identifiers
            lock_type: Type of locks ("span", "parent", "pending", "mined")
            timeout: Optional timeout in seconds
            
        Returns:
            True if all locks acquired, False otherwise
        """
        # Sort lock IDs to prevent deadlocks
        sorted_ids = sorted(lock_ids)
        
        # Get the appropriate lock getter function
        if lock_type == "span":
            get_lock = self.get_span_lock
        elif lock_type == "parent":
            get_lock = self.get_parent_lock
        elif lock_type == "pending":
            get_lock = self.get_pending_lock
        elif lock_type == "mined":
            get_lock = self.get_mined_lock
        else:
            raise ValueError(f"Invalid lock type: {lock_type}")
        
        # Calculate per-lock timeout if total timeout is specified
        per_lock_timeout = timeout / len(sorted_ids) if timeout else None
        
        # Acquire locks in order
        acquired_locks = []
        
        try:
            for lock_id in sorted_ids:
                lock = get_lock(lock_id)
                
                acquired = lock.acquire(timeout=per_lock_timeout)
                if not acquired:
                    # Failed to acquire this lock
                    with self.meta_lock:
                        self.stats["lock_contentions"] += 1
                    
                    # Release any locks we've acquired so far
                    for acquired_id in acquired_locks:
                        acquired_lock = get_lock(acquired_id)
                        acquired_lock.release()
                    
                    return False
                
                acquired_locks.append(lock_id)
            
            # Successfully acquired all locks
            with self.meta_lock:
                self.stats["lock_acquisitions"] += len(sorted_ids)
            
            return True
            
        except Exception as e:
            # Release any locks we've acquired so far
            for acquired_id in acquired_locks:
                acquired_lock = get_lock(acquired_id)
                try:
                    acquired_lock.release()
                except:
                    pass
            
            logger.error(f"Error acquiring multiple locks: {str(e)}")
            return False
    
    def release_multi_locks(self, lock_ids: Set[str], lock_type: str):
        """
        Release multiple locks
        
        Args:
            lock_ids: Set of lock identifiers
            lock_type: Type of locks ("span", "parent", "pending", "mined")
        """
        # Get the appropriate lock getter function
        if lock_type == "span":
            get_lock = self.get_span_lock
        elif lock_type == "parent":
            get_lock = self.get_parent_lock
        elif lock_type == "pending":
            get_lock = self.get_pending_lock
        elif lock_type == "mined":
            get_lock = self.get_mined_lock
        else:
            raise ValueError(f"Invalid lock type: {lock_type}")
        
        # Release locks in reverse order
        for lock_id in sorted(lock_ids, reverse=True):
            try:
                lock = get_lock(lock_id)
                lock.release()
            except RuntimeError:
                # Not holding this lock
                pass
            except Exception as e:
                logger.error(f"Error releasing lock {lock_id}: {str(e)}")
    
    def cleanup(self):
        """Clean up locks that are no longer needed"""
        with self.meta_lock:
            # This would remove locks that haven't been used for a while
            # For now, just log the current counts
            logger.info(f"Lock counts: pending={len(self.pending_locks)}, "
                       f"mined={len(self.mined_locks)}, span={len(self.span_locks)}, "
                       f"parent={len(self.parent_locks)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get lock statistics
        
        Returns:
            Dictionary of statistics
        """
        with self.meta_lock:
            stats = self.stats.copy()
            stats.update({
                "pending_lock_count": len(self.pending_locks),
                "mined_lock_count": len(self.mined_locks),
                "span_lock_count": len(self.span_locks),
                "parent_lock_count": len(self.parent_locks),
                "contention_rate": stats["lock_contentions"] / max(1, stats["lock_acquisitions"] + stats["lock_contentions"])
            })
            return stats

# Example optimized mining worker implementation
class OptimizedMiningWorker:
    """
    Optimized mining worker that uses fine-grained locking
    """
    
    def __init__(self, worker_id: int, lock_system: OptimizedMinerLock):
        """
        Initialize mining worker
        
        Args:
            worker_id: Worker identifier
            lock_system: Optimized lock system
        """
        self.worker_id = worker_id
        self.locks = lock_system
        self.running = False
        
        logger.info(f"OptimizedMiningWorker {worker_id} initialized")
    
    def mine_span(self, span_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mine a span with optimized locking
        
        Args:
            span_data: Span data to mine
            
        Returns:
            Mining result
        """
        span_id = span_data.get("id", str(uuid.uuid4()))
        parent_ids = span_data.get("parent_ids", [])
        
        # First, acquire the span lock
        if not self.locks.acquire_span_lock(span_id, timeout=1.0):
            logger.warning(f"Worker {self.worker_id}: Failed to acquire lock for span {span_id}")
            return {"status": "failed", "reason": "Lock acquisition failed"}
        
        try:
            # Then, acquire parent locks if needed
            if parent_ids:
                parent_locks_acquired = self.locks.acquire_multi_locks(
                    set(parent_ids), 
                    lock_type="parent",
                    timeout=2.0
                )
                
                if not parent_locks_acquired:
                    logger.warning(f"Worker {self.worker_id}: Failed to acquire parent locks for span {span_id}")
                    return {"status": "failed", "reason": "Parent lock acquisition failed"}
            
            try:
                # Perform the actual mining operation
                # This would be the CPU-intensive part
                result = self._perform_mining(span_data)
                
                return result
                
            finally:
                # Release parent locks if acquired
                if parent_ids:
                    self.locks.release_multi_locks(set(parent_ids), lock_type="parent")
                
        finally:
            # Always release the span lock
            self.locks.release_span_lock(span_id)
    
    def _perform_mining(self, span_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform the actual mining operation
        This is a placeholder for the CPU-intensive work
        
        Args:
            span_data: Span data to mine
            
        Returns:
            Mining result
        """
        # This would be the actual mining logic
        # For now, just simulate success with 90% probability
        import random
        if random.random() < 0.9:
            return {
                "status": "success",
                "span": {**span_data, "mined": True},
                "mining_time": 0.1,
                "reward": 10.0
            }
        else:
            return {
                "status": "failed",
                "reason": "Mining algorithm failed",
                "mining_time": 0.1
            }

# Example usage:
# lock_system = OptimizedMinerLock()
# worker = OptimizedMiningWorker(1, lock_system)
# result = worker.mine_span({"id": "span-123", "parent_ids": ["parent-1", "parent-2"]})