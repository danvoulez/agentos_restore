"""
Audit System for LogLineOS/DiamondSpan
Provides comprehensive audit trails for all system operations
Created: 2025-07-19 05:28:32 UTC
User: danvoulez
"""
import os
import json
import time
import uuid
import hashlib
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import sqlite3
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/audit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AuditSystem")

@dataclass
class AuditEntry:
    """A single audit entry"""
    id: str
    timestamp: str
    operation: str
    actor: str
    target_type: str
    target_id: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "actor": self.actor,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "status": self.status,
            "details": self.details,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEntry':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            operation=data["operation"],
            actor=data["actor"],
            target_type=data["target_type"],
            target_id=data["target_id"],
            status=data["status"],
            details=data.get("details", {}),
            metadata=data.get("metadata", {})
        )
    
    def get_hash(self) -> str:
        """Generate a deterministic hash for this entry"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

class AuditSystem:
    """
    Comprehensive audit system for LogLineOS
    """
    
    def __init__(self, db_path: str = "data/audit.db"):
        self.db_path = db_path
        self.initialized = False
        self.conn = None
        self.ensure_db_path()
    
    def ensure_db_path(self):
        """Ensure the database directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    async def initialize(self):
        """Initialize the audit system"""
        if self.initialized:
            return
        
        # Connect to database
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_entries (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                operation TEXT NOT NULL,
                actor TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT,
                metadata TEXT,
                entry_hash TEXT NOT NULL
            )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_entries (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_operation ON audit_entries (operation)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_actor ON audit_entries (actor)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_target ON audit_entries (target_type, target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON audit_entries (status)')
        
        self.conn.commit()
        self.initialized = True
        logger.info("Audit system initialized")
    
    @asynccontextmanager
    async def connection(self):
        """Get a database connection"""
        if not self.initialized:
            await self.initialize()
        
        # SQLite connections are not thread-safe, so create a new one for each operation
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    async def create_entry(self, 
                        operation: str,
                        actor: str,
                        target_type: str,
                        target_id: str,
                        status: str = "success",
                        details: Dict[str, Any] = None,
                        metadata: Dict[str, Any] = None) -> AuditEntry:
        """Create a new audit entry"""
        entry = AuditEntry(
            id=f"audit-{uuid.uuid4()}",
            timestamp=datetime.now().isoformat(),
            operation=operation,
            actor=actor,
            target_type=target_type,
            target_id=target_id,
            status=status,
            details=details or {},
            metadata=metadata or {}
        )
        
        # Calculate hash
        entry_hash = entry.get_hash()
        
        # Store in database
        async with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_entries 
                (id, timestamp, operation, actor, target_type, target_id, status, details, metadata, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.id,
                entry.timestamp,
                entry.operation,
                entry.actor,
                entry.target_type,
                entry.target_id,
                entry.status,
                json.dumps(entry.details),
                json.dumps(entry.metadata),
                entry_hash
            ))
        
        logger.info(f"Created audit entry: {entry.operation} on {entry.target_type}/{entry.target_id} by {entry.actor}")
        return entry
    
    async def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Get an audit entry by ID"""
        async with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM audit_entries WHERE id = ?', (entry_id,))
            row = cursor.fetchone()
            
            if row:
                return AuditEntry(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    operation=row['operation'],
                    actor=row['actor'],
                    target_type=row['target_type'],
                    target_id=row['target_id'],
                    status=row['status'],
                    details=json.loads(row['details']),
                    metadata=json.loads(row['metadata'])
                )
            
            return None
    
    async def query_entries(self, 
                          operations: List[str] = None,
                          actors: List[str] = None,
                          target_types: List[str] = None,
                          target_ids: List[str] = None,
                          statuses: List[str] = None,
                          start_time: str = None,
                          end_time: str = None,
                          limit: int = 100,
                          offset: int = 0) -> Tuple[List[AuditEntry], int]:
        """Query audit entries with filters"""
        query_parts = ['SELECT * FROM audit_entries WHERE 1=1']
        params = []
        
        # Add filters
        if operations:
            placeholders = ','.join(['?'] * len(operations))
            query_parts.append(f'AND operation IN ({placeholders})')
            params.extend(operations)
        
        if actors:
            placeholders = ','.join(['?'] * len(actors))
            query_parts.append(f'AND actor IN ({placeholders})')
            params.extend(actors)
        
        if target_types:
            placeholders = ','.join(['?'] * len(target_types))
            query_parts.append(f'AND target_type IN ({placeholders})')
            params.extend(target_types)
        
        if target_ids:
            placeholders = ','.join(['?'] * len(target_ids))
            query_parts.append(f'AND target_id IN ({placeholders})')
            params.extend(target_ids)
        
        if statuses:
            placeholders = ','.join(['?'] * len(statuses))
            query_parts.append(f'AND status IN ({placeholders})')
            params.extend(statuses)
        
        if start_time:
            query_parts.append('AND timestamp >= ?')
            params.append(start_time)
        
        if end_time:
            query_parts.append('AND timestamp <= ?')
            params.append(end_time)
        
        # Get total count first
        count_query = ' '.join(['SELECT COUNT(*) as count FROM audit_entries WHERE'] + query_parts[1:])
        
        # Add order and pagination
        query_parts.append('ORDER BY timestamp DESC')
        query_parts.append('LIMIT ? OFFSET ?')
        params.extend([limit, offset])
        
        query = ' '.join(query_parts)
        
        async with self.connection() as conn:
            # Get total count
            cursor = conn.cursor()
            cursor.execute(count_query, params[:-2])
            total_count = cursor.fetchone()['count']
            
            # Get paginated results
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            entries = []
            for row in rows:
                entries.append(AuditEntry(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    operation=row['operation'],
                    actor=row['actor'],
                    target_type=row['target_type'],
                    target_id=row['target_id'],
                    status=row['status'],
                    details=json.loads(row['details']),
                    metadata=json.loads(row['metadata'])
                ))
        
        return entries, total_count
    
    async def validate_audit_trail(self, 
                                 target_type: str = None, 
                                 target_id: str = None,
                                 start_time: str = None,
                                 end_time: str = None) -> Dict[str, Any]:
        """Validate the integrity of the audit trail"""
        # Build query based on filters
        query_parts = ['SELECT * FROM audit_entries WHERE 1=1']
        params = []
        
        if target_type:
            query_parts.append('AND target_type = ?')
            params.append(target_type)
        
        if target_id:
            query_parts.append('AND target_id = ?')
            params.append(target_id)
        
        if start_time:
            query_parts.append('AND timestamp >= ?')
            params.append(start_time)
        
        if end_time:
            query_parts.append('AND timestamp <= ?')
            params.append(end_time)
        
        query_parts.append('ORDER BY timestamp ASC')
        query = ' '.join(query_parts)
        
        async with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            valid_count = 0
            invalid_entries = []
            
            for row in rows:
                # Reconstruct the entry and compute its hash
                entry = AuditEntry(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    operation=row['operation'],
                    actor=row['actor'],
                    target_type=row['target_type'],
                    target_id=row['target_id'],
                    status=row['status'],
                    details=json.loads(row['details']),
                    metadata=json.loads(row['metadata'])
                )
                
                computed_hash = entry.get_hash()
                stored_hash = row['entry_hash']
                
                if computed_hash != stored_hash:
                    invalid_entries.append({
                        "id": entry.id,
                        "timestamp": entry.timestamp,
                        "stored_hash": stored_hash,
                        "computed_hash": computed_hash
                    })
                else:
                    valid_count += 1
        
        result = {
            "total_entries": len(rows),
            "valid_entries": valid_count,
            "invalid_entries": len(invalid_entries),
            "invalid_details": invalid_entries if invalid_entries else None,
            "valid": len(invalid_entries) == 0,
            "validation_time": datetime.now().isoformat()
        }
        
        logger.info(f"Audit trail validation: {result['valid_entries']}/{result['total_entries']} entries valid")
        
        return result
    
    async def create_system_snapshot(self) -> str:
        """Create a system snapshot for forensic purposes"""
        snapshot_id = f"snapshot-{int(time.time())}"
        snapshot_dir = f"data/audit/snapshots/{snapshot_id}"
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Copy the audit database
        db_snapshot_path = os.path.join(snapshot_dir, "audit.db")
        
        # Use a new connection to create the snapshot
        source_conn = sqlite3.connect(self.db_path)
        dest_conn = sqlite3.connect(db_snapshot_path)
        
        source_conn.backup(dest_conn)
        
        source_conn.close()
        dest_conn.close()
        
        # Create a metadata file
        metadata = {
            "id": snapshot_id,
            "created_at": datetime.now().isoformat(),
            "source_db": self.db_path,
            "entries_count": await self._get_entry_count(),
            "snapshot_reason": "System audit",
            "hash": await self._calculate_db_hash()
        }
        
        with open(os.path.join(snapshot_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created system snapshot: {snapshot_id}")
        return snapshot_id
    
    async def _get_entry_count(self) -> int:
        """Get the total number of audit entries"""
        async with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM audit_entries')
            return cursor.fetchone()['count']
    
    async def _calculate_db_hash(self) -> str:
        """Calculate a hash of the entire database"""
        async with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT entry_hash FROM audit_entries ORDER BY timestamp ASC')
            hashes = [row['entry_hash'] for row in cursor.fetchall()]
            
            if not hashes:
                return "empty_db"
            
            combined_hash = hashlib.sha256(''.join(hashes).encode()).hexdigest()
            return combined_hash
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get audit system statistics"""
        async with self.connection() as conn:
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute('SELECT COUNT(*) as count FROM audit_entries')
            total_entries = cursor.fetchone()['count']
            
            # Entries by operation
            cursor.execute('''
                SELECT operation, COUNT(*) as count 
                FROM audit_entries 
                GROUP BY operation 
                ORDER BY count DESC
            ''')
            operations = {row['operation']: row['count'] for row in cursor.fetchall()}
            
            # Entries by actor (top 10)
            cursor.execute('''
                SELECT actor, COUNT(*) as count 
                FROM audit_entries 
                GROUP BY actor 
                ORDER BY count DESC
                LIMIT 10
            ''')
            actors = {row['actor']: row['count'] for row in cursor.fetchall()}
            
            # Entries by status
            cursor.execute('''
                SELECT status, COUNT(*) as count 
                FROM audit_entries 
                GROUP BY status 
                ORDER BY count DESC
            ''')
            statuses = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Entries by day (last 7 days)
            seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute('''
                SELECT 
                    substr(timestamp, 1, 10) as day, 
                    COUNT(*) as count 
                FROM audit_entries 
                WHERE timestamp >= ?
                GROUP BY day 
                ORDER BY day ASC
            ''', (seven_days_ago,))
            daily_counts = {row['day']: row['count'] for row in cursor.fetchall()}
        
        return {
            "total_entries": total_entries,
            "by_operation": operations,
            "by_actor": actors,
            "by_status": statuses,
            "daily_counts": daily_counts,
            "generated_at": datetime.now().isoformat()
        }