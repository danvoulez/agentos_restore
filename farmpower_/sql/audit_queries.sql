-- LogLineOS Audit System SQL Queries
-- Created: 2025-07-19 05:28:32 UTC
-- User: danvoulez

-- Query 1: Get all failed operations in the last 24 hours
SELECT 
    id, 
    timestamp, 
    operation, 
    actor, 
    target_type, 
    target_id, 
    status, 
    details
FROM audit_entries
WHERE 
    status = 'failure' 
    AND timestamp >= datetime('now', '-1 day')
ORDER BY timestamp DESC;

-- Query 2: Get operations by specific actor
SELECT 
    operation, 
    COUNT(*) as count, 
    MIN(timestamp) as first_seen,
    MAX(timestamp) as last_seen
FROM audit_entries
WHERE actor = ?  -- Replace with actor name
GROUP BY operation
ORDER BY count DESC;

-- Query 3: Get operation history for a specific span
SELECT 
    id, 
    timestamp, 
    operation, 
    actor, 
    status, 
    details
FROM audit_entries
WHERE 
    target_type = 'span' 
    AND target_id = ?  -- Replace with span ID
ORDER BY timestamp ASC;

-- Query 4: Get all operations on a specific target type in a time window
SELECT 
    id, 
    timestamp, 
    operation, 
    actor, 
    target_id, 
    status
FROM audit_entries
WHERE 
    target_type = ? 
    AND timestamp BETWEEN ? AND ?
ORDER BY timestamp DESC;

-- Query 5: Get operations by hour of day (for pattern analysis)
SELECT 
    SUBSTR(timestamp, 12, 2) as hour,
    COUNT(*) as operation_count
FROM audit_entries
GROUP BY hour
ORDER BY hour;

-- Query 6: Get suspicious operations (multiple failures by same actor)
SELECT 
    actor,
    operation,
    COUNT(*) as failure_count
FROM audit_entries
WHERE 
    status = 'failure'
    AND timestamp >= datetime('now', '-7 day')
GROUP BY actor, operation
HAVING COUNT(*) >= 5
ORDER BY failure_count DESC;

-- Query 7: Get system-wide success rate by operation
SELECT 
    operation,
    COUNT(*) as total_count,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
    ROUND(100.0 * SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate
FROM audit_entries
GROUP BY operation
ORDER BY total_count DESC;

-- Query 8: Get all operations on critical target types
SELECT 
    id, 
    timestamp, 
    operation, 
    actor, 
    target_type, 
    target_id, 
    status
FROM audit_entries
WHERE 
    target_type IN ('constitution', 'governance_policy', 'emergency_action', 'kill_switch')
    AND timestamp >= datetime('now', '-30 day')
ORDER BY timestamp DESC;

-- Query 9: Get all operations by unknown actors
SELECT 
    id, 
    timestamp, 
    operation,  
    actor, 
    target_type, 
    target_id, 
    status,
    details
FROM audit_entries
WHERE 
    actor NOT IN ('system', 'danvoulez', 'admin')
    AND timestamp >= datetime('now', '-7 day')
ORDER BY timestamp DESC;

-- Query 10: Audit trail validation query
SELECT 
    id,
    entry_hash as stored_hash,
    (
        SELECT hex(sha256(
            json_object(
                'id', id,
                'timestamp', timestamp,
                'operation', operation,
                'actor', actor,
                'target_type', target_type,
                'target_id', target_id,
                'status', status,
                'details', details,
                'metadata', metadata
            )
        )) as computed_hash
        FROM audit_entries ae2
        WHERE ae2.id = ae1.id
    ) as computed_hash
FROM audit_entries ae1
WHERE stored_hash != computed_hash;

-- Query 11: Get most active day in audit history
SELECT 
    SUBSTR(timestamp, 1, 10) as day,
    COUNT(*) as operation_count
FROM audit_entries
GROUP BY day
ORDER BY operation_count DESC
LIMIT 1;

-- Query 12: Get operations with large payloads (potential abuse)
SELECT 
    id, 
    timestamp, 
    operation, 
    actor,
    target_type,
    target_id,
    length(details) as details_size
FROM audit_entries
WHERE length(details) > 10000  -- Adjust threshold as needed
ORDER BY details_size DESC
LIMIT 100;