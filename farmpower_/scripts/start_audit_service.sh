#!/bin/bash
# Start the LogLineOS Audit Service
# Created: 2025-07-19 05:28:32 UTC
# User: danvoulez

set -e  # Exit on any error

# Configuration
export LOG_DIR="logs"
export DATA_DIR="data/audit"
export AUDIT_PORT=${AUDIT_PORT:-8083}
export GOD_KEY=${GOD_KEY:-"user_special_key_never_share"}

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $DATA_DIR
mkdir -p $DATA_DIR/snapshots

echo "=========================================="
echo "🔍 LOGLINEOS AUDIT SERVICE STARTUP 🔍"
echo "=========================================="
echo "Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "User: danvoulez"
echo "Port: $AUDIT_PORT"
echo "=========================================="

# Start the Audit API service
echo "Starting Audit API Service on port $AUDIT_PORT..."
uvicorn api.audit_api:app --host 0.0.0.0 --port $AUDIT_PORT --log-level info &

# Save PID for later shutdown
echo $! > $DATA_DIR/audit_api.pid

echo "Audit service started with PID: $(cat $DATA_DIR/audit_api.pid)"
echo "To stop the service: kill $(cat $DATA_DIR/audit_api.pid)"
echo "=========================================="