#!/bin/bash
# Start the LogLineOS Diamond Miner Service
# Created: 2025-07-19 05:42:25 UTC
# User: danvoulez

set -e  # Exit on any error

# Configuration
export LOG_DIR="logs"
export DATA_DIR="data/miner"
export MINER_PORT=${MINER_PORT:-8085}
export MINER_THREADS=${MINER_THREADS:-4}
export MINER_DIFFICULTY=${MINER_DIFFICULTY:-1.0}
export ENERGY_FACTOR=${ENERGY_FACTOR:-1.0}
export REWARD_SCHEMA=${REWARD_SCHEMA:-"logarithmic"}
export GOD_KEY=${GOD_KEY:-"user_special_key_never_share"}

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $DATA_DIR

echo "=========================================="
echo "⛏️ LOGLINEOS DIAMOND MINER STARTUP ⛏️"
echo "=========================================="
echo "Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "User: danvoulez"
echo "Port: $MINER_PORT"
echo "Threads: $MINER_THREADS"
echo "=========================================="

# Start the Miner API service
echo "Starting Diamond Miner API Service on port $MINER_PORT..."
uvicorn api.miner_api:app --host 0.0.0.0 --port $MINER_PORT --log-level info &

# Save PID for later shutdown
echo $! > $DATA_DIR/miner_api.pid

echo "Diamond Miner service started with PID: $(cat $DATA_DIR/miner_api.pid)"
echo "To stop the service: kill $(cat $DATA_DIR/miner_api.pid)"
echo "=========================================="

# Start mining automatically
echo "Starting mining operations..."
curl -s -X POST "http://localhost:$MINER_PORT/start" \
  -H "X-API-Key: default_key" \
  -H "Content-Type: application/json" \
  -d "{}"

echo "Mining operations started!"
echo "Use 'curl http://localhost:$MINER_PORT/stats' to check status"
echo "=========================================="