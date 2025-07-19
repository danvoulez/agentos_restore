#!/bin/bash
# Start the LogLineOS Simulation Service
# Created: 2025-07-19 05:37:29 UTC
# User: danvoulez

set -e  # Exit on any error

# Configuration
export LOG_DIR="logs"
export DATA_DIR="data/simulation"
export SIMULATION_PORT=${SIMULATION_PORT:-8084}
export SIMULATION_API_KEY=${SIMULATION_API_KEY:-"default_key"}

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $DATA_DIR

echo "=========================================="
echo "🧪 LOGLINEOS SIMULATION SERVICE STARTUP 🧪"
echo "=========================================="
echo "Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "User: danvoulez"
echo "Port: $SIMULATION_PORT"
echo "=========================================="

# Start the Simulation API service
echo "Starting Simulation API Service on port $SIMULATION_PORT..."
uvicorn api.simulation_api:app --host 0.0.0.0 --port $SIMULATION_PORT --log-level info &

# Save PID for later shutdown
echo $! > $DATA_DIR/simulation_api.pid

echo "Simulation service started with PID: $(cat $DATA_DIR/simulation_api.pid)"
echo "To stop the service: kill $(cat $DATA_DIR/simulation_api.pid)"
echo "=========================================="