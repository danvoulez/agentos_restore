#!/bin/bash
# Deploy script for Diamond Span Farm and LogLineOS

# Set the god key for special span creation
export GOD_KEY="user_special_key_never_share"

# Install dependencies
pip install fastapi uvicorn pydantic numpy networkx matplotlib requests

# Create directories if they don't exist
mkdir -p logs
mkdir -p data/spans
mkdir -p data/market

echo "Starting Diamond Span API..."
python -m api.diamond_api &
API_PID=$!
echo "API started with PID: $API_PID"

# Wait for API to start
sleep 2

# Create a genesis span
echo "Creating genesis span..."
curl -X POST http://localhost:8080/spans/mine \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GOD_KEY" \
  -d '{"prompt":"Genesis Diamond Span for LogLineOS. The beginning of cognitive sovereignty.","is_god_mode":true}'

echo "Diamond Span Farm deployment complete!"
echo "API running at: http://localhost:8080/"
echo "To stop: kill $API_PID"

# Keep the script running
wait $API_PID