#!/bin/bash
# Deploy script for comprehensive span collection
# Created: 2025-07-19 05:09:54 UTC
# Author: danvoulez

set -e

# Configuration
export GOD_KEY="user_special_key_never_share"
export API_URL="http://localhost:8080"

# Create necessary directories
mkdir -p data/spans
mkdir -p logs

echo "===== DEPLOYING COMPREHENSIVE SPAN COLLECTION ====="
echo "Current timestamp: $(date -u +'%Y-%m-%d %H:%M:%S')"
echo "User: danvoulez"

# Generate the comprehensive span collection
echo "Generating comprehensive span collection..."
python -c "
from farm.span_collection import SpanCollection
collector = SpanCollection(god_key='$GOD_KEY')
result = collector.generate_comprehensive_set()
collector.export_collection('data/comprehensive_spans.json')
print(f'Generated {result[\"total_spans\"]} spans across {len(result[\"by_kind\"])} categories')
"

# Register all spans with the API
echo "Registering spans with the API..."
python -c "
import json
import requests
import os
import time

with open('data/comprehensive_spans.json', 'r') as f:
    spans = json.load(f)

api_url = os.environ['API_URL']
god_key = os.environ['GOD_KEY']

print(f'Found {len(spans)} spans to deploy')

# Register spans in order of dependencies
registered = []
attempts = 0
max_attempts = 10

while len(registered) < len(spans) and attempts < max_attempts:
    for span_id, span in spans.items():
        if span_id in registered:
            continue
            
        # Check if all parents are registered
        parents_ready = True
        for parent_id in span.get('parent_ids', []):
            if parent_id != 'genesis' and parent_id not in registered:
                parents_ready = False
                break
                
        if not parents_ready:
            continue
            
        # Register span
        headers = {'Authorization': f'Bearer {god_key}'} if span.get('metadata', {}).get('creator') == god_key else {}
        
        try:
            resp = requests.post(f'{api_url}/spans/register', json=span, headers=headers)
            if resp.status_code == 200:
                print(f'Registered span: {span_id} ({span.get(\"kind\", \"unknown\")})')
                registered.append(span_id)
            else:
                print(f'Failed to register {span_id}: {resp.text}')
        except Exception as e:
            print(f'Error registering {span_id}: {str(e)}')
            
    attempts += 1
    if len(registered) < len(spans):
        print(f'Registered {len(registered)}/{len(spans)} spans, retrying...')
        time.sleep(1)

print(f'Successfully registered {len(registered)}/{len(spans)} spans')
"

# Start the mining service
echo "Starting mining service..."
curl -X POST "${API_URL}/services/start" \
  -H "Authorization: Bearer ${GOD_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"service": "mining", "workers": 4}'

echo "===== DEPLOYMENT COMPLETE ====="
echo "Run 'curl ${API_URL}/market/stats' to see system status"