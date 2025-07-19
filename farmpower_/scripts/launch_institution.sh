#!/bin/bash
# LogLineOS Institutional Launch Script
# Current timestamp: 2025-07-19 05:19:26 UTC
# User: danvoulez

set -e  # Exit on any error

# Configuration
export GOD_KEY="user_special_key_never_share"
export LOG_DIR="logs"
export DATA_DIR="data"
export SPAN_DIR="$DATA_DIR/spans"

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $SPAN_DIR

echo "=========================================="
echo "🏛️ LOGLINEOS INSTITUTIONAL LAUNCH 🏛️"
echo "=========================================="
echo "Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "User: danvoulez"
echo "Hardware: Mac Mini M1"
echo "Executing from: $(pwd)"
echo "=========================================="

# Step 1: Generate Core Spans
echo "[1/7] Generating Foundational Spans..."
python -c "
from farm.span_collection import SpanCollection
collection = SpanCollection(god_key='$GOD_KEY')
collection.generate_comprehensive_set()
collection.export_collection('$SPAN_DIR/foundational_spans.json')
print(f'Generated {collection.span_count} foundational spans')
" >> $LOG_DIR/launch.log 2>&1

# Step 2: Initialize Core Components
echo "[2/7] Initializing Core Components..."
python -c "
from core.lingua_mater import LinguaMater
from core.vector_clock import VectorClock
from core.logline_vm import LogLineVM

# Initialize Lingua Mater
lingua = LinguaMater()
lingua.initialize_core_ontology()
print(f'Lingua Mater initialized with hash: {lingua.ontology_hash}')

# Initialize LogLine VM
vm = LogLineVM()
print(f'LogLine VM initialized with tension threshold: {vm.tension_threshold}')

# Save component states
import json
import os
os.makedirs('$DATA_DIR/core', exist_ok=True)
with open('$DATA_DIR/core/lingua_mater.json', 'w') as f:
    json.dump({
        'version': lingua.version,
        'ontology_hash': lingua.ontology_hash,
        'term_count': len(lingua.terms),
        'rule_count': len(lingua.rules)
    }, f, indent=2)
" >> $LOG_DIR/launch.log 2>&1

# Step 3: Prepare LLM and Diamond Farm
echo "[3/7] Preparing LLM Services..."
python -c "
import os
from llm.span_generator import SpanGenerator
from farm.advanced_diamond_farm import EnhancedDiamondFarm
from core.scarcity_engine import ScarcityEngine

# Create the generator
generator = SpanGenerator(model_path=os.getenv('LLM_MODEL_PATH'))

# Initialize farm components
scarcity_engine = ScarcityEngine(total_supply=21000000)
print(f'Scarcity Engine initialized with total supply: {scarcity_engine.total_supply}')
" >> $LOG_DIR/launch.log 2>&1

# Step 4: Compile Core Spans
echo "[4/7] Compiling Spans for Apple Silicon..."
python -c "
import json
from core.span_compiler import SpanCompiler

# Load spans
with open('$SPAN_DIR/foundational_spans.json', 'r') as f:
    spans = json.load(f)

# Create compiler
compiler = SpanCompiler(target_arch='apple_silicon', optimize_level=3)
print(f'Span Compiler initialized for Apple Silicon, optimization level 3')

# Compile key spans
compiled_count = 0
for span_id, span in spans.items():
    if span.get('kind') in ['genesis', 'constitution', 'setup_model']:
        compiled = compiler.compile(span)
        compiler.save_to_file(compiled, f'$SPAN_DIR/compiled/{span_id}.dspn')
        compiled_count += 1

print(f'Compiled {compiled_count} critical spans')
" >> $LOG_DIR/launch.log 2>&1

# Step 5: Launch API Services
echo "[5/7] Launching API Services..."
python api/deploy_spans_api.py >> $LOG_DIR/api.log 2>&1 &
API_PID=$!
echo "API service launched with PID: $API_PID"
echo $API_PID > $DATA_DIR/api.pid

# Step 6: Deploy Governance Structure
echo "[6/7] Deploying Governance Structure..."
curl -s -X POST "http://localhost:8080/spans/deploy" \
  -H "Authorization: Bearer $GOD_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"collection_path\": \"$SPAN_DIR/foundational_spans.json\", \"is_god_mode\": true}" \
  >> $LOG_DIR/governance.log

# Step 7: Start Mining and Monitoring
echo "[7/7] Starting Farm Services..."
curl -s -X POST "http://localhost:8080/services/start" \
  -H "Authorization: Bearer $GOD_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"service\": \"mining\", \"workers\": 4}" \
  >> $LOG_DIR/mining.log

# Launch complete
echo "=========================================="
echo "🌟 INSTITUTION LAUNCH COMPLETE 🌟"
echo "=========================================="
echo "API running on: http://localhost:8080"
echo "Logs available in: $LOG_DIR"
echo "Spans stored in: $SPAN_DIR"
echo ""
echo "Run the following command to check system status:"
echo "curl http://localhost:8080/market/stats"
echo ""
echo "To stop all services:"
echo "kill \$(cat $DATA_DIR/api.pid)"
echo "=========================================="

# Generate initial status report
curl -s http://localhost:8080/market/stats | python -m json.tool > $DATA_DIR/initial_status.json
cat $DATA_DIR/initial_status.json