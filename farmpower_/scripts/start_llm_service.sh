#!/bin/bash
# Script to start the LogLineOS LLM Service
# Current timestamp: 2025-07-19 05:24:06 UTC
# User: danvoulez

set -e  # Exit on any error

# Configuration
export LOG_DIR="logs"
export LLM_MODEL_ID=${LLM_MODEL_ID:-"gpt-3.5-turbo"}  # Default model
export LLM_PORT=${LLM_PORT:-8082}
export GOD_KEY=${GOD_KEY:-"user_special_key_never_share"}  # Use environment or default
export WASM_RUNTIME_PATH="./wasm_runtime"

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $WASM_RUNTIME_PATH
mkdir -p checkpoints

echo "=========================================="
echo "🧠 LOGLINEOS LLM SERVICE STARTUP 🧠"
echo "=========================================="
echo "Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "User: danvoulez"
echo "Model: $LLM_MODEL_ID"
echo "Port: $LLM_PORT"
echo "=========================================="

# Check if we have a local model or need to use API
if [[ $LLM_MODEL_ID == *"gpt"* ]]; then
    echo "Using OpenAI API. Checking API key..."
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Warning: OPENAI_API_KEY not set. Service may fail if using OpenAI models."
    else
        echo "API key detected (length: ${#OPENAI_API_KEY})"
    fi
else
    echo "Using local model: $LLM_MODEL_ID"
    # Check for model files
    MODEL_DIR="models/$LLM_MODEL_ID"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Model directory not found. Creating..."
        mkdir -p $MODEL_DIR
        
        # Note: In a real deployment, you would download model files here
        echo "Warning: You need to place model files in $MODEL_DIR"
    else
        echo "Model directory found: $MODEL_DIR"
    fi
fi

# Start the LLM API service
echo "Starting LLM API Service on port $LLM_PORT..."
uvicorn api.llm_api:app --host 0.0.0.0 