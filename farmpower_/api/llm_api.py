"""
LLM API for LogLineOS/DiamondSpan
Provides HTTP access to the LLM services
"""
from fastapi import FastAPI, HTTPException, Depends, Request, Header, Response, WebSocket, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Generator
import uvicorn
import os
import json
import asyncio
import logging
from datetime import datetime

from llm.llm_service import LLMService, LLMConfig
from llm.span_processor import SpanProcessor
from llm.enzima import EnzimaEngine
from llm.checkpoint_manager import CheckpointManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM_API")

# Initialize components
app = FastAPI(title="LogLineOS LLM API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM service
llm_config = LLMConfig(
    model_id=os.getenv("LLM_MODEL_ID", "gpt-3.5-turbo"),
    temperature=0.7,
    max_tokens=1024
)
llm_service = LLMService(llm_config)
span_processor = SpanProcessor(llm_service)
enzima_engine = EnzimaEngine(llm_service)
checkpoint_manager = CheckpointManager()

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    await llm_service.initialize()
    await span_processor.initialize()
    await enzima_engine.initialize()

# Request models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ProcessSpanRequest(BaseModel):
    span: Dict[str, Any]
    operations: Optional[List[str]] = ["process"]

class GenerateSpanRequest(BaseModel):
    prompt: str
    kind: Optional[str] = "diamond"
    parent_ids: Optional[List[str]] = None

class ExecuteEnzimaRequest(BaseModel):
    span: Dict[str, Any]

class EmbeddingRequest(BaseModel):
    text: str

# Authentication middleware
async def verify_api_key(api_key: str = Header(None, alias="X-API-Key")):
    # In production, you would validate against a database of API keys
    # For now, we'll accept any key for demonstration
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is required")
    return api_key

# Routes
@app.get("/")
async def root():
    return {
        "name": "LogLineOS LLM API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate")
async def generate(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    """Generate text from a prompt"""
    if request.stream:
        # Stream the response
        async def generate_stream():
            async for text in llm_service.generate_text(
                request.prompt, 
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True
            ):
                yield f"data: {json.dumps({'text': text})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    else:
        # Regular response
        try:
            text, stats = await llm_service.generate_text(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            return {
                "text": text,
                "stats": {
                    "tokens": stats.tokens_generated,
                    "time_ms": stats.generation_time_ms,
                    "tokens_per_second": stats.tokens_per_second
                }
            }
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-span")
async def process_span(request: ProcessSpanRequest, api_key: str = Depends(verify_api_key)):
    """Process a Diamond Span"""
    try:
        operations = request.operations
        span = request.span
        
        # Process based on requested operations
        results = {"span": span}
        
        for operation in operations:
            if operation == "process":
                span = await span_processor.process_span(span)
            elif operation == "validate":
                results["validation"] = await span_processor.validate_span(span)
            elif operation == "enhance":
                span = await span_processor.enhance_span(span)
            else:
                results[f"unknown_{operation}"] = {"error": f"Unknown operation: {operation}"}
        
        results["span"] = span
        return results
    
    except Exception as e:
        logger.error(f"Error processing span: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-span")
async def generate_span(request: GenerateSpanRequest, api_key: str = Depends(verify_api_key)):
    """Generate a Diamond Span from a prompt"""
    try:
        span = await span_processor.generate_from_prompt(
            request.prompt,
            kind=request.kind,
            parent_ids=request.parent_ids
        )
        
        return {"span": span}
    except Exception as e:
        logger.error(f"Error generating span: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute-enzima")
async def execute_enzima(request: ExecuteEnzimaRequest, api_key: str = Depends(verify_api_key)):
    """Execute an Enzima span"""
    try:
        result = await enzima_engine.execute_span(request.span)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error executing enzima: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/enzimas")
async def list_enzimas(api_key: str = Depends(verify_api_key)):
    """List available Enzimas"""
    enzimas = await enzima_engine.list_enzimas()
    return {"enzimas": enzimas}

@app.post("/embeddings")
async def get_embedding(request: EmbeddingRequest, api_key: str = Depends(verify_api_key)):
    """Get embedding for text"""
    try:
        embedding = await llm_service.get_embedding(request.text)
        return {"embedding": embedding}
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/checkpoints")
async def list_checkpoints(
    name: str = None,
    api_key: str = Depends(verify_api_key)
):
    """List available checkpoints"""
    checkpoints = await checkpoint_manager.list_checkpoints(filter_name=name)
    return {"checkpoints": checkpoints}

@app.get("/status")
async def get_status():
    """Get LLM service status"""
    stats = await llm_service.get_stats()
    return {
        "status": "operational",
        "model": stats["model_id"],
        "requests": stats["total_requests"],
        "tokens_generated": stats["total_tokens_generated"],
        "average_generation_time_ms": stats["average_generation_time_ms"],
        "timestamp": datetime.now().isoformat()
    }

# WebSocket for streaming
@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            
            # Parse request
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 1024)
            temperature = data.get("temperature", 0.7)
            
            # Generate
            text, stats = await llm_service.generate_text(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Send response
            await websocket.send_json({
                "text": text,
                "stats": {
                    "tokens": stats.tokens_generated,
                    "time_ms": stats.generation_time_ms,
                    "tokens_per_second": stats.tokens_per_second
                }
            })
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("llm_api:app", host="0.0.0.0", port=8082, reload=True)