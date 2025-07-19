"""
Enzyme Runtime for LogLineOS
Executes enzymatic spans in isolated environments
Created: 2025-07-19 05:47:12 UTC
User: danvoulez
"""
import os
import json
import time
import logging
import asyncio
import uuid
import importlib.util
import hashlib
import subprocess
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/enzyme_runtime.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnzymeRuntime")

@dataclass
class EnzymeConfig:
    """Configuration for an enzyme"""
    name: str
    description: str
    version: str
    capabilities: List[str]
    module_path: Optional[str] = None
    function_name: Optional[str] = None
    wasm_path: Optional[str] = None
    python_code: Optional[str] = None
    max_runtime_seconds: int = 60
    memory_limit_mb: int = 512
    isolation_level: str = "process"  # process, thread, sandbox, wasm
    authenticated: bool = False
    requires_gpu: bool = False

@dataclass
class EnzymeContext:
    """Execution context for an enzyme"""
    span_id: str
    parameters: Dict[str, Any]
    parent_ids: List[str]
    actor: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    env_vars: Dict[str, str] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    execution_id: str = field(default_factory=lambda: f"exec-{uuid.uuid4()}")

class EnzymeRuntime:
    """
    Runtime for executing enzyme spans in isolated environments
    """
    
    def __init__(self, wasm_runtime_path: str = None):
        self.enzymes: Dict[str, EnzymeConfig] = {}
        self.results_cache: Dict[str, Any] = {}
        self.wasm_runtime_path = wasm_runtime_path or os.environ.get("WASM_RUNTIME_PATH", "./wasm_runtime")
        self.execution_history: List[Dict[str, Any]] = []
        self.current_executions: Dict[str, Dict[str, Any]] = {}
        
        # Create wasm runtime directory if it doesn't exist
        os.makedirs(self.wasm_runtime_path, exist_ok=True)
        
        # Register built-in enzymes
        self._register_builtin_enzymes()
        
        logger.info(f"Enzyme Runtime initialized with WASM path: {self.wasm_runtime_path}")
    
    def register_enzyme(self, config: EnzymeConfig) -> None:
        """Register a new enzyme"""
        self.enzymes[config.name] = config
        logger.info(f"Registered enzyme: {config.name} (v{config.version})")
    
    async def execute_span(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an enzyme span
        
        Args:
            span: Enzyme span to execute
            
        Returns:
            Execution result
        """
        # Validate span
        if "kind" not in span or span["kind"] != "enzima":
            raise ValueError("Not an enzima span")
        
        # Extract enzyme name
        enzyme_name = self._extract_enzyme_name(span)
        if not enzyme_name:
            raise ValueError("Missing enzyme name in span")
        
        # Check if enzyme exists
        if enzyme_name not in self.enzymes:
            raise ValueError(f"Unknown enzyme: {enzyme_name}")
        
        # Create execution context
        context = EnzymeContext(
            span_id=span.get("id", f"span-{uuid.uuid4()}"),
            parameters=self._extract_parameters(span),
            parent_ids=span.get("parent_ids", []),
            actor=span.get("actor", "system")
        )
        
        # Generate cache key
        cache_key = self._generate_cache_key(enzyme_name, context.parameters)
        
        # Check cache
        if cache_key in self.results_cache:
            logger.info(f"Cache hit for enzyme {enzyme_name}")
            return self.results_cache[cache_key]
        
        # Track execution
        self.current_executions[context.execution_id] = {
            "enzyme": enzyme_name,
            "span_id": context.span_id,
            "start_time": time.time(),
            "status": "running"
        }
        
        # Execute the enzyme
        try:
            result = await self._execute_enzyme(enzyme_name, context)
            
            # Update execution history
            execution_record = {
                "id": context.execution_id,
                "enzyme": enzyme_name,
                "span_id": context.span_id,
                "start_time": self.current_executions[context.execution_id]["start_time"],
                "end_time": time.time(),
                "status": "success",
                "duration": time.time() - self.current_executions[context.execution_id]["start_time"]
            }
            self.execution_history.append(execution_record)
            
            # Update current execution
            self.current_executions[context.execution_id]["status"] = "success"
            self.current_executions[context.execution_id]["end_time"] = time.time()
            
            # Cache the result
            self.results_cache[cache_key] = result
            
            # Cleanup
            del self.current_executions[context.execution_id]
            
            logger.info(f"Successfully executed enzyme {enzyme_name} in "
                       f"{execution_record['duration']:.3f}s")
            
            return result
            
        except Exception as e:
            # Update execution history
            execution_record = {
                "id": context.execution_id,
                "enzyme": enzyme_name,
                "span_id": context.span_id,
                "start_time": self.current_executions[context.execution_id]["start_time"],
                "end_time": time.time(),
                "status": "failed",
                "error": str(e),
                "duration": time.time() - self.current_executions[context.execution_id]["start_time"]
            }
            self.execution_history.append(execution_record)
            
            # Update current execution
            self.current_executions[context.execution_id]["status"] = "failed"
            self.current_executions[context.execution_id]["error"] = str(e)
            self.current_executions[context.execution_id]["end_time"] = time.time()
            
            # Cleanup
            del self.current_executions[context.execution_id]
            
            logger.error(f"Error executing enzyme {enzyme_name}: {str(e)}")
            raise
    
    async def list_enzymes(self) -> List[Dict[str, Any]]:
        """List all available enzymes"""
        return [
            {
                "name": config.name,
                "description": config.description,
                "version": config.version,
                "capabilities": config.capabilities
            }
            for config in self.enzymes.values()
        ]
    
    async def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history"""
        # Return most recent executions first
        return sorted(
            self.execution_history[-limit:],
            key=lambda x: x["start_time"],
            reverse=True
        )
    
    async def get_enzyme_stats(self) -> Dict[str, Any]:
        """Get enzyme statistics"""
        total_executions = len(self.execution_history)
        successful = sum(1 for exec in self.execution_history if exec["status"] == "success")
        failed = total_executions - successful
        
        if total_executions > 0:
            success_rate = successful / total_executions
            avg_duration = sum(exec["duration"] for exec in self.execution_history) / total_executions
        else:
            success_rate = 0
            avg_duration = 0
        
        return {
            "total_enzymes": len(self.enzymes),
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "active_executions": len(self.current_executions),
            "cache_entries": len(self.results_cache),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_cache(self) -> int:
        """Clear the results cache"""
        cache_size = len(self.results_cache)
        self.results_cache.clear()
        return cache_size
    
    def _extract_enzyme_name(self, span: Dict[str, Any]) -> Optional[str]:
        """Extract enzyme name from a span"""
        # Check different possible locations
        if "payload" in span and isinstance(span["payload"], dict):
            if "enzyme" in span["payload"]:
                return span["payload"]["enzyme"]
            if "enzima" in span["payload"]:
                return span["payload"]["enzima"]
        
        if "metadata" in span and isinstance(span["metadata"], dict):
            if "enzyme" in span["metadata"]:
                return span["metadata"]["enzyme"]
            if "enzima" in span["metadata"]:
                return span["metadata"]["enzima"]
        
        return None
    
    def _extract_parameters(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from a span"""
        if "payload" in span and isinstance(span["payload"], dict):
            if "params" in span["payload"]:
                return span["payload"]["params"]
            if "parameters" in span["payload"]:
                return span["payload"]["parameters"]
        
        # Default empty parameters
        return {}
    
    def _generate_cache_key(self, enzyme_name: str, parameters: Dict[str, Any]) -> str:
        """Generate a cache key for an execution"""
        params_str = json.dumps(parameters, sort_keys=True)
        hash_input = f"{enzyme_name}:{params_str}".encode()
        return hashlib.sha256(hash_input).hexdigest()
    
    async def _execute_enzyme(self, enzyme_name: str, context: EnzymeContext) -> Dict[str, Any]:
        """Execute an enzyme with the given context"""
        config = self.enzymes[enzyme_name]
        
        # Select execution method based on configuration
        if config.isolation_level == "wasm" and config.wasm_path:
            return await self._execute_wasm_enzyme(config, context)
        elif config.isolation_level == "process" and (config.module_path or config.python_code):
            return await self._execute_process_enzyme(config, context)
        elif config.isolation_level == "thread" and (config.module_path or config.python_code):
            return await self._execute_thread_enzyme(config, context)
        elif config.isolation_level == "sandbox":
            return await self._execute_sandbox_enzyme(config, context)
        else:
            # Default to direct execution
            return await self._execute_direct_enzyme(config, context)
    
    async def _execute_wasm_enzyme(self, config: EnzymeConfig, context: EnzymeContext) -> Dict[str, Any]:
        """Execute a WASM-based enzyme"""
        # Check if WASM file exists
        if not config.wasm_path or not os.path.exists(config.wasm_path):
            raise ValueError(f"WASM file not found: {config.wasm_path}")
        
        # Prepare input file
        input_file = os.path.join(
            self.wasm_runtime_path, 
            f"input_{context.execution_id}.json"
        )
        
        with open(input_file, "w") as f:
            json.dump({
                "context": {
                    "span_id": context.span_id,
                    "parameters": context.parameters,
                    "parent_ids": context.parent_ids,
                    "actor": context.actor,
                    "timestamp": context.timestamp,
                    "execution_id": context.execution_id
                }
            }, f)
        
        # Prepare output file
        output_file = os.path.join(
            self.wasm_runtime_path, 
            f"output_{context.execution_id}.json"
        )
        
        try:
            # Execute WASM module
            # Note: This assumes a WASM runtime like wasmtime is available
            cmd = [
                "wasmtime", 
                "--dir", ".", 
                config.wasm_path, 
                input_file, 
                output_file
            ]
            
            # Set timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.max_runtime_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError(f"Enzyme execution timed out after {config.max_runtime_seconds}s")
            
            # Check for errors
            if process.returncode != 0:
                stderr_str = stderr.decode('utf-8') if stderr else "Unknown error"
                raise RuntimeError(f"WASM execution failed: {stderr_str}")
            
            # Read output file
            with open(output_file, "r") as f:
                result = json.load(f)
            
            return result
            
        finally:
            # Clean up temporary files
            if os.path.exists(input_file):
                os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    async def _execute_process_enzyme(self, config: EnzymeConfig, context: EnzymeContext) -> Dict[str, Any]:
        """Execute an enzyme in a separate process"""
        # Create temporary script file
        script_file = os.path.join(
            self.wasm_runtime_path, 
            f"enzyme_{context.execution_id}.py"
        )
        
        try:
            # Determine code to execute
            if config.python_code:
                code = config.python_code
            elif config.module_path and config.function_name:
                code = f"""
import sys
import json
import importlib.util

# Load module
spec = importlib.util.spec_from_file_location("enzyme_module", "{config.module_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Load context
with open(sys.argv[1], "r") as f:
    context = json.load(f)

# Execute function
result = module.{config.function_name}(context)

# Save result
with open(sys.argv[2], "w") as f:
    json.dump(result, f)
"""
            else:
                raise ValueError("Either python_code or module_path+function_name must be provided")
            
            # Write script file
            with open(script_file, "w") as f:
                f.write(code)
            
            # Prepare input/output files
            input_file = os.path.join(
                self.wasm_runtime_path, 
                f"input_{context.execution_id}.json"
            )
            
            output_file = os.path.join(
                self.wasm_runtime_path, 
                f"output_{context.execution_id}.json"
            )
            
            # Write input file
            with open(input_file, "w") as f:
                json.dump({
                    "span_id": context.span_id,
                    "parameters": context.parameters,
                    "parent_ids": context.parent_ids,
                    "actor": context.actor,
                    "timestamp": context.timestamp,
                    "execution_id": context.execution_id
                }, f)
            
            # Execute in subprocess
            cmd = [
                sys.executable,
                script_file,
                input_file,
                output_file
            ]
            
            # Set environment variables
            env = os.environ.copy()
            env.update(context.env_vars)
            
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.max_runtime_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError(f"Enzyme execution timed out after {config.max_runtime_seconds}s")
            
            # Check for errors
            if process.returncode != 0:
                stderr_str = stderr.decode('utf-8') if stderr else "Unknown error"
                raise RuntimeError(f"Process execution failed: {stderr_str}")
            
            # Read output file
            try:
                with open(output_file, "r") as f:
                    result = json.load(f)
                return result
            except FileNotFoundError:
                raise RuntimeError("Output file was not created")
            except json.JSONDecodeError:
                raise RuntimeError("Output file contains invalid JSON")
            
        finally:
            # Clean up temporary files
            for file in [script_file, input_file, output_file]:
                if os.path.exists(file):
                    os.unlink(file)
    
    async def _execute_thread_enzyme(self, config: EnzymeConfig, context: EnzymeContext) -> Dict[str, Any]:
        """Execute an enzyme in a separate thread"""
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def thread_function():
            try:
                # Import module if needed
                if config.module_path and config.function_name:
                    spec = importlib.util.spec_from_file_location("enzyme_module", config.module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    function = getattr(module, config.function_name)
                elif config.python_code:
                    # Create function from code
                    namespace = {}
                    exec(config.python_code, namespace)
                    function = namespace.get("execute_enzyme")
                else:
                    raise ValueError("Either python_code or module_path+function_name must be provided")
                
                # Convert context to dict
                context_dict = {
                    "span_id": context.span_id,
                    "parameters": context.parameters,
                    "parent_ids": context.parent_ids,
                    "actor": context.actor,
                    "timestamp": context.timestamp,
                    "execution_id": context.execution_id
                }
                
                # Execute function
                result = function(context_dict)
                result_queue.put(result)
            except Exception as e:
                error_queue.put(e)
        
        # Start thread
        thread = threading.Thread(target=thread_function)
        thread.daemon = True
        thread.start()
        
        # Wait for result with timeout
        thread.join(config.max_runtime_seconds)
        
        if thread.is_alive():
            # Thread is still running after timeout
            raise TimeoutError(f"Enzyme execution timed out after {config.max_runtime_seconds}s")
        
        # Check for errors
        if not error_queue.empty():
            error = error_queue.get()
            raise RuntimeError(f"Thread execution failed: {str(error)}")
        
        # Get result
        if result_queue.empty():
            raise RuntimeError("No result produced")
        
        return result_queue.get()
    
    async def _execute_sandbox_enzyme(self, config: EnzymeConfig, context: EnzymeContext) -> Dict[str, Any]:
        """Execute an enzyme in a sandbox"""
        # This is a placeholder for a more sophisticated sandbox implementation
        logger.warning("Sandbox execution not fully implemented, falling back to process isolation")
        return await self._execute_process_enzyme(config, context)
    
    async def _execute_direct_enzyme(self, config: EnzymeConfig, context: EnzymeContext) -> Dict[str, Any]:
        """Execute an enzyme directly (no isolation)"""
        # Try to find the function
        function = None
        
        # Check if we have a registered handler
        if hasattr(self, f"_enzyme_{config.name}"):
            function = getattr(self, f"_enzyme_{config.name}")
        elif config.module_path and config.function_name:
            # Import module
            spec = importlib.util.spec_from_file_location("enzyme_module", config.module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            function = getattr(module, config.function_name)
        elif config.python_code:
            # Create function from code
            namespace = {}
            exec(config.python_code, namespace)
            function = namespace.get("execute_enzyme")
        
        if not function:
            raise ValueError("No executable function found for enzyme")
        
        # Convert context to dict
        context_dict = {
            "span_id": context.span_id,
            "parameters": context.parameters,
            "parent_ids": context.parent_ids,
            "actor": context.actor,
            "timestamp": context.timestamp,
            "execution_id": context.execution_id
        }
        
        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                self._run_async_or_sync_function(function, context_dict),
                timeout=config.max_runtime_seconds
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Enzyme execution timed out after {config.max_runtime_seconds}s")
    
    async def _run_async_or_sync_function(self, function, arg):
        """Run a function that may be sync or async"""
        if asyncio.iscoroutinefunction(function):
            return await function(arg)
        else:
            return function(arg)
    
    def _register_builtin_enzymes(self):
        """Register built-in enzymes"""
        # Text generation enzyme
        self.register_enzyme(EnzymeConfig(
            name="text_generation",
            description="Generate text using LLM",
            version="1.0.0",
            capabilities=["text_generation", "summarization"],
            isolation_level="thread"
        ))
        
        # Diamond mining enzyme
        self.register_enzyme(EnzymeConfig(
            name="diamond_mining",
            description="Mine diamond spans",
            version="1.0.0",
            capabilities=["span_mining", "value_calculation"],
            isolation_level="thread"
        ))
        
        # File manipulation enzyme
        self.register_enzyme(EnzymeConfig(
            name="file_operations",
            description="Perform file operations",
            version="1.0.0",
            capabilities=["read_file", "write_file", "list_directory"],
            isolation_level="sandbox"
        ))
        
        # Image processing enzyme
        self.register_enzyme(EnzymeConfig(
            name="image_processing",
            description="Process images",
            version="1.0.0",
            capabilities=["resize", "filter", "convert"],
            isolation_level="process",
            requires_gpu=True
        ))
    
    # Built-in enzyme handlers
    
    async def _enzyme_text_generation(self, context):
        """Built-in handler for text generation"""
        prompt = context["parameters"].get("prompt", "")
        max_tokens = context["parameters"].get("max_tokens", 100)
        
        # This is a mock implementation
        return {
            "text": f"Generated text for prompt: {prompt[:20]}...",
            "tokens": max_tokens,
            "execution_id": context["execution_id"]
        }
    
    async def _enzyme_diamond_mining(self, context):
        """Built-in handler for diamond mining"""
        content = context["parameters"].get("content", "")
        
        # This is a mock implementation
        return {
            "span_id": f"span-{uuid.uuid4()}",
            "energy": 10.0 + (len(content) / 100),
            "execution_id": context["execution_id"]
        }