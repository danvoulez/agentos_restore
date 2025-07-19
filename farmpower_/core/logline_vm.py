"""
LogLine Virtual Machine for executing Diamond Spans
"""
import hashlib
import time
from enum import Enum
from typing import Dict, List, Any, Optional

class SpanStatus(Enum):
    PENDING = "pending"
    SIMULATED = "simulated"
    EXECUTED = "executed"
    COMPENSATED = "compensated"
    FAILED = "failed"

class LogLineVM:
    def __init__(self):
        self.span_registry: Dict[str, Dict] = {}
        self.execution_history: List[str] = []
        self.tension_threshold = 17.3
        
    def register_span(self, span: Dict[str, Any]) -> str:
        """Register a span in the VM"""
        span_id = self._generate_span_id(span)
        span['id'] = span_id
        span['status'] = SpanStatus.PENDING.value
        span['created_at'] = time.time()
        self.span_registry[span_id] = span
        return span_id
        
    def simulate(self, span_id: str) -> Dict[str, Any]:
        """Simulate a span execution without side effects"""
        if span_id not in self.span_registry:
            raise ValueError(f"Span {span_id} not found")
            
        span = self.span_registry[span_id]
        tension = self._calculate_tension(span)
        
        if tension > self.tension_threshold:
            return {
                "status": "rejected",
                "reason": f"Tension {tension} exceeds threshold {self.tension_threshold}",
                "tension": tension
            }
            
        # Predict resource usage
        resources = self._predict_resources(span)
        
        span['status'] = SpanStatus.SIMULATED.value
        span['simulated_at'] = time.time()
        span['predicted_tension'] = tension
        span['predicted_resources'] = resources
        
        return {
            "status": "approved",
            "tension": tension,
            "resources": resources,
            "simulation_id": hashlib.sha256(f"{span_id}:{time.time()}".encode()).hexdigest()[:16]
        }
        
    def execute(self, span_id: str) -> Dict[str, Any]:
        """Execute a span that has been simulated"""
        if span_id not in self.span_registry:
            raise ValueError(f"Span {span_id} not found")
            
        span = self.span_registry[span_id]
        if span['status'] != SpanStatus.SIMULATED.value:
            raise ValueError(f"Span {span_id} must be simulated before execution")
            
        # Check ethical constraints
        if not self._verify_ethical_constraints(span):
            span['status'] = SpanStatus.FAILED.value
            return {
                "status": "rejected",
                "reason": "Ethical constraints violated"
            }
            
        # Execute span based on type
        result = self._dispatch_execution(span)
        
        span['status'] = SpanStatus.EXECUTED.value
        span['executed_at'] = time.time()
        span['execution_result'] = result
        self.execution_history.append(span_id)
        
        return {
            "status": "executed",
            "result": result,
            "span_id": span_id
        }
        
    def compensate(self, span_id: str) -> Dict[str, Any]:
        """Compensate (undo) an executed span"""
        if span_id not in self.span_registry:
            raise ValueError(f"Span {span_id} not found")
            
        span = self.span_registry[span_id]
        if span['status'] != SpanStatus.EXECUTED.value:
            raise ValueError(f"Span {span_id} must be executed before compensation")
            
        # Generate compensation span
        comp_span = self._generate_compensation_span(span)
        comp_id = self.register_span(comp_span)
        
        # Execute compensation immediately (no simulation for emergency rollback)
        result = self._dispatch_execution(comp_span)
        
        span['status'] = SpanStatus.COMPENSATED.value
        span['compensated_at'] = time.time()
        span['compensation_span_id'] = comp_id
        
        # Update compensation span
        comp_span['status'] = SpanStatus.EXECUTED.value
        comp_span['executed_at'] = time.time()
        comp_span['execution_result'] = result
        self.span_registry[comp_id] = comp_span
        
        return {
            "status": "compensated",
            "compensation_span_id": comp_id,
            "result": result
        }
        
    def _generate_span_id(self, span: Dict[str, Any]) -> str:
        """Generate a unique ID for a span"""
        content = f"{span.get('kind', '')}:{span.get('who', '')}:{span.get('what', '')}:{time.time()}"
        return f"span-{hashlib.sha256(content.encode()).hexdigest()[:16]}"
        
    def _calculate_tension(self, span: Dict[str, Any]) -> float:
        """Calculate the tension (potential risk) of a span"""
        base_tension = 1.0
        
        # Higher tension for certain operations
        if span.get('kind') in ['emergency_action', 'governance_policy']:
            base_tension *= 2.0
            
        # Tension modified by payload complexity
        payload = span.get('payload', {})
        payload_size = len(str(payload))
        complexity_factor = min(3.0, payload_size / 1000)
        
        # Check for dependencies - more dependencies = higher tension
        parents = span.get('parent_ids', [])
        dependency_factor = 1.0 + (len(parents) * 0.1)
        
        # Calculate final tension
        tension = base_tension * complexity_factor * dependency_factor
        
        return tension
        
    def _predict_resources(self, span: Dict[str, Any]) -> Dict[str, float]:
        """Predict resource usage for a span"""
        kind = span.get('kind', '')
        payload_size = len(str(span.get('payload', {})))
        
        # Base resource estimates
        resources = {
            "memory_mb": 10 + (payload_size / 1000),
            "cpu_seconds": 0.1 + (payload_size / 10000),
            "energy_joules": 5 + (payload_size / 2000)
        }
        
        # Adjust based on span kind
        if kind == 'train':
            resources["memory_mb"] *= 10
            resources["cpu_seconds"] *= 20
            resources["energy_joules"] *= 15
        elif kind == 'simulate':
            resources["memory_mb"] *= 5
            resources["cpu_seconds"] *= 8
            resources["energy_joules"] *= 5
            
        return resources
        
    def _verify_ethical_constraints(self, span: Dict[str, Any]) -> bool:
        """Verify that a span meets ethical constraints"""
        constraints = [
            "cannot_harm_humans",
            "cannot_override_free_will",
            "must_preserve_cognitive_diversity"
        ]
        
        # Check if span has explicit ethical constraints
        span_constraints = span.get('ethical_constraints', [])
        
        # If no constraints specified, use defaults
        if not span_constraints:
            span_constraints = constraints
            
        # TODO: Implement actual constraint verification logic
        # For now, just check that nothing explicitly violates constraints
        payload = span.get('payload', {})
        what = span.get('what', '')
        
        if 'harm' in what.lower() or 'weapon' in str(payload).lower():
            return False
            
        return True
        
    def _dispatch_execution(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch execution based on span kind"""
        kind = span.get('kind', '')
        
        # Map kinds to execution functions
        executors = {
            'train': self._execute_train,
            'api': self._execute_api,
            'predict': self._execute_predict,
            'fine_tune': self._execute_fine_tune,
            'tokenizer_config': self._execute_tokenizer,
            'batch': self._execute_batch,
            # Add more executors as needed
        }
        
        # Default executor
        executor = executors.get(kind, self._execute_default)
        return executor(span)
        
    def _generate_compensation_span(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a compensation span to undo an executed span"""
        return {
            'kind': 'compensate',
            'who': 'system',
            'what': f"compensate_{span.get('what', '')}",
            'why': f"rollback_{span.get('id', '')}",
            'parent_ids': [span['id']],
            'payload': {
                'target_span': span['id'],
                'reason': 'Explicit compensation request'
            }
        }
        
    # Execution implementations for different span kinds
    def _execute_train(self, span: Dict[str, Any]) -> Dict[str, Any]:
        # Simulated training execution
        return {"status": "success", "message": "Training executed"}
        
    def _execute_api(self, span: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "message": "API call executed"}
        
    def _execute_predict(self, span: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "message": "Prediction executed"}
        
    def _execute_fine_tune(self, span: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "message": "Fine-tuning executed"}
        
    def _execute_tokenizer(self, span: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "message": "Tokenizer configured"}
        
    def _execute_batch(self, span: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "message": "Batch processing executed"}
        
    def _execute_default(self, span: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "message": f"Generic execution for {span.get('kind', 'unknown')}"}