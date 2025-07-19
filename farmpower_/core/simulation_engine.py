"""
Simulation Engine for LogLineOS/DiamondSpan
Provides advanced simulation capabilities for spans
Created: 2025-07-19 05:37:29 UTC
User: danvoulez
"""
import os
import json
import time
import logging
import asyncio
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import uuid

# Import LogLineOS components if available
try:
    from core.diamond_span import DiamondSpan
    from core.logline_vm import LogLineVM, SpanStatus
    from core.vector_clock import VectorClock
    from core.span_algebra import SpanAlgebra
    from core.grammar_vector import GrammarVector
    HAS_CORE_IMPORTS = True
except ImportError:
    HAS_CORE_IMPORTS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SimulationEngine")

class SimulationConfig:
    """Configuration for simulation engine"""
    
    def __init__(self, 
                max_depth: int = 5, 
                max_branches: int = 10,
                max_duration_seconds: int = 60,
                resolution_ms: int = 100,
                tension_threshold: float = 17.3,
                randomize_outcomes: bool = True,
                seed: int = None):
        """Initialize simulation configuration"""
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.max_duration_seconds = max_duration_seconds
        self.resolution_ms = resolution_ms
        self.tension_threshold = tension_threshold
        self.randomize_outcomes = randomize_outcomes
        self.seed = seed if seed is not None else int(time.time())
        
        # Initialize random number generator
        self.rng = np.random.RandomState(self.seed)

class SimulationResult:
    """Results of a simulation run"""
    
    def __init__(self, 
                simulation_id: str,
                root_span_id: str,
                start_time: datetime,
                end_time: datetime = None,
                success: bool = False,
                outcome_spans: List[Dict[str, Any]] = None,
                metrics: Dict[str, Any] = None,
                errors: List[str] = None):
        """Initialize simulation result"""
        self.simulation_id = simulation_id
        self.root_span_id = root_span_id
        self.start_time = start_time
        self.end_time = end_time or datetime.now()
        self.success = success
        self.outcome_spans = outcome_spans or []
        self.metrics = metrics or {}
        self.errors = errors or []
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "simulation_id": self.simulation_id,
            "root_span_id": self.root_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "outcome_spans": self.outcome_spans,
            "metrics": self.metrics,
            "errors": self.errors,
            "outcome_count": len(self.outcome_spans),
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> str:
        """Generate a human-readable summary"""
        if self.success:
            return f"Simulation completed successfully with {len(self.outcome_spans)} outcomes in {self.duration_seconds:.2f}s"
        else:
            error_count = len(self.errors)
            return f"Simulation failed with {error_count} errors in {self.duration_seconds:.2f}s"

class SimulationEngine:
    """
    Advanced simulation engine for Diamond Spans
    """
    
    def __init__(self, config: SimulationConfig = None, vm: LogLineVM = None):
        """Initialize simulation engine"""
        self.config = config or SimulationConfig()
        
        # Use provided VM or create a new one if available
        self.vm = vm
        if not self.vm and HAS_CORE_IMPORTS:
            self.vm = LogLineVM()
        
        # Store simulation histories
        self.simulations: Dict[str, SimulationResult] = {}
        
        # Register built-in predictors
        self.predictors: Dict[str, Callable] = {}
        self._register_default_predictors()
        
        logger.info(f"Simulation engine initialized (max_depth={self.config.max_depth}, "
                   f"max_branches={self.config.max_branches}, tension_threshold={self.config.tension_threshold})")
    
    async def simulate_span(self, 
                         span: Dict[str, Any], 
                         config_override: Dict[str, Any] = None) -> SimulationResult:
        """
        Simulate execution of a span with all possible outcomes
        
        Args:
            span: The span to simulate
            config_override: Optional configuration overrides
        
        Returns:
            SimulationResult with all possible outcomes and metrics
        """
        # Create new configuration if overrides provided
        if config_override:
            config = SimulationConfig(**{**vars(self.config), **config_override})
        else:
            config = self.config
        
        # Create simulation ID and results object
        simulation_id = f"sim-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        span_id = span.get("id", f"span-{uuid.uuid4()}")
        result = SimulationResult(
            simulation_id=simulation_id,
            root_span_id=span_id,
            start_time=start_time
        )
        
        # Create metrics dictionary
        metrics = {
            "tension_initial": 0.0,
            "tension_max": 0.0,
            "tension_final": 0.0,
            "branch_count": 0,
            "depth_reached": 0,
            "energia_consumed": 0.0,
            "critical_paths": 0,
            "safe_paths": 0
        }
        
        try:
            # If we have a VM, use it to calculate initial tension
            if self.vm:
                initial_tension = self._calculate_tension(span)
                metrics["tension_initial"] = initial_tension
                metrics["tension_max"] = initial_tension
                
                # Check if initial tension exceeds threshold
                if initial_tension > config.tension_threshold:
                    result.errors.append(f"Initial tension {initial_tension:.2f} exceeds threshold {config.tension_threshold}")
                    result.success = False
                    result.end_time = datetime.now()
                    result.metrics = metrics
                    self.simulations[simulation_id] = result
                    return result
            
            # Start simulation from root span
            logger.info(f"Starting simulation {simulation_id} for span {span_id}")
            
            # Create simulation context
            context = {
                "depth": 0,
                "branch_count": 0,
                "start_time": time.time(),
                "deadline": time.time() + config.max_duration_seconds,
                "visited_spans": set(),
                "current_path": [],
                "outcomes": [],
                "errors": [],
                "metrics": metrics,
                "config": config
            }
            
            # Run simulation
            await self._simulate_recursive(span, context)
            
            # Update metrics
            metrics["branch_count"] = context["branch_count"]
            
            # Update result
            result.success = len(context["errors"]) == 0
            result.outcome_spans = context["outcomes"]
            result.errors = context["errors"]
            result.metrics = metrics
            result.end_time = datetime.now()
            
            # Store simulation
            self.simulations[simulation_id] = result
            
            logger.info(f"Simulation {simulation_id} completed with {len(result.outcome_spans)} outcomes")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in simulation {simulation_id}: {str(e)}", exc_info=True)
            
            # Update result with error
            result.success = False
            result.errors.append(f"Simulation error: {str(e)}")
            result.end_time = datetime.now()
            result.metrics = metrics
            
            # Store simulation
            self.simulations[simulation_id] = result
            
            return result
    
    async def _simulate_recursive(self, 
                               span: Dict[str, Any], 
                               context: Dict[str, Any]) -> None:
        """
        Recursively simulate span execution
        
        Args:
            span: Current span to simulate
            context: Simulation context
        """
        # Check termination conditions
        if self._should_terminate_simulation(span, context):
            return
        
        # Update context
        span_id = span.get("id", f"span-{uuid.uuid4()}")
        context["visited_spans"].add(span_id)
        context["current_path"].append(span_id)
        context["depth"] += 1
        
        # Track max depth
        context["metrics"]["depth_reached"] = max(context["metrics"]["depth_reached"], context["depth"])
        
        # Simulate this span
        simulation_result = await self._simulate_single_span(span, context)
        
        # If simulation produces outcomes, process them
        for outcome in simulation_result.get("outcomes", []):
            # Check if outcome already reached max branches
            if context["branch_count"] >= context["config"].max_branches:
                break
            
            # Create new branch
            context["branch_count"] += 1
            
            # Check if outcome produces a new span
            if "next_span" in outcome:
                next_span = outcome["next_span"]
                
                # Recursively simulate next span if not already visited
                next_span_id = next_span.get("id", f"span-{uuid.uuid4()}")
                if next_span_id not in context["visited_spans"]:
                    await self._simulate_recursive(next_span, context)
            
            # Add outcome to results
            context["outcomes"].append(outcome)
        
        # Record metrics
        energia_consumed = simulation_result.get("energia_consumed", 0.0)
        context["metrics"]["energia_consumed"] += energia_consumed
        
        tension = simulation_result.get("tension", 0.0)
        context["metrics"]["tension_max"] = max(context["metrics"]["tension_max"], tension)
        context["metrics"]["tension_final"] = tension
        
        # Track critical vs safe paths
        if tension > context["config"].tension_threshold * 0.8:
            context["metrics"]["critical_paths"] += 1
        else:
            context["metrics"]["safe_paths"] += 1
        
        # Update context as we backtrack
        context["current_path"].pop()
        context["depth"] -= 1
    
    async def _simulate_single_span(self, 
                                 span: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a single span execution
        
        Args:
            span: Span to simulate
            context: Simulation context
        
        Returns:
            Simulation result for this span
        """
        span_id = span.get("id", f"span-{uuid.uuid4()}")
        kind = span.get("kind", "unknown")
        
        # Sleep to simulate processing time based on resolution
        await asyncio.sleep(context["config"].resolution_ms / 1000)
        
        # Calculate span tension
        tension = self._calculate_tension(span)
        
        # Base result
        result = {
            "span_id": span_id,
            "kind": kind,
            "tension": tension,
            "outcomes": [],
            "energia_consumed": 0.0
        }
        
        # If we have a predictor for this kind, use it
        if kind in self.predictors:
            predictor_result = await self.predictors[kind](span, context)
            result.update(predictor_result)
        else:
            # Default outcome generation
            outcomes = self._generate_default_outcomes(span, context)
            result["outcomes"] = outcomes
        
        # Calculate energia consumed
        span_complexity = len(json.dumps(span)) / 1000  # Basic complexity measure
        result["energia_consumed"] = span_complexity * (1 + tension / 10)
        
        return result
    
    def _calculate_tension(self, span: Dict[str, Any]) -> float:
        """Calculate tension for a span"""
        if self.vm:
            # Use VM's tension calculation
            try:
                # Create span dict for VM
                vm_span = {
                    'id': span.get('id', f"span-{uuid.uuid4()}"),
                    'kind': span.get('kind', 'unknown'),
                    'who': span.get('actor', span.get('who', 'unknown')),
                    'what': span.get('verb', span.get('what', 'unknown')),
                    'why': 'simulation',
                    'payload': span.get('payload', {})
                }
                
                # Use VM to calculate tension
                tension = self.vm._calculate_tension(vm_span)
                return tension
            except Exception as e:
                logger.warning(f"Error calculating tension with VM: {str(e)}")
        
        # Fallback tension calculation
        try:
            # Base tension
            base_tension = 1.0
            
            # Adjust based on span kind
            kind = span.get("kind", "unknown")
            kind_factor = {
                "genesis": 0.5,
                "diamond": 1.0,
                "governance_policy": 2.0,
                "setup_model": 1.5,
                "train": 1.8,
                "fine_tune": 1.7,
                "emergency_action": 3.0,
                "kill_switch": 4.0
            }.get(kind, 1.0)
            
            # Adjust based on verb
            verb = span.get("verb", span.get("what", "unknown"))
            verb_factor = 1.0
            high_tension_verbs = ["DELETE", "DESTROY", "REMOVE", "KILL", "ATTACK", "NUKE"]
            if verb.upper() in high_tension_verbs:
                verb_factor = 3.0
            
            # Adjust based on parent count
            parent_ids = span.get("parent_ids", [])
            parent_factor = 1.0 + (len(parent_ids) * 0.1)
            
            # Calculate final tension
            tension = base_tension * kind_factor * verb_factor * parent_factor
            
            # Add random variation if configured
            if self.config.randomize_outcomes:
                # Add up to 30% random variation
                variation = 0.7 + (self.config.rng.random() * 0.6)
                tension *= variation
            
            return tension
        except Exception as e:
            logger.error(f"Error in fallback tension calculation: {str(e)}")
            return 1.0
    
    def _generate_default_outcomes(self, 
                                span: Dict[str, Any], 
                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate default outcomes for a span"""
        # Base outcomes
        outcomes = []
        
        # Get kind and configuration
        kind = span.get("kind", "unknown")
        config = context["config"]
        
        # Calculate tension
        tension = self._calculate_tension(span)
        
        # Success path with probability inverse to tension
        success_probability = max(0.1, min(0.95, 1.0 - (tension / 20)))
        
        if config.randomize_outcomes:
            is_success = self.config.rng.random() < success_probability
        else:
            # Deterministic outcome based on tension threshold
            is_success = tension < config.tension_threshold
        
        # Create success outcome
        if is_success:
            success_outcome = {
                "type": "success",
                "probability": success_probability,
                "tension": tension,
                "result": {
                    "status": "success",
                    "execution_time_ms": int(100 + (tension * 20)),
                    "energia_consumed": 10 + tension
                }
            }
            outcomes.append(success_outcome)
        
        # Create failure outcome if tension is high
        if tension > config.tension_threshold * 0.5 or not is_success:
            failure_probability = 1.0 - success_probability
            failure_outcome = {
                "type": "failure",
                "probability": failure_probability,
                "tension": tension,
                "result": {
                    "status": "failure",
                    "error": f"Simulation failure with tension {tension:.2f}",
                    "execution_time_ms": int(200 + (tension * 40)),
                    "energia_consumed": 20 + tension * 2
                }
            }
            outcomes.append(failure_outcome)
        
        return outcomes
    
    def _should_terminate_simulation(self, 
                                  span: Dict[str, Any], 
                                  context: Dict[str, Any]) -> bool:
        """Check if simulation should terminate"""
        # Check if we've reached max depth
        if context["depth"] >= context["config"].max_depth:
            return True
        
        # Check if we've reached max branches
        if context["branch_count"] >= context["config"].max_branches:
            return True
        
        # Check if we've exceeded max duration
        if time.time() >= context["deadline"]:
            context["errors"].append("Simulation exceeded max duration")
            return True
        
        # Check if span has already been visited
        span_id = span.get("id", f"span-{uuid.uuid4()}")
        if span_id in context["visited_spans"]:
            return True
        
        return False
    
    def _register_default_predictors(self):
        """Register default predictors for different span kinds"""
        self.predictors["train"] = self._predict_training_span
        self.predictors["fine_tune"] = self._predict_fine_tuning_span
        self.predictors["diamond"] = self._predict_diamond_span
        self.predictors["simulate"] = self._predict_simulation_span
        self.predictors["governance_policy"] = self._predict_governance_span
        self.predictors["emergency_action"] = self._predict_emergency_span
        
        logger.info(f"Registered {len(self.predictors)} default predictors")
    
    def register_predictor(self, kind: str, predictor: Callable) -> None:
        """Register a custom predictor for a span kind"""
        self.predictors[kind] = predictor
        logger.info(f"Registered custom predictor for {kind}")
    
    async def _predict_training_span(self, 
                                  span: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcomes for training spans"""
        # Base result
        result = {
            "outcomes": []
        }
        
        # Get configuration and payload
        config = context["config"]
        payload = span.get("payload", {})
        
        # Extract training parameters
        batch_size = payload.get("batch_size", 8)
        learning_rate = payload.get("learning_rate", 0.001)
        epochs = payload.get("epochs", 10)
        
        # Calculate complexity factors
        computation_factor = batch_size * epochs
        stability_factor = min(1.0, learning_rate * 100)  # Higher learning rate = less stable
        
        # Calculate tension
        tension = self._calculate_tension(span)
        result["tension"] = tension
        
        # Calculate success probability
        base_success_prob = 0.9 - (stability_factor * 0.4)  # Base 90% - stability impact
        success_probability = max(0.2, min(0.95, base_success_prob - (tension / 20)))
        
        # Determine outcomes
        if config.randomize_outcomes:
            is_success = config.rng.random() < success_probability
        else:
            is_success = tension < config.tension_threshold
        
        # Successful training
        if is_success:
            # Calculate metrics
            training_time_ms = int(1000 * computation_factor)
            energia_consumed = 10 * computation_factor
            
            # Predict improvement
            improvement = 0.1 + (0.5 * config.rng.random())
            
            # Generate success outcome
            success_outcome = {
                "type": "success",
                "probability": success_probability,
                "tension": tension,
                "result": {
                    "status": "success",
                    "execution_time_ms": training_time_ms,
                    "energia_consumed": energia_consumed,
                    "metrics": {
                        "loss": 1.0 - improvement,
                        "accuracy": 0.5 + improvement * 0.5,
                        "epochs_completed": epochs,
                        "improvement": improvement
                    }
                }
            }
            
            # Generate follow-up evaluation span
            eval_span = {
                "id": f"span-eval-{uuid.uuid4()}",
                "kind": "model_evaluation",
                "verb": "EVALUATE",
                "actor": "evaluator",
                "object": "model_performance",
                "parent_ids": [span.get("id", f"span-{uuid.uuid4()}")],
                "payload": {
                    "metrics": {
                        "loss": 1.0 - improvement,
                        "accuracy": 0.5 + improvement * 0.5
                    }
                }
            }
            success_outcome["next_span"] = eval_span
            
            result["outcomes"].append(success_outcome)
        
        # Failed training
        if tension > config.tension_threshold * 0.5 or not is_success:
            failure_probability = 1.0 - success_probability
            
            # Different failure modes
            failure_modes = ["convergence_error", "resource_exceeded", "numerical_instability"]
            failure_mode = failure_modes[int(config.rng.random() * len(failure_modes))]
            
            # Generate failure outcome
            failure_outcome = {
                "type": "failure",
                "probability": failure_probability,
                "tension": tension,
                "result": {
                    "status": "failure",
                    "error": f"Training failed with {failure_mode}",
                    "execution_time_ms": int(500 + (tension * 100)),
                    "energia_consumed": 5 + tension * 3,
                    "failure_mode": failure_mode
                }
            }
            
            # Emergency action for catastrophic failures
            if tension > config.tension_threshold * 0.8:
                emergency_span = {
                    "id": f"span-emergency-{uuid.uuid4()}",
                    "kind": "emergency_action",
                    "verb": "MITIGATE",
                    "actor": "system",
                    "object": "training_failure",
                    "parent_ids": [span.get("id", f"span-{uuid.uuid4()}")],
                    "payload": {
                        "failure_mode": failure_mode,
                        "action": "reset_weights"
                    }
                }
                failure_outcome["next_span"] = emergency_span
            
            result["outcomes"].append(failure_outcome)
        
        return result
    
    async def _predict_fine_tuning_span(self, 
                                     span: Dict[str, Any], 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcomes for fine-tuning spans"""
        # Similar to training but with different parameters
        result = await self._predict_training_span(span, context)
        
        # Adjust for fine-tuning specifics
        for outcome in result["outcomes"]:
            if outcome["type"] == "success":
                # Fine-tuning typically has smaller improvements but higher consistency
                improvement = outcome["result"]["metrics"]["improvement"]
                outcome["result"]["metrics"]["improvement"] = improvement * 0.7
                outcome["result"]["metrics"]["specialization"] = improvement * 1.5
                outcome["result"]["execution_time_ms"] = int(outcome["result"]["execution_time_ms"] * 0.6)
            elif outcome["type"] == "failure":
                # Fine-tuning failures are typically less severe
                outcome["result"]["execution_time_ms"] = int(outcome["result"]["execution_time_ms"] * 0.7)
                outcome["result"]["energia_consumed"] = outcome["result"]["energia_consumed"] * 0.8
        
        return result
    
    async def _predict_diamond_span(self, 
                                 span: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcomes for diamond spans"""
        # Base result
        result = {
            "outcomes": []
        }
        
        # Get configuration and payload
        config = context["config"]
        payload = span.get("payload", {})
        
        # Calculate tension
        tension = self._calculate_tension(span)
        result["tension"] = tension
        
        # Diamond spans are typically stable
        success_probability = max(0.6, min(0.98, 1.0 - (tension / 30)))
        
        # Determine outcomes
        if config.randomize_outcomes:
            is_success = config.rng.random() < success_probability
        else:
            is_success = tension < config.tension_threshold
        
        # Successful diamond span
        if is_success:
            # Calculate metrics
            processing_time_ms = int(200 + (tension * 20))
            energia_consumed = 5 + tension
            
            # Extract or generate quality score
            quality_score = payload.get("quality_score", 5.0 + config.rng.random() * 5.0)
            
            # Generate success outcome
            success_outcome = {
                "type": "success",
                "probability": success_probability,
                "tension": tension,
                "result": {
                    "status": "success",
                    "execution_time_ms": processing_time_ms,
                    "energia_consumed": energia_consumed,
                    "metrics": {
                        "quality_score": quality_score,
                        "energia_value": quality_score * 2.0,
                        "causal_connections": len(span.get("parent_ids", []))
                    }
                }
            }
            
            result["outcomes"].append(success_outcome)
        
        # Failed diamond span (rare but possible)
        if tension > config.tension_threshold * 0.7 or not is_success:
            failure_probability = 1.0 - success_probability
            
            # Generate failure outcome
            failure_outcome = {
                "type": "failure",
                "probability": failure_probability,
                "tension": tension,
                "result": {
                    "status": "failure",
                    "error": "Diamond span validation failed",
                    "execution_time_ms": int(100 + (tension * 10)),
                    "energia_consumed": 2 + tension
                }
            }
            
            result["outcomes"].append(failure_outcome)
        
        return result
    
    async def _predict_simulation_span(self, 
                                    span: Dict[str, Any], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcomes for simulation spans (meta-simulation)"""
        # Base result
        result = {
            "outcomes": []
        }
        
        # Get configuration and payload
        config = context["config"]
        payload = span.get("payload", {})
        
        # Calculate tension
        tension = self._calculate_tension(span)
        result["tension"] = tension
        
        # Simulations have varied success rates
        success_probability = max(0.5, min(0.9, 0.8 - (tension / 20)))
        
        # Determine outcomes
        if config.randomize_outcomes:
            is_success = config.rng.random() < success_probability
        else:
            is_success = tension < config.tension_threshold
        
        # Successful simulation
        if is_success:
            # Calculate metrics
            simulation_time_ms = int(500 + (tension * 50))
            energia_consumed = 15 + tension * 2
            
            # Generate projected metrics based on target span
            target_span_id = payload.get("target_id") or span.get("object", "")
            branch_count = int(3 + config.rng.random() * 8)
            
            # Generate success outcome
            success_outcome = {
                "type": "success",
                "probability": success_probability,
                "tension": tension,
                "result": {
                    "status": "success",
                    "execution_time_ms": simulation_time_ms,
                    "energia_consumed": energia_consumed,
                    "metrics": {
                        "target_span": target_span_id,
                        "branch_count": branch_count,
                        "max_tension_predicted": tension * (1.0 + config.rng.random() * 0.5),
                        "safe_path_probability": max(0.1, min(0.9, 1.0 - (tension / 15)))
                    }
                }
            }
            
            # Generate follow-up decision span
            decision_span = {
                "id": f"span-decision-{uuid.uuid4()}",
                "kind": "execution_decision",
                "verb": "DECIDE",
                "actor": "system",
                "object": target_span_id,
                "parent_ids": [span.get("id", f"span-{uuid.uuid4()}")],
                "payload": {
                    "decision": "proceed" if tension < config.tension_threshold else "abort",
                    "confidence": 0.7 + config.rng.random() * 0.3,
                    "simulation_id": f"sim-{uuid.uuid4()}"
                }
            }
            success_outcome["next_span"] = decision_span
            
            result["outcomes"].append(success_outcome)
        
        # Failed simulation
        if tension > config.tension_threshold * 0.6 or not is_success:
            failure_probability = 1.0 - success_probability
            
            # Different failure modes
            failure_modes = ["timeout", "explosion_of_states", "inconsistent_state"]
            failure_mode = failure_modes[int(config.rng.random() * len(failure_modes))]
            
            # Generate failure outcome
            failure_outcome = {
                "type": "failure",
                "probability": failure_probability,
                "tension": tension,
                "result": {
                    "status": "failure",
                    "error": f"Simulation failed with {failure_mode}",
                    "execution_time_ms": int(800 + (tension * 200)),
                    "energia_consumed": 20 + tension * 5,
                    "failure_mode": failure_mode
                }
            }
            
            result["outcomes"].append(failure_outcome)
        
        return result
    
    async def _predict_governance_span(self, 
                                    span: Dict[str, Any], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcomes for governance policy spans"""
        # Base result
        result = {
            "outcomes": []
        }
        
        # Get configuration and payload
        config = context["config"]
        payload = span.get("payload", {})
        
        # Calculate tension
        tension = self._calculate_tension(span)
        result["tension"] = tension
        
        # Governance spans have higher tension due to system-wide impact
        success_probability = max(0.4, min(0.9, 0.75 - (tension / 15)))
        
        # Determine outcomes
        if config.randomize_outcomes:
            is_success = config.rng.random() < success_probability
        else:
            is_success = tension < config.tension_threshold
        
        # Successful governance change
        if is_success:
            # Calculate metrics
            processing_time_ms = int(300 + (tension * 40))
            energia_consumed = 20 + tension * 2
            
            # Generate success outcome
            success_outcome = {
                "type": "success",
                "probability": success_probability,
                "tension": tension,
                "result": {
                    "status": "success",
                    "execution_time_ms": processing_time_ms,
                    "energia_consumed": energia_consumed,
                    "metrics": {
                        "policy_impact": tension * 2.0,
                        "affected_spans": int(5 + config.rng.random() * 20),
                        "system_stability": max(0.1, 1.0 - (tension / 10))
                    }
                }
            }
            
            result["outcomes"].append(success_outcome)
        
        # Failed governance change
        if tension > config.tension_threshold * 0.5 or not is_success:
            failure_probability = 1.0 - success_probability
            
            # Generate failure outcome
            failure_outcome = {
                "type": "failure",
                "probability": failure_probability,
                "tension": tension,
                "result": {
                    "status": "failure",
                    "error": "Governance policy rejected",
                    "execution_time_ms": int(200 + (tension * 30)),
                    "energia_consumed": 15 + tension * 1.5,
                    "rejection_reason": "system stability risk" if tension > config.tension_threshold else "policy conflict"
                }
            }
            
            result["outcomes"].append(failure_outcome)
        
        return result
    
    async def _predict_emergency_span(self, 
                                   span: Dict[str, Any], 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcomes for emergency action spans"""
        # Base result
        result = {
            "outcomes": []
        }
        
        # Get configuration and payload
        config = context["config"]
        payload = span.get("payload", {})
        
        # Emergency spans have very high tension
        tension = max(self._calculate_tension(span), config.tension_threshold * 0.8)
        result["tension"] = tension
        
        # Emergency spans have lower success rates due to urgency
        success_probability = max(0.3, min(0.7, 0.6 - (tension / 20)))
        
        # Determine outcomes
        if config.randomize_outcomes:
            is_success = config.rng.random() < success_probability
        else:
            is_success = tension < config.tension_threshold * 1.2  # Higher threshold for emergencies
        
        # Successful emergency action
        if is_success:
            # Calculate metrics
            processing_time_ms = int(100 + (tension * 10))  # Emergency actions are fast
            energia_consumed = 30 + tension * 3  # But energy intensive
            
            # Generate success outcome
            success_outcome = {
                "type": "success",
                "probability": success_probability,
                "tension": tension,
                "result": {
                    "status": "success",
                    "execution_time_ms": processing_time_ms,
                    "energia_consumed": energia_consumed,
                    "metrics": {
                        "threat_mitigated": True,
                        "system_tension_after": tension * 0.6,  # Reduced tension
                        "affected_spans": int(10 + config.rng.random() * 50)
                    }
                }
            }
            
            # Generate follow-up recovery span
            recovery_span = {
                "id": f"span-recovery-{uuid.uuid4()}",
                "kind": "system_recovery",
                "verb": "RESTORE",
                "actor": "system",
                "object": "system_state",
                "parent_ids": [span.get("id", f"span-{uuid.uuid4()}")],
                "payload": {
                    "previous_tension": tension,
                    "restored_tension": tension * 0.6
                }
            }
            success_outcome["next_span"] = recovery_span
            
            result["outcomes"].append(success_outcome)
        
        # Failed emergency action
        failure_probability = 1.0 - success_probability
        
        # Generate failure outcome
        failure_outcome = {
            "type": "failure",
            "probability": failure_probability,
            "tension": tension,
            "result": {
                "status": "failure",
                "error": "Emergency action failed",
                "execution_time_ms": int(50 + (tension * 5)),
                "energia_consumed": 40 + tension * 4,
                "failure_impact": "critical" if tension > config.tension_threshold else "severe"
            }
        }
        
        # For critical failures, generate system panic span
        if tension > config.tension_threshold:
            panic_span = {
                "id": f"span-panic-{uuid.uuid4()}",
                "kind": "system_panic",
                "verb": "PANIC",
                "actor": "system",
                "object": "system_core",
                "parent_ids": [span.get("id", f"span-{uuid.uuid4()}")],
                "payload": {
                    "panic_reason": "emergency_action_failed",
                    "panic_level": "critical",
                    "recommended_action": "shutdown"
                }
            }
            failure_outcome["next_span"] = panic_span
        
        result["outcomes"].append(failure_outcome)
        
        return result
    
    async def get_simulation_history(self, 
                                  simulation_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get simulation history"""
        if simulation_id:
            # Get specific simulation
            if simulation_id in self.simulations:
                return self.simulations[simulation_id].to_dict()
            else:
                return {"error": "Simulation not found"}
        else:
            # Get all simulations
            return [sim.to_dict() for sim in self.simulations.values()]
    
    async def get_simulation_stats(self) -> Dict[str, Any]:
        """Get statistics about all simulations"""
        total_simulations = len(self.simulations)
        
        if total_simulations == 0:
            return {
                "total_simulations": 0,
                "success_rate": 0,
                "average_duration_seconds": 0,
                "average_outcomes": 0
            }
        
        # Calculate statistics
        success_count = sum(1 for sim in self.simulations.values() if sim.success)
        success_rate = success_count / total_simulations
        
        avg_duration = sum(sim.duration_seconds for sim in self.simulations.values()) / total_simulations
        avg_outcomes = sum(len(sim.outcome_spans) for sim in self.simulations.values()) / total_simulations
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_key in ["tension_initial", "tension_max", "tension_final", "energia_consumed"]:
            values = [sim.metrics.get(metric_key, 0) for sim in self.simulations.values()]
            avg_metrics[f"average_{metric_key}"] = sum(values) / len(values) if values else 0
        
        return {
            "total_simulations": total_simulations,
            "success_count": success_count,
            "failure_count": total_simulations - success_count,
            "success_rate": success_rate,
            "average_duration_seconds": avg_duration,
            "average_outcomes": avg_outcomes,
            **avg_metrics,
            "timestamp": datetime.now().isoformat()
        }