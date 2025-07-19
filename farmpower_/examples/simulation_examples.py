"""
Simulation Examples for LogLineOS/DiamondSpan
Examples of different simulation scenarios
Created: 2025-07-19 05:37:29 UTC
User: danvoulez
"""
import os
import json
import asyncio
import uuid
from datetime import datetime

# Import simulation components
try:
    from core.simulation_engine import SimulationEngine, SimulationConfig
    from core.logline_vm import LogLineVM
except ImportError:
    print("Failed to import simulation components. Make sure they're in your Python path.")
    print("You can run these examples from the project root using:")
    print("python -m examples.simulation_examples")
    import sys
    sys.exit(1)

# Create output directory
os.makedirs("examples/output", exist_ok=True)

# Sample spans for simulation
SAMPLE_SPANS = {
    "training": {
        "id": f"span-{uuid.uuid4()}",
        "kind": "train",
        "verb": "EXECUTE",
        "actor": "danvoulez",
        "object": "model_training",
        "parent_ids": [],
        "payload": {
            "corpus": "/spans/diamante/linguamater",
            "batch_size": 8,
            "learning_rate": 0.0002,
            "epochs": 10,
            "use_grammatical_acceleration": True
        }
    },
    "diamond": {
        "id": f"span-{uuid.uuid4()}",
        "kind": "diamond",
        "verb": "STORE",
        "actor": "danvoulez",
        "object": "knowledge_fragment",
        "parent_ids": [],
        "payload": {
            "text": "Diamond span containing high-value knowledge about LogLineOS architecture and governance.",
            "tags": ["architecture", "governance", "knowledge"],
            "grammar_complexity": 8.5,
            "cognitive_value": 17.3
        }
    },
    "governance": {
        "id": f"span-{uuid.uuid4()}",
        "kind": "governance_policy",
        "verb": "ESTABLISH",
        "actor": "danvoulez",
        "object": "voting_rules",
        "parent_ids": [],
        "payload": {
            "quorum_threshold": 0.51,
            "proposal_duration_days": 7,
            "vote_weight_by_energy": True,
            "min_proposer_energy": 100.0
        }
    },
    "emergency": {
        "id": f"span-{uuid.uuid4()}",
        "kind": "emergency_action",
        "verb": "MITIGATE",
        "actor": "system",
        "object": "thermal_overload",
        "parent_ids": [],
        "payload": {
            "temperature": 92.5,
            "action": "throttle_processing",
            "target_temperature": 75.0
        }
    },
    "simulation": {
        "id": f"span-{uuid.uuid4()}",
        "kind": "simulate",
        "verb": "PREDICT",
        "actor": "system",
        "object": "span-abc123", # Would be a real span ID in practice
        "parent_ids": [],
        "payload": {
            "expected_memory": "14.7GB",
            "expected_duration": "2.1h",
            "risk_factors": ["swap_usage>30%", "thermal_throttling"],
            "decision_threshold": "mem<15GB AND temp<90°C"
        }
    }
}

# Function to save simulation result
def save_result(result, filename):
    """Save simulation result to file"""
    result_dict = result.to_dict()
    filepath = f"examples/output/{filename}"
    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"Saved result to {filepath}")

# Example 1: Basic simulation
async def basic_simulation():
    """Run a basic simulation on a training span"""
    print("\n=== Example 1: Basic Training Span Simulation ===")
    
    # Create simulation engine with default config
    config = SimulationConfig()
    engine = SimulationEngine(config)
    
    # Run simulation
    span = SAMPLE_SPANS["training"]
    print(f"Simulating span: {span['kind']} - {span['verb']}")
    
    result = await engine.simulate_span(span)
    
    # Print results
    print(f"Simulation completed: {result.success}")
    print(f"Outcome count: {len(result.outcome_spans)}")
    print(f"Initial tension: {result.metrics.get('tension_initial', 0):.2f}")
    print(f"Max tension: {result.metrics.get('tension_max', 0):.2f}")
    print(f"Energia consumed: {result.metrics.get('energia_consumed', 0):.2f}")
    
    # Save results
    save_result(result, "basic_training_simulation.json")

# Example 2: High-tension simulation
async def high_tension_simulation():
    """Run a simulation with high tension"""
    print("\n=== Example 2: High-Tension Emergency Span Simulation ===")
    
    # Create simulation engine with stricter config
    config = SimulationConfig(
        max_depth=3,
        tension_threshold=10.0,  # Lower threshold to trigger more failures
        randomize_outcomes=False  # Make it deterministic
    )
    engine = SimulationEngine(config)
    
    # Run simulation