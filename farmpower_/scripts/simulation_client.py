#!/usr/bin/env python3
"""
Simulation Client for LogLineOS/DiamondSpan
Command-line tool to interact with the simulation system
Created: 2025-07-19 05:37:29 UTC
User: danvoulez
"""
import os
import sys
import json
import argparse
import requests
import datetime
from tabulate import tabulate
from typing import Dict, List, Any, Optional

class SimulationClient:
    """Client for the LogLineOS Simulation API"""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """Initialize simulation client"""
        self.base_url = base_url or os.environ.get("SIMULATION_API_URL", "http://localhost:8084")
        self.api_key = api_key or os.environ.get("SIMULATION_API_KEY", "default_key")
    
    def _request(self, method: str, path: str, data: Dict = None) -> Dict:
        """Make a request to the simulation API"""
        url = f"{self.base_url}{path}"
        headers = {"X-API-Key": self.api_key}
        
        if method.lower() == "get":
            response = requests.get(url, headers=headers)
        elif method.lower() == "post":
            headers["Content-Type"] = "application/json"
            response = requests.post(url, json=data, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        if response.status_code >= 400:
            print(f"Error {response.status_code}: {response.text}")
            sys.exit(1)
        
        return response.json()
    
    def simulate_span(self, span: Dict[str, Any], config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simulate a span"""
        data = {
            "span": span,
            "config_override": config_override
        }
        return self._request("post", "/simulate", data)
    
    def get_simulation(self, simulation_id: str) -> Dict[str, Any]:
        """Get a specific simulation"""
        return self._request("get", f"/simulations/{simulation_id}")
    
    def list_simulations(self) -> Dict[str, Any]:
        """List all simulations"""
        return self._request("get", "/simulations")
    
    def get_outcomes(self, simulation_id: str) -> Dict[str, Any]:
        """Get outcomes for a specific simulation"""
        return self._request("get", f"/simulations/{simulation_id}/outcomes")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simulation statistics"""
        return self._request("get", "/stats")

def load_span_from_file(file_path: str) -> Dict[str, Any]:
    """Load a span from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading span from {file_path}: {str(e)}")
        sys.exit(1)

def save_json_to_file(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving to {file_path}: {str(e)}")
        sys.exit(1)

def format_simulation_result(result: Dict[str, Any]) -> str:
    """Format simulation result for display"""
    return f"""
Simulation: {result.get('simulation_id', 'unknown')}
Status: {'✅ Success' if result.get('success') else '❌ Failed'}
Outcomes: {result.get('outcome_count', 0)}
Duration: {result.get('duration_seconds', 0):.2f}s

Metrics:
  - Tension (initial): {result.get('metrics', {}).get('tension_initial', 0):.2f}
  - Tension (max): {result.get('metrics', {}).get('tension_max', 0):.2f}
  - Tension (final): {result.get('metrics', {}).get('tension_final', 0):.2f}
  - Energia consumed: {result.get('metrics', {}).get('energia_consumed', 0):.2f}
  - Critical paths: {result.get('metrics', {}).get('critical_paths', 0)}
  - Safe paths: {result.get('metrics', {}).get('safe_paths', 0)}
"""

def format_outcomes(outcomes: List[Dict[str, Any]]) -> str:
    """Format simulation outcomes for display"""
    if not outcomes:
        return "No outcomes found."
    
    table_data = []
    for outcome in outcomes:
        result = outcome.get("result", {})
        table_data.append([
            outcome.get("type", "unknown"),
            f"{outcome.get('probability', 0):.2f}",
            f"{outcome.get('tension', 0):.2f}",
            result.get("status", "unknown"),
            f"{result.get('execution_time_ms', 0)}ms",
            f"{result.get('energia_consumed', 0):.2f}"
        ])
    
    headers = ["Type", "Probability", "Tension", "Status", "Time", "Energia"]
    return tabulate(table_data, headers=headers, tablefmt="grid")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LogLineOS Simulation Client")
    parser.add_argument("--url", help="Simulation API URL", default=os.environ.get("SIMULATION_API_URL", "http://localhost:8084"))
    parser.add_argument("--key", help="API Key", default=os.environ.get("SIMULATION_API_KEY", "default_key"))
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # simulate command
    simulate_parser = subparsers.add_parser("simulate", help="Simulate a span")
    simulate_parser.add_argument("span_file", help="Path to span JSON file")
    simulate_parser.add_argument("--config", help="Path to config override JSON file")
    simulate_parser.add_argument("--output", help="Save result to file")
    
    # get command
    get_parser = subparsers.add_parser("get", help="Get a simulation")
    get_parser.add_argument("simulation_id", help="Simulation ID")
    get_parser.add_argument("--output", help="Save result to file")
    
    # list command
    list_parser = subparsers.add_parser("list", help="List all simulations")
    list_parser.add_argument("--output", help="Save result to file")
    
    # outcomes command
    outcomes_parser = subparsers.add_parser("outcomes", help="Get simulation outcomes")
    outcomes_parser.add_argument("simulation_id", help="Simulation ID")
    outcomes_parser.add_argument("--output", help="Save result to file")
    
    # stats command
    stats_parser = subparsers.add_parser("stats", help="Get simulation statistics")
    stats_parser.add_argument("--output", help="Save result to file")
    
    args = parser.parse_args()
    
    client = SimulationClient(args.url, args.key)
    
    if args.command == "simulate":
        # Load span from file
        span = load_span_from_file(args.span_file)
        
        # Load config override if provided
        config_override = None
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    config_override = json.load(f)
            except Exception as e:
                print(f"Error loading config from {args.config}: {str(e)}")
                sys.exit(1)
        
        # Run simulation
        result = client.simulate_span(span, config_override)
        print(format_simulation_result(result))
        
        # Save result if requested
        if args.output:
            save_json_to_file(result, args.output)
    
    elif args.command == "get":
        result = client.get_simulation(args.simulation_id)
        print(json.dumps(result, indent=2))
        
        # Save result if requested
        if args.output:
            save_json_to_file(result, args.output)
    
    elif args.command == "list":
        result = client.list_simulations()
        simulations = result.get("simulations", [])
        
        print(f"Found {len(simulations)} simulations:")
        for sim in simulations:
            sim_id = sim.get("simulation_id", "unknown")
            success = "✅" if sim.get("success") else "❌"
            outcomes = sim.get("outcome_count", 0)
            duration = sim.get("duration_seconds", 0)
            print(f"{sim_id}: {success} {outcomes} outcomes in {duration:.2f}s")
        
        # Save result if requested
        if args.output:
            save_json_to_file(result, args.output)
    
    elif args.command == "outcomes":
        result = client.get_outcomes(args.simulation_id)
        outcomes = result.get("outcomes", [])
        
        print(f"Outcomes for simulation {args.simulation_id}:")
        print(format_outcomes(outcomes))
        
        # Save result if requested
        if args.output:
            save_json_to_file(result, args.output)
    
    elif args.command == "stats":
        result = client.get_stats()
        
        print("Simulation Statistics:")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Save result if requested
        if args.output:
            save_json_to_file(result, args.output)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()