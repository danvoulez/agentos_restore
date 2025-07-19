#!/usr/bin/env python3
"""
Audit Client for LogLineOS/DiamondSpan
Command-line tool to interact with the audit system
Created: 2025-07-19 05:28:32 UTC
User: danvoulez
"""
import os
import sys
import json
import argparse
import requests
import datetime
import textwrap
from tabulate import tabulate
from typing import Dict, List, Any, Optional

class AuditClient:
    """Client for the LogLineOS Audit API"""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url or os.environ.get("AUDIT_API_URL", "http://localhost:8083")
        self.api_key = api_key or os.environ.get("AUDIT_API_KEY", "default_key")
    
    def _request(self, method: str, path: str, data: Dict = None) -> Dict:
        """Make a request to the audit API"""
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
    
    def create_entry(self, operation: str, actor: str, target_type: str, 
                     target_id: str, status: str = "success", 
                     details: Dict = None, metadata: Dict = None) -> Dict:
        """Create a new audit entry"""
        data = {
            "operation": operation,
            "actor": actor,
            "target_type": target_type,
            "target_id": target_id,
            "status": status,
            "details": details or {},
            "metadata": metadata or {}
        }
        return self._request("post", "/entries", data)
    
    def get_entry(self, entry_id: str) -> Dict:
        """Get an audit entry by ID"""
        return self._request("get", f"/entries/{entry_id}")
    
    def query_entries(self, operations: List[str] = None, actors: List[str] = None,
                     target_types: List[str] = None, target_ids: List[str] = None,
                     statuses: List[str] = None, start_time: str = None,
                     end_time: str = None, limit: int = 100, offset: int = 0) -> Dict:
        """Query audit entries with filters"""
        data = {
            "operations": operations,
            "actors": actors,
            "target_types": target_types,
            "target_ids": target_ids,
            "statuses": statuses,
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit,
            "offset": offset
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return self._request("post", "/query", data)
    
    def get_recent(self, hours: int = 24, limit: int = 50) -> Dict:
        """Get recent audit entries"""
        return self._request("get", f"/recent?hours={hours}&limit={limit}")
    
    def get_statistics(self) -> Dict:
        """Get audit system statistics"""
        return self._request("get", "/statistics")
    
    def validate_audit_trail(self, target_type: str = None, target_id: str = None,
                           start_time: str = None, end_time: str = None) -> Dict:
        """Validate the audit trail integrity"""
        # This requires god mode authentication, which the CLI doesn't support yet
        print("Validation requires god mode authentication, not implemented in CLI")
        return {}

def format_entry(entry: Dict) -> str:
    """Format an audit entry for display"""
    timestamp = entry.get("timestamp", "").replace("T", " ").split(".")[0]
    details = json.dumps(entry.get("details", {}), indent=2)
    if len(details) > 200:
        details = details[:197] + "..."
    
    return textwrap.dedent(f"""
    ID: {entry.get("id")}
    Timestamp: {timestamp}
    Operation: {entry.get("operation")}
    Actor: {entry.get("actor")}
    Target: {entry.get("target_type")}/{entry.get("target_id")}
    Status: {entry.get("status")}
    
    Details:
    {details}
    """)

def main():
    parser = argparse.ArgumentParser(description="LogLineOS Audit Client")
    parser.add_argument("--url", help="Audit API URL", default=os.environ.get("AUDIT_API_URL", "http://localhost:8083"))
    parser.add_argument("--key", help="API Key", default=os.environ.get("AUDIT_API_KEY", "default_key"))
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # create command
    create_parser = subparsers.add_parser("create", help="Create a new audit entry")
    create_parser.add_argument("operation", help="Operation name")
    create_parser.add_argument("actor", help="Actor name")
    create_parser.add_argument("target_type", help="Target type")
    create_parser.add_argument("target_id", help="Target ID")
    create_parser.add_argument("--status", help="Status", default="success")
    create_parser.add_argument("--details", help="Details JSON", default="{}")
    create_parser.add_argument("--metadata", help="Metadata JSON", default="{}")
    
    # get command
    get_parser = subparsers.add_parser("get", help="Get an audit entry")
    get_parser.add_argument("entry_id", help="Audit entry ID")
    
    # query command
    query_parser = subparsers.add_parser("query", help="Query audit entries")
    query_parser.add_argument("--operations", help="Operations (comma-separated)")
    query_parser.add_argument("--actors", help="Actors (comma-separated)")
    query_parser.add_argument("--target-types", help="Target types (comma-separated)")
    query_parser.add_argument("--target-ids", help="Target IDs (comma-separated)")
    query_parser.add_argument("--statuses", help="Statuses (comma-separated)")
    query_parser.add_argument("--start-time", help="Start time (ISO format)")
    query_parser.add_argument("--end-time", help="End time (ISO format)")
    query_parser.add_argument("--limit", help="Limit", type=int, default=100)
    query_parser.add_argument("--offset", help="Offset", type=int, default=0)
    
    # recent command
    recent_parser = subparsers.add_parser("recent", help="Get recent audit entries")
    recent_parser.add_argument("--hours", help="Hours", type=int, default=24)
    recent_parser.add_argument("--limit", help="Limit", type=int, default=50)
    
    # stats command
    subparsers.add_parser("stats", help="Get audit statistics")
    
    args = parser.parse_args()
    
    client = AuditClient(args.url, args.key)
    
    if args.command == "create":
        try:
            details = json.loads(args.details)
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Error: Details and metadata must be valid JSON")
            sys.exit(1)
        
        result = client.create_entry(
            args.operation, args.actor, args.target_type, args.target_id,
            args.status, details, metadata
        )
        entry = result.get("entry", {})
        print(format_entry(entry))
    
    elif args.command == "get":
        result = client.get_entry(args.entry_id)
        entry = result.get("entry", {})
        print(format_entry(entry))
    
    elif args.command == "query":
        # Parse comma-separated lists
        operations = args.operations.split(",") if args.operations else None
        actors = args.actors.split(",") if args.actors else None
        target_types = args.target_types.split(",") if args.target_types else None
        target_ids = args.target_ids.split(",") if args.target_ids else None
        statuses = args.statuses.split(",") if args.statuses else None
        
        result = client.query_entries(
            operations, actors, target_types, target_ids, statuses,
            args.start_time, args.end_time, args.limit, args.offset
        )
        entries = result.get("entries", [])
        total = result.get("total", 0)
        
        print(f"Found {len(entries)} entries (total: {total})")
        
        table_data = []
        for entry in entries:
            timestamp = entry.get("timestamp", "").replace("T", " ").split(".")[0]
            table_data.append([
                entry.get("id", "")[:8] + "...",
                timestamp,
                entry.get("operation", ""),
                entry.get("actor", ""),
                f"{entry.get('target_type', '')}/{entry.get('target_id', '')}",
                entry.get("status", "")
            ])
        
        headers = ["ID", "Timestamp", "Operation", "Actor", "Target", "Status"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        if total > len(entries):
            print(f"\nShowing {len(entries)} of {total} entries.")
            print(f"Use --offset {args.offset + args.limit} to see the next page.")
    
    elif args.command == "recent":
        result = client.get_recent(args.hours, args.limit)
        entries = result.get("entries", [])
        total = result.get("total", 0)
        
        print(f"Recent audit entries (last {args.hours} hours)")
        
        table_data = []
        for entry in entries:
            timestamp = entry.get("timestamp", "").replace("T", " ").split(".")[0]
            table_data.append([
                entry.get("id", "")[:8] + "...",
                timestamp,
                entry.get("operation", ""),
                entry.get("actor", ""),
                f"{entry.get('target_type', '')}/{entry.get('target_id', '')}",
                entry.get("status", "")
            ])
        
        headers = ["ID", "Timestamp", "Operation", "Actor", "Target", "Status"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    elif args.command == "stats":
        result = client.get_statistics()
        stats = result.get("statistics", {})
        
        print("Audit System Statistics")
        print(f"Total entries: {stats.get('total_entries', 0)}")
        
        print("\nEntries by operation:")
        operations = stats.get("by_operation", {})
        for op, count in operations.items():
            print(f"  {op}: {count}")
        
        print("\nEntries by actor (top 10):")
        actors = stats.get("by_actor", {})
        for actor, count in actors.items():
            print(f"  {actor}: {count}")
        
        print("\nEntries by status:")
        statuses = stats.get("by_status", {})
        for status, count in statuses.items():
            print(f"  {status}: {count}")
        
        print("\nDaily counts (last 7 days):")
        daily = stats.get("daily_counts", {})
        for day, count in daily.items():
            print(f"  {day}: {count}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()