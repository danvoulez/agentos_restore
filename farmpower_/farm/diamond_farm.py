"""
Diamond Farm implementation for mining, storing, and trading spans
"""
from typing import Dict, List, Any, Optional
import time
import uuid
import hashlib
import json
import os
import threading
import queue
from core.diamond_span import DiamondSpan
from core.scarcity_engine import ScarcityEngine

class DiamondFarm:
    def __init__(self, scarcity_engine=None, god_key=None):
        self.spans: Dict[str, DiamondSpan] = {}
        self.pending_spans = queue.Queue()
        self.mined_spans = queue.Queue()
        self.scarcity_engine = scarcity_engine or ScarcityEngine()
        self.miners: List[threading.Thread] = []
        self.mining_active = False
        self.god_key = god_key or os.getenv("GOD_KEY")
        self.market_price = 10.0  # Base price per unit of energy
        
    def register_span(self, span: DiamondSpan) -> str:
        """Register a span in the farm"""
        if span.id in self.spans:
            return span.id
            
        self.spans[span.id] = span
        return span.id
        
    def submit_for_mining(self, content, parent_ids=None, metadata=None):
        """Submit content for mining into a Diamond Span"""
        if metadata is None:
            metadata = {}
            
        # Set creator to god_key if available
        if self.god_key:
            metadata["creator"] = self.god_key
            
        span = DiamondSpan(
            id=None,  # Will be set during mining
            parent_ids=parent_ids or [],
            content=content,
            metadata=metadata
        )
        
        self.pending_spans.put(span)
        return "submitted"
        
    def start_mining(self, num_miners=4):
        """Start mining threads"""
        if self.mining_active:
            return
            
        self.mining_active = True
        
        for i in range(num_miners):
            miner = threading.Thread(target=self._mining_worker, args=(i,))
            miner.daemon = True
            miner.start()
            self.miners.append(miner)
            
        return f"Started {num_miners} mining threads"
        
    def stop_mining(self):
        """Stop mining threads"""
        self.mining_active = False
        for miner in self.miners:
            if miner.is_alive():
                miner.join(1.0)  # Give each thread 1s to finish
                
        self.miners = []
        return "Mining stopped"
        
    def _mining_worker(self, worker_id):
        """Worker thread for mining spans"""
        print(f"Miner {worker_id} started")
        
        while self.mining_active:
            try:
                # Get a pending span
                span = self.pending_spans.get(timeout=1.0)
                
                # Process the span
                mined_span = self._mine_span(span, worker_id)
                
                if mined_span:
                    # Register the mined span
                    self.register_span(mined_span)
                    self.mined_spans.put(mined_span)
                    print(f"Miner {worker_id} mined span {mined_span.id} with energy {mined_span.energy}")
                    
                self.pending_spans.task_done()
                
            except queue.Empty:
                time.sleep(0.1)  # Sleep briefly when no spans to process
            except Exception as e:
                print(f"Mining error: {str(e)}")
                
        print(f"Miner {worker_id} stopped")
        
    def _mine_span(self, span, worker_id) -> Optional[DiamondSpan]:
        """Mine a single span, applying scarcity rules"""
        start_time = time.time()
        
        # Set ID if not already set
        if not span.id:
            span.id = str(uuid.uuid4())
            
        # Calculate base energy
        base_energy = 10.0
        
        # Add complexity factor
        content_complexity = len(json.dumps(span.content)) / 100
        base_energy *= (1 + content_complexity * 0.2)
        
        # Add mining effort factor
        mining_effort = 0
        difficulty = self.scarcity_engine._calculate_difficulty()
        
        # Check if this is a god_key span
        is_god = span.metadata.get("creator") == self.god_key
        
        if is_god:
            # God spans automatically succeed with boosted energy
            span.energy = base_energy * 5.0
            return span
            
        # Regular span mining process
        while mining_effort < 100:  # Cap mining attempts
            # Simulated proof of work
            nonce = os.urandom(4)
            hash_input = f"{span.id}:{json.dumps(span.content)}:{nonce.hex()}".encode()
            hash_result = hashlib.sha256(hash_input).hexdigest()
            
            # Check if hash meets difficulty
            if int(hash_result[:8], 16) < (2**32) / difficulty:
                # Mining successful
                mining_energy = base_energy * (1 + mining_effort * 0.1)
                span.energy = mining_energy
                span.metadata["mined_by"] = f"worker_{worker_id}"
                span.metadata["mining_time"] = time.time() - start_time
                
                if self.scarcity_engine.mint_span(span):
                    return span
                else:
                    print(f"Span rejected by scarcity engine: {span.id}")
                    return None
                    
            mining_effort += 1
            
        print(f"Mining failed after {mining_effort} attempts: {span.id}")
        return None
        
    def get_span(self, span_id):
        """Get a specific span by ID"""
        return self.spans.get(span_id)
        
    def get_all_spans(self):
        """Get all registered spans"""
        return list(self.spans.values())
        
    def calculate_span_value(self, span):
        """Calculate the market value of a span"""
        if not span:
            return 0
            
        # Base value from energy
        base_value = span.energy * self.market_price
        
        # Adjust for scarcity
        circulating = self.scarcity_engine.minted - self.scarcity_engine.burned
        scarcity_factor = max(1.0, (self.scarcity_engine.total_supply / max(1, circulating)) * 0.1)
        
        # Adjust for age (newer spans worth more)
        age_days = (time.time() - span.created_at.timestamp()) / (3600 * 24)
        age_factor = max(0.5, min(1.5, 2.0 - (age_days / 30)))  # Decay over 30 days
        
        # Calculate final value
        value = base_value * scarcity_factor * age_factor
        
        # God spans have special value
        if span.is_exempt():
            value *= 10.0
            
        return value
        
    def get_market_stats(self):
        """Get market statistics"""
        total_spans = len(self.spans)
        total_energy = sum(span.energy for span in self.spans.values())
        total_value = sum(self.calculate_span_value(span) for span in self.spans.values())
        
        return {
            "total_spans": total_spans,
            "total_energy": total_energy,
            "total_value": total_value,
            "market_price": self.market_price,
            "minted": self.scarcity_engine.minted,
            "burned": self.scarcity_engine.burned,
            "circulating": self.scarcity_engine.minted - self.scarcity_engine.burned
        }