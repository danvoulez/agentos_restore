"""
Enhanced Diamond Farm implementation with advanced mining, trading, and governance features
"""
from typing import Dict, List, Any, Optional, Tuple
import time
import uuid
import hashlib
import json
import os
import threading
import queue
import logging
from datetime import datetime, timedelta
import asyncio
import random
from dataclasses import dataclass, field

from core.diamond_span import DiamondSpan
from core.scarcity_engine import ScarcityEngine
from core.logline_vm import LogLineVM, SpanStatus
from core.grammar_vector import GrammarVector
from core.span_algebra import SpanAlgebra

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/diamond_farm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DiamondFarm")

@dataclass
class MiningStats:
    total_mined: int = 0
    successful_mines: int = 0
    failed_mines: int = 0
    total_energy: float = 0
    highest_energy: float = 0
    start_time: float = field(default_factory=time.time)
    
    def success_rate(self) -> float:
        if self.total_mined == 0:
            return 0
        return self.successful_mines / self.total_mined
    
    def energy_per_second(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0
        return self.total_energy / elapsed
    
    def mining_time(self) -> str:
        elapsed = time.time() - self.start_time
        return str(timedelta(seconds=int(elapsed)))

class EnhancedDiamondFarm:
    """Advanced Diamond Farm with enhanced features"""
    
    def __init__(self, scarcity_engine=None, god_key=None):
        """Initialize the enhanced farm"""
        self.spans: Dict[str, DiamondSpan] = {}
        self.pending_spans = asyncio.Queue()
        self.mined_spans = asyncio.Queue()
        self.scarcity_engine = scarcity_engine or ScarcityEngine()
        self.miners: List[threading.Thread] = []
        self.mining_active = False
        self.vm = LogLineVM()
        self.god_key = god_key or os.getenv("GOD_KEY")
        self.market_price = 10.0  # Base price per unit of energy
        
        # Enhanced features
        self.span_quality_cache: Dict[str, float] = {}
        self.mining_stats = MiningStats()
        self.transaction_history: List[Dict[str, Any]] = []
        self.governance_votes: Dict[str, Dict[str, int]] = {}
        self.auction_house: Dict[str, Dict[str, Any]] = {}
        self.staking_pools: Dict[str, Dict[str, Any]] = {}
        
        # Auto-save configuration
        self.auto_save = True
        self.save_interval = 300  # 5 minutes
        self.last_save_time = time.time()
        
        logger.info("Enhanced Diamond Farm initialized")
    
    async def register_span(self, span: DiamondSpan) -> str:
        """Register a span in the farm with enhanced validation"""
        if span.id in self.spans:
            return span.id
        
        # Enhanced validation
        if not self._validate_span_structure(span):
            logger.warning(f"Invalid span structure rejected: {span.id}")
            raise ValueError("Invalid span structure")
        
        # Check for ethical constraints
        if not self._check_ethical_constraints(span):
            logger.warning(f"Span rejected due to ethical constraints: {span.id}")
            raise ValueError("Span violates ethical constraints")
        
        # Register with the VM
        span_dict = {
            'id': span.id,
            'kind': 'diamond',
            'who': span.metadata.get('creator', 'unknown'),
            'what': 'span_creation',
            'why': 'diamond_farming',
            'payload': {
                'content': span.content,
                'parent_ids': span.parent_ids,
                'energy': span.energy
            }
        }
        
        self.vm.register_span(span_dict)
        self.spans[span.id] = span
        
        # Calculate and cache quality score
        self.span_quality_cache[span.id] = self._calculate_quality_score(span)
        
        logger.info(f"Span registered: {span.id} with energy {span.energy:.2f}")
        return span.id
    
    async def submit_for_mining(self, content, parent_ids=None, metadata=None):
        """Submit content for mining into a Diamond Span with enhanced metadata"""
        if metadata is None:
            metadata = {}
        
        # Add enhanced metadata
        metadata.update({
            "submitted_at": datetime.utcnow().isoformat(),
            "mining_priority": metadata.get("mining_priority", 1.0),
            "governance_level": metadata.get("governance_level", 0)
        })
        
        # Set creator to god_key if available
        if self.god_key and metadata.get("creator") == self.god_key:
            # God mode enabled - add special flags
            metadata["is_exempt"] = True
            metadata["decay_rate"] = 0.0
            metadata["governance_level"] = 10
        
        # Create grammar vector for content
        if isinstance(content, str):
            content = {"text": content}
        
        if "text" in content:
            grammar = GrammarVector.from_natural_language(content["text"])
            content["grammar_vector"] = grammar.to_vector().tolist()
        
        # Create preliminary span
        span = DiamondSpan(
            id=None,  # Will be set during mining
            parent_ids=parent_ids or [],
            content=content,
            metadata=metadata
        )
        
        # Add to mining queue with priority
        priority = metadata.get("mining_priority", 1.0)
        await self.pending_spans.put((priority, span))
        
        logger.info(f"Span submitted for mining with priority {priority}")
        return {"status": "submitted", "queue_position": self.pending_spans.qsize()}
    
    async def start_mining(self, num_miners=4):
        """Start asynchronous mining workers"""
        if self.mining_active:
            return {"status": "already_running", "miners": len(self.miners)}
        
        self.mining_active = True
        self.miners = []
        
        for i in range(num_miners):
            # Create task for each miner
            task = asyncio.create_task(self._mining_worker(i))
            self.miners.append(task)
        
        # Start autosave task if enabled
        if self.auto_save:
            asyncio.create_task(self._autosave_worker())
        
        logger.info(f"Started {num_miners} mining tasks")
        return {"status": "started", "miners": num_miners}
    
    async def stop_mining(self):
        """Stop mining tasks gracefully"""
        if not self.mining_active:
            return {"status": "not_running"}
        
        self.mining_active = False
        
        # Wait for all miners to complete
        for i, task in enumerate(self.miners):
            try:
                await asyncio.wait_for(task, timeout=5.0)
                logger.info(f"Miner {i} stopped gracefully")
            except asyncio.TimeoutError:
                logger.warning(f"Miner {i} did not stop in time")
        
        self.miners = []
        
        # Save data before shutdown
        await self._save_data()
        
        logger.info("Mining stopped")
        return {"status": "stopped"}
    
    async def _mining_worker(self, worker_id):
        """Asynchronous worker for mining spans"""
        logger.info(f"Miner {worker_id} started")
        
        while self.mining_active:
            try:
                # Get a pending span with highest priority
                try:
                    priority, span = await asyncio.wait_for(self.pending_spans.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process the span
                mined_span = await self._mine_span(span, worker_id)
                
                self.mining_stats.total_mined += 1
                
                if mined_span:
                    # Register the mined span
                    await self.register_span(mined_span)
                    await self.mined_spans.put(mined_span)
                    
                    self.mining_stats.successful_mines += 1
                    self.mining_stats.total_energy += mined_span.energy
                    self.mining_stats.highest_energy = max(self.mining_stats.highest_energy, mined_span.energy)
                    
                    logger.info(f"Miner {worker_id} mined span {mined_span.id[:8]} with energy {mined_span.energy:.2f}")
                else:
                    self.mining_stats.failed_mines += 1
                    logger.warning(f"Miner {worker_id} failed to mine span")
                
                self.pending_spans.task_done()
                
            except Exception as e:
                logger.error(f"Mining error in worker {worker_id}: {str(e)}")
                await asyncio.sleep(1.0)  # Backoff on error
        
        logger.info(f"Miner {worker_id} stopped")
    
    async def _mine_span(self, span, worker_id) -> Optional[DiamondSpan]:
        """Mine a single span with enhanced proof-of-value algorithm"""
        start_time = time.time()
        
        # Set ID if not already set
        if not span.id:
            span.id = str(uuid.uuid4())
        
        # Calculate base energy with more factors
        base_energy = self._calculate_base_energy(span)
        
        # Special case: God mode spans
        if span.metadata.get("creator") == self.god_key:
            # God spans automatically succeed with boosted energy
            span.energy = base_energy * 10.0
            span.metadata["mined_at"] = datetime.utcnow().isoformat()
            span.metadata["mining_time"] = 0.0
            
            logger.info(f"God mode span created with energy {span.energy:.2f}")
            return span
        
        # Regular span mining process
        difficulty = self.scarcity_engine._calculate_difficulty()
        mining_start = time.time()
        mining_effort = 0
        max_attempts = 200  # Increased max attempts
        
        # Mining algorithm with adaptive difficulty
        while mining_effort < max_attempts and self.mining_active:
            # Enhanced proof of work
            nonce = os.urandom(8)  # Increased randomness
            content_hash = hashlib.sha256(json.dumps(span.content).encode()).hexdigest()
            hash_input = f"{span.id}:{content_hash}:{nonce.hex()}:{mining_effort}".encode()
            hash_result = hashlib.sha256(hash_input).hexdigest()
            
            # Dynamic difficulty targeting
            difficulty_target = (2**64) / (difficulty * (1 + mining_effort/50))
            if int(hash_result[:16], 16) < difficulty_target:
                # Mining successful
                mining_time = time.time() - mining_start
                mining_energy = base_energy * (1 + (mining_effort * 0.05))
                
                # Add bonuses for longer mining time (up to a cap)
                time_bonus = min(2.0, mining_time / 10)
                mining_energy *= (1 + time_bonus)
                
                span.energy = mining_energy
                span.metadata["mined_by"] = f"worker_{worker_id}"
                span.metadata["mined_at"] = datetime.utcnow().isoformat()
                span.metadata["mining_time"] = mining_time
                span.metadata["mining_effort"] = mining_effort
                span.metadata["hash"] = hash_result[:16]
                
                # Add quality score
                quality_score = self._calculate_quality_score(span)
                span.metadata["quality_score"] = quality_score
                
                # Apply energy bonus for quality
                span.energy *= (1 + quality_score/10)
                
                # Try to mint with scarcity engine
                if await self._try_mint_span(span):
                    return span
                else:
                    logger.warning(f"Span rejected by scarcity engine: {span.id}")
                    return None
            
            mining_effort += 1
            
            # Async yield to prevent blocking
            if mining_effort % 10 == 0:
                await asyncio.sleep(0.001)
        
        logger.warning(f"Mining failed after {mining_effort} attempts: {span.id}")
        return None
    
    async def _try_mint_span(self, span) -> bool:
        """Try to mint a span with the scarcity engine, with retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.scarcity_engine.mint_span(span):
                    return True
                else:
                    # If rejected due to scarcity, wait and retry
                    await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                logger.error(f"Minting error (attempt {attempt+1}): {str(e)}")
                await asyncio.sleep(1.0)
        
        return False
    
    def _calculate_base_energy(self, span) -> float:
        """Calculate base energy with enhanced algorithm"""
        # Start with a base value
        base = 10.0
        
        # Factor 1: Content complexity
        if isinstance(span.content, dict) and "text" in span.content:
            # Text length factor (longer = more energy, but with diminishing returns)
            text = span.content["text"]
            length_factor = min(3.0, len(text) / 500)
            
            # Language complexity factor
            unique_words = len(set(text.lower().split()))
            vocabulary_factor = min(2.0, unique_words / 100)
            
            # Combine text factors
            content_factor = length_factor * vocabulary_factor
        else:
            # For non-text content, use JSON complexity
            content_size = len(json.dumps(span.content))
            content_factor = min(2.5, content_size / 1000)
        
        # Factor 2: Parent relationship
        parent_count = len(span.parent_ids)
        parent_factor = 1.0 + (parent_count * 0.15)
        
        # Factor 3: Metadata richness
        metadata_factor = min(1.5, len(json.dumps(span.metadata)) / 200)
        
        # Combine all factors
        energy = base * content_factor * parent_factor * metadata_factor
        
        # Cap at reasonable values
        return min(100.0, energy)
    
    def _calculate_quality_score(self, span) -> float:
        """Calculate quality score for a span"""
        # Base score
        score = 5.0
        
        # Factor 1: Content quality
        if isinstance(span.content, dict) and "text" in span.content:
            text = span.content["text"]
            
            # Length factor (with diminishing returns)
            length_score = min(2.0, len(text) / 1000)
            
            # Vocabulary richness
            words = text.lower().split()
            unique_ratio = len(set(words)) / max(1, len(words))
            vocab_score = unique_ratio * 3.0
            
            # Combine text scores
            content_score = length_score + vocab_score
        else:
            # For non-text content, use structure complexity
            content_score = min(3.0, len(json.dumps(span.content)) / 500)
        
        # Factor 2: Causal richness
        parent_score = min(2.0, len(span.parent_ids) * 0.5)
        
        # Factor 3: Energy level
        energy_score = min(3.0, span.energy / 30)
        
        # Combine all scores
        final_score = score + content_score + parent_score + energy_score
        
        # Cap at 10
        return min(10.0, final_score)
    
    def _validate_span_structure(self, span) -> bool:
        """Validate the structure of a span"""
        # Basic validation
        if not span.id or not isinstance(span.id, str):
            return False
        
        if not hasattr(span, 'content') or span.content is None:
            return False
        
        if not hasattr(span, 'parent_ids') or not isinstance(span.parent_ids, list):
            return False
        
        # Validate parent references
        for parent_id in span.parent_ids:
            if parent_id not in self.spans and parent_id != "genesis":
                return False
        
        return True
    
    def _check_ethical_constraints(self, span) -> bool:
        """Check if a span meets ethical constraints"""
        # Get content as string for analysis
        if isinstance(span.content, dict) and "text" in span.content:
            content = span.content["text"].lower()
        else:
            content = json.dumps(span.content).lower()
        
        # List of harmful terms to check for
        harmful_terms = [
            "weapon", "bomb", "kill", "harm", "attack", "destroy", "genocide",
            "terror", "torture", "violent", "abuse", "hatred", "racist"
        ]
        
        # Check for harmful content
        for term in harmful_terms:
            if term in content:
                return False
        
        return True
    
    async def get_span(self, span_id):
        """Get a specific span by ID with enhanced metadata"""
        span = self.spans.get(span_id)
        if not span:
            return None
        
        # Add dynamic metadata
        span.metadata["current_value"] = self.calculate_span_value(span)
        span.metadata["quality_score"] = self.span_quality_cache.get(
            span_id, self._calculate_quality_score(span)
        )
        
        # Add market data if in auction
        if span_id in self.auction_house:
            span.metadata["auction"] = self.auction_house[span_id]
        
        return span
    
    async def get_all_spans(self, page=1, page_size=50, sort_by="energy", descending=True):
        """Get all registered spans with pagination and sorting"""
        # Get all spans
        all_spans = list(self.spans.values())
        
        # Sort spans
        if sort_by == "energy":
            all_spans.sort(key=lambda s: s.energy, reverse=descending)
        elif sort_by == "created_at":
            all_spans.sort(key=lambda s: s.created_at, reverse=descending)
        elif sort_by == "quality":
            all_spans.sort(
                key=lambda s: self.span_quality_cache.get(s.id, self._calculate_quality_score(s)),
                reverse=descending
            )
        elif sort_by == "value":
            all_spans.sort(key=lambda s: self.calculate_span_value(s), reverse=descending)
        
        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_spans = all_spans[start_idx:end_idx]
        
        # Add pagination metadata
        result = {
            "spans": page_spans,
            "pagination": {
                "total": len(all_spans),
                "page": page,
                "page_size": page_size,
                "total_pages": (len(all_spans) + page_size - 1) // page_size
            }
        }
        
        return result
    
    def calculate_span_value(self, span) -> float:
        """Calculate the market value of a span with enhanced economics"""
        if not span:
            return 0
        
        # Base value from energy
        base_value = span.energy * self.market_price
        
        # Scarcity factor
        circulating = self.scarcity_engine.minted - self.scarcity_engine.burned
        scarcity_factor = max(1.0, (self.scarcity_engine.total_supply / max(1, circulating)) * 0.1)
        
        # Quality factor
        quality_score = self.span_quality_cache.get(span.id, self._calculate_quality_score(span))
        quality_factor = 0.5 + (quality_score / 10)
        
        # Age factor (newer spans worth more)
        age_days = (datetime.utcnow() - span.created_at).total_seconds() / (3600 * 24)
        age_factor = max(0.5, min(1.5, 2.0 - (age_days / 30)))  # Decay over 30 days
        
        # Network factor (spans with more connections worth more)
        parent_count = len(span.parent_ids)
        network_factor = 1.0 + (parent_count * 0.05)
        
        # Calculate final value
        value = base_value * scarcity_factor * quality_factor * age_factor * network_factor
        
        # God spans have special value
        if span.is_exempt():
            value *= 10.0
        
        return value
    
    async def get_market_stats(self):
        """Get enhanced market statistics"""
        spans = list(self.spans.values())
        total_spans = len(spans)
        
        # Calculate basic stats
        total_energy = sum(span.energy for span in spans)
        total_value = sum(self.calculate_span_value(span) for span in spans)
        
        # Calculate mining stats
        mining_stats = {
            "total_mined": self.mining_stats.total_mined,
            "successful_mines": self.mining_stats.successful_mines,
            "failed_mines": self.mining_stats.failed_mines,
            "success_rate": self.mining_stats.success_rate(),
            "energy_per_second": self.mining_stats.energy_per_second(),
            "highest_energy": self.mining_stats.highest_energy,
            "mining_time": self.mining_stats.mining_time()
        }
        
        # Calculate quality distribution
        if total_spans > 0:
            quality_scores = [self.span_quality_cache.get(span.id, self._calculate_quality_score(span)) for span in spans]
            quality_stats = {
                "average": sum(quality_scores) / total_spans,
                "median": sorted(quality_scores)[total_spans // 2],
                "highest": max(quality_scores),
                "lowest": min(quality_scores),
                "distribution": {
                    "0-2": len([s for s in quality_scores if s <= 2]),
                    "2-4": len([s for s in quality_scores if 2 < s <= 4]),
                    "4-6": len([s for s in quality_scores if 4 < s <= 6]),
                    "6-8": len([s for s in quality_scores if 6 < s <= 8]),
                    "8-10": len([s for s in quality_scores if 8 < s <= 10])
                }
            }
        else:
            quality_stats = {"average": 0, "median": 0, "highest": 0, "lowest": 0, "distribution": {}}
        
        # Get market dynamics
        market_dynamics = {
            "price": self.market_price,
            "24h_change": self._calculate_24h_price_change(),
            "volume": self._calculate_trading_volume(),
            "active_auctions": len(self.auction_house)
        }
        
        # Calculate network stats
        connections = sum(len(span.parent_ids) for span in spans)
        avg_connections = connections / total_spans if total_spans > 0 else 0
        network_stats = {
            "total_connections": connections,
            "avg_connections": avg_connections,
            "density": connections / (total_spans * (total_spans - 1)) if total_spans > 1 else 0
        }
        
        # Return comprehensive stats
        return {
            "total_spans": total_spans,
            "total_energy": total_energy,
            "total_value": total_value,
            "minted": self.scarcity_engine.minted,
            "burned": self.scarcity_engine.burned,
            "circulating": self.scarcity_engine.minted - self.scarcity_engine.burned,
            "mining": mining_stats,
            "quality": quality_stats,
            "market": market_dynamics,
            "network": network_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def create_auction(self, span_id: str, min_bid: float, duration_hours: float = 24.0):
        """Create an auction for a span"""
        if span_id not in self.spans:
            logger.warning(f"Cannot auction non-existent span: {span_id}")
            return {"status": "error", "message": "Span not found"}
        
        span = self.spans[span_id]
        
        # Check ownership (simplified - in production would check actual ownership)
        # For now, anyone can auction any span
        
        # Create auction entry
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        auction = {
            "span_id": span_id,
            "min_bid": min_bid,
            "current_bid": 0.0,
            "current_bidder": None,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": end_time.isoformat(),
            "bids": [],
            "status": "active"
        }
        
        self.auction_house[span_id] = auction
        
        logger.info(f"Auction created for span {span_id} with min bid {min_bid}")
        return {"status": "success", "auction": auction}
    
    async def place_bid(self, span_id: str, bidder: str, amount: float):
        """Place a bid on an auction"""
        if span_id not in self.auction_house:
            return {"status": "error", "message": "Auction not found"}
        
        auction = self.auction_house[span_id]
        
        # Check if auction is active
        end_time = datetime.fromisoformat(auction["end_time"])
        if datetime.utcnow() > end_time:
            auction["status"] = "ended"
            return {"status": "error", "message": "Auction has ended"}
        
        # Check if bid is high enough
        current_bid = auction["current_bid"]
        min_bid = auction["min_bid"]
        required_bid = max(min_bid, current_bid * 1.05)  # 5% increment
        
        if amount < required_bid:
            return {
                "status": "error", 
                "message": f"Bid too low, minimum bid is {required_bid}"
            }
        
        # Place bid
        auction["current_bid"] = amount
        auction["current_bidder"] = bidder
        auction["bids"].append({
            "bidder": bidder,
            "amount": amount,
            "time": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Bid placed on span {span_id}: {amount} by {bidder}")
        return {"status": "success", "auction": auction}
    
    async def finalize_auction(self, span_id: str):
        """Finalize an auction"""
        if span_id not in self.auction_house:
            return {"status": "error", "message": "Auction not found"}
        
        auction = self.auction_house[span_id]
        
        # Check if auction should be finalized
        end_time = datetime.fromisoformat(auction["end_time"])
        if datetime.utcnow() < end_time and auction["status"] != "force_end":
            return {"status": "error", "message": "Auction still in progress"}
        
        # Finalize the auction
        if auction["current_bidder"]:
            # Auction successful
            auction["status"] = "completed"
            
            # Record transaction
            self.transaction_history.append({
                "type": "auction_sale",
                "span_id": span_id,
                "seller": "farm",  # In production, would be actual seller
                "buyer": auction["current_bidder"],
                "price": auction["current_bid"],
                "time": datetime.utcnow().isoformat()
            })
            
            # Update market price based on sale
            self._update_market_price(span_id, auction["current_bid"])
            
            logger.info(f"Auction completed for span {span_id}: sold to {auction['current_bidder']} for {auction['current_bid']}")
            
        else:
            # No bids
            auction["status"] = "ended_no_bids"
            logger.info(f"Auction ended with no bids for span {span_id}")
        
        # Keep auction in history but mark as finalized
        self.auction_house[span_id] = auction
        
        return {"status": "success", "auction": auction}
    
    async def create_stake(self, span_id: str, stake_amount: float, staker: str, duration_days: int = 30):
        """Stake a span for passive income"""
        if span_id not in self.spans:
            return {"status": "error", "message": "Span not found"}
        
        span = self.spans[span_id]
        
        # Create staking entry
        end_time = datetime.utcnow() + timedelta(days=duration_days)
        stake = {
            "span_id": span_id,
            "staker": staker,
            "amount": stake_amount,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": end_time.isoformat(),
            "duration_days": duration_days,
            "rewards": 0.0,
            "status": "active"
        }
        
        pool_id = f"pool_{span.energy:.0f}"
        if pool_id not in self.staking_pools:
            self.staking_pools[pool_id] = {
                "total_staked": 0.0,
                "stakes": {}
            }
        
        self.staking_pools[pool_id]["stakes"][span_id] = stake
        self.staking_pools[pool_id]["total_staked"] += stake_amount
        
        logger.info(f"Stake created for span {span_id} with amount {stake_amount}")
        return {"status": "success", "stake": stake}
    
    async def collect_rewards(self, span_id: str, staker: str):
        """Collect rewards from a staked span"""
        # Find the stake
        stake = None
        pool_id = None
        
        for p_id, pool in self.staking_pools.items():
            if span_id in pool["stakes"]:
                stake = pool["stakes"][span_id]
                pool_id = p_id
                break
        
        if not stake:
            return {"status": "error", "message": "Stake not found"}
        
        if stake["staker"] != staker:
            return {"status": "error", "message": "Not your stake"}
        
        # Calculate rewards
        days_staked = (datetime.utcnow() - datetime.fromisoformat(stake["start_time"])).days
        
        # Base APY of 10%
        base_apy = 0.10
        
        # Calculate rewards
        rewards = stake["amount"] * (base_apy / 365 * days_staked)
        stake["rewards"] += rewards
        
        # Reset reward timer
        stake["last_collection"] = datetime.utcnow().isoformat()
        
        logger.info(f"Rewards collected for span {span_id}: {rewards}")
        return {
            "status": "success",
            "collected": rewards,
            "total_rewards": stake["rewards"],
            "stake": stake
        }
    
    async def get_governance_proposals(self):
        """Get all active governance proposals"""
        return self.governance_votes
    
    async def create_governance_proposal(self, title: str, description: str, proposer: str, action: Dict[str, Any]):
        """Create a governance proposal"""
        proposal_id = f"gov_{len(self.governance_votes) + 1}_{int(time.time())}"
        
        proposal = {
            "id": proposal_id,
            "title": title,
            "description": description,
            "proposer": proposer,
            "created_at": datetime.utcnow().isoformat(),
            "end_time": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            "action": action,
            "votes_for": 0,
            "votes_against": 0,
            