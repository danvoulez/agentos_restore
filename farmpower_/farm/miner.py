"""
Diamond Farm Miner for LogLineOS/DiamondSpan
Advanced mining operations for diamond spans
Created: 2025-07-19 05:42:25 UTC
User: danvoulez
"""
import os
import json
import time
import uuid
import hashlib
import logging
import asyncio
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/miner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DiamondMiner")

@dataclass
class MinerConfig:
    """Configuration for a diamond miner"""
    threads: int = 4
    batch_size: int = 10
    target_difficulty: float = 1.0
    difficulty_adjustment_interval: int = 100
    target_mining_rate: int = 10  # spans per minute
    energy_factor: float = 1.0
    reward_schema: str = "logarithmic"
    min_quality_threshold: float = 3.0
    adaptive_difficulty: bool = True
    verify_parents: bool = True
    mutation_rate: float = 0.05
    god_key: Optional[str] = None
    checkpoint_interval: int = 60  # seconds

@dataclass
class MiningStats:
    """Statistics for mining operations"""
    total_attempts: int = 0
    successful_mines: int = 0
    failed_mines: int = 0
    total_energy: float = 0.0
    highest_energy: float = 0.0
    average_mining_time: float = 0.0
    current_difficulty: float = 1.0
    start_time: float = field(default_factory=time.time)
    spans_per_minute: float = 0.0
    total_rewards: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    def calculate_metrics(self):
        """Calculate derived metrics"""
        mining_time = time.time() - self.start_time
        minutes = mining_time / 60
        
        if minutes > 0:
            self.spans_per_minute = self.successful_mines / minutes
        
        if self.total_attempts > 0:
            self.success_rate = self.successful_mines / self.total_attempts
        else:
            self.success_rate = 0.0
            
        self.last_updated = time.time()

class DiamondMiner:
    """
    Advanced Diamond Span Miner
    """
    
    def __init__(self, config: MinerConfig = None, farm=None):
        """Initialize the diamond miner"""
        self.config = config or MinerConfig()
        self.farm = farm
        self.stats = MiningStats(current_difficulty=self.config.target_difficulty)
        self.mining_active = False
        self.miners: List[threading.Thread] = []
        
        # Work queues
        self.pending_spans = queue.PriorityQueue()  # (priority, span)
        self.mined_spans = queue.Queue()
        
        # Threading resources
        self.executor = ThreadPoolExecutor(max_workers=self.config.threads)
        self.lock = threading.RLock()
        
        # Mining algorithms
        self.difficulty_history: List[float] = []
        self.mining_time_history: List[float] = []
        self.proof_cache: Dict[str, str] = {}  # Cache for proof of work calculations
        
        # Load any saved state
        self._load_state()
        
        logger.info(f"Diamond Miner initialized with {self.config.threads} threads, "
                   f"difficulty: {self.stats.current_difficulty:.2f}")
    
    def start_mining(self) -> Dict[str, Any]:
        """Start mining threads"""
        if self.mining_active:
            return {"status": "already_active", "threads": len(self.miners)}
        
        self.mining_active = True
        self.stats.start_time = time.time()
        
        # Start mining threads
        self.miners = []
        for i in range(self.config.threads):
            miner = threading.Thread(
                target=self._mining_worker, 
                args=(i,),
                name=f"miner-{i}"
            )
            miner.daemon = True
            miner.start()
            self.miners.append(miner)
            
        # Start auxiliary threads for difficulty adjustment and checkpointing
        if self.config.adaptive_difficulty:
            self._start_difficulty_adjuster()
            
        self._start_checkpointing_thread()
        
        logger.info(f"Started {self.config.threads} mining threads")
        return {"status": "started", "threads": len(self.miners)}
    
    def stop_mining(self) -> Dict[str, Any]:
        """Stop all mining threads gracefully"""
        if not self.mining_active:
            return {"status": "not_active"}
        
        logger.info("Stopping mining operations...")
        self.mining_active = False
        
        # Wait for threads to finish
        for i, miner in enumerate(self.miners):
            logger.info(f"Waiting for miner {i} to stop...")
            miner.join(timeout=2.0)
            
        # Save state before shutting down
        self._save_state()
        
        self.miners = []
        logger.info("Mining operations stopped")
        
        return {"status": "stopped"}
    
    def submit_for_mining(self, 
                        content: Union[str, Dict[str, Any]], 
                        parent_ids: List[str] = None,
                        metadata: Dict[str, Any] = None,
                        priority: float = 1.0) -> Dict[str, Any]:
        """
        Submit content for mining into a Diamond Span
        
        Args:
            content: Content for the span (text or dict)
            parent_ids: Optional parent span IDs
            metadata: Optional metadata
            priority: Mining priority (lower = higher priority)
            
        Returns:
            Status dictionary
        """
        # Normalize content
        if isinstance(content, str):
            content = {"text": content}
        
        # Initialize metadata
        if metadata is None:
            metadata = {}
            
        # Add submission metadata
        metadata.update({
            "submitted_at": datetime.utcnow().isoformat(),
            "mining_priority": priority
        })
        
        # Check if using god mode
        is_god_mode = metadata.get("creator") == self.config.god_key and self.config.god_key is not None
        
        if is_god_mode:
            metadata["is_exempt"] = True
            metadata["energy_factor"] = 5.0
            
        # Create preliminary span
        span = {
            "id": str(uuid.uuid4()),
            "content": content,
            "parent_ids": parent_ids or [],
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Add to mining queue with priority
        self.pending_spans.put((priority, span))
        queue_size = self.pending_spans.qsize()
        
        logger.info(f"Span submitted for mining with priority {priority} (queue size: {queue_size})")
        
        return {
            "status": "submitted",
            "id": span["id"],
            "queue_position": queue_size,
            "estimated_time": self._estimate_mining_time(priority)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current mining statistics"""
        with self.lock:
            # Update calculations
            self.stats.calculate_metrics()
            
            # Create stats dictionary
            stats_dict = {
                "total_attempts": self.stats.total_attempts,
                "successful_mines": self.stats.successful_mines,
                "failed_mines": self.stats.failed_mines,
                "success_rate": self.stats.success_rate,
                "total_energy": self.stats.total_energy,
                "highest_energy": self.stats.highest_energy,
                "average_mining_time": self.stats.average_mining_time,
                "current_difficulty": self.stats.current_difficulty,
                "spans_per_minute": self.stats.spans_per_minute,
                "uptime_seconds": time.time() - self.stats.start_time,
                "pending_spans": self.pending_spans.qsize(),
                "mined_spans": self.mined_spans.qsize(),
                "total_rewards": self.stats.total_rewards,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return stats_dict
    
    def _mining_worker(self, worker_id: int) -> None:
        """Worker thread for mining spans"""
        logger.info(f"Miner {worker_id} started")
        
        while self.mining_active:
            try:
                # Get next span to mine with priority
                try:
                    priority, span = self.pending_spans.get(timeout=1.0)
                except queue.Empty:
                    # No spans to mine, sleep briefly
                    time.sleep(0.1)
                    continue
                
                # Mine the span
                logger.debug(f"Miner {worker_id} mining span {span['id']}")
                result = self._mine_span(span, worker_id)
                
                # Update statistics
                with self.lock:
                    self.stats.total_attempts += 1
                    
                    if result["status"] == "success":
                        self.stats.successful_mines += 1
                        self.stats.total_energy += result["span"]["energy"]
                        
                        if result["span"]["energy"] > self.stats.highest_energy:
                            self.stats.highest_energy = result["span"]["energy"]
                        
                        # Add to mining time history for average calculation
                        mining_time = result["mining_time"]
                        self.mining_time_history.append(mining_time)
                        if len(self.mining_time_history) > 100:
                            self.mining_time_history.pop(0)
                        
                        # Calculate average mining time
                        self.stats.average_mining_time = sum(self.mining_time_history) / len(self.mining_time_history)
                        
                        # Add mining rewards
                        self.stats.total_rewards += result["reward"]
                        
                        # Put in mined spans queue
                        self.mined_spans.put(result["span"])
                        
                        logger.info(f"Miner {worker_id} successfully mined span {span['id']} "
                                   f"with energy {result['span']['energy']:.2f} in {mining_time:.2f}s")
                    else:
                        self.stats.failed_mines += 1
                        logger.warning(f"Miner {worker_id} failed to mine span {span['id']}: {result['reason']}")
                
                # Mark task as done
                self.pending_spans.task_done()
                
            except Exception as e:
                logger.error(f"Error in miner {worker_id}: {str(e)}", exc_info=True)
                time.sleep(1.0)  # Sleep on error to prevent tight loops
        
        logger.info(f"Miner {worker_id} stopped")
    
    def _mine_span(self, span: Dict[str, Any], worker_id: int) -> Dict[str, Any]:
        """Mine a single span"""
        start_time = time.time()
        
        # Check if this is a god key span
        is_god_mode = (span["metadata"].get("creator") == self.config.god_key and 
                     self.config.god_key is not None)
        
        # Check parent spans if required
        if self.config.verify_parents and span["parent_ids"] and not is_god_mode:
            if not self._verify_parent_spans(span["parent_ids"]):
                return {
                    "status": "failed",
                    "reason": "Invalid parent spans",
                    "mining_time": time.time() - start_time
                }
        
        # Calculate base energy
        base_energy = self._calculate_base_energy(span)
        
        # Handle god mode spans (automatic success)
        if is_god_mode:
            energy_factor = span["metadata"].get("energy_factor", 5.0)
            
            # Create successfully mined span
            mined_span = {
                **span,
                "energy": base_energy * energy_factor,
                "mined_at": datetime.utcnow().isoformat(),
                "mined_by": f"worker_{worker_id}",
                "mining_time": 0.0,
                "signature": self._generate_signature(span["id"], 0)
            }
            
            # Calculate reward (higher for god spans)
            reward = self._calculate_reward(mined_span["energy"], 0.0)
            
            return {
                "status": "success",
                "span": mined_span,
                "mining_time": 0.0,
                "reward": reward * 2
            }
        
        # Standard mining process
        current_difficulty = self.stats.current_difficulty
        max_attempts = 500  # Cap maximum attempts
        
        mining_start = time.time()
        nonce = 0
        
        # Mining loop with proof of work
        while nonce < max_attempts and self.mining_active:
            # Generate proof hash
            proof = self._generate_proof(span["id"], span["content"], nonce)
            
            # Check if meets difficulty target
            if self._check_proof_meets_difficulty(proof, current_difficulty):
                # Mining successful!
                mining_time = time.time() - mining_start
                
                # Calculate mining boost based on attempts
                mining_effort = nonce / 50  # normalize effort
                effort_boost = 1.0 + min(1.0, mining_effort * 0.2)  # +0-20% based on effort
                
                # Calculate energy with quality factors
                energy = base_energy * effort_boost * self.config.energy_factor
                
                # Apply random mutation if enabled
                if self.config.mutation_rate > 0:
                    energy = self._apply_energy_mutation(energy)
                
                # Create successfully mined span
                mined_span = {
                    **span,
                    "energy": energy,
                    "mined_at": datetime.utcnow().isoformat(),
                    "mined_by": f"worker_{worker_id}",
                    "mining_time": mining_time,
                    "mining_difficulty": current_difficulty,
                    "mining_effort": nonce,
                    "signature": self._generate_signature(span["id"], nonce)
                }
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(mined_span)
                mined_span["metadata"]["quality_score"] = quality_score
                
                # Reject if below quality threshold
                if quality_score < self.config.min_quality_threshold:
                    return {
                        "status": "failed",
                        "reason": f"Quality score {quality_score:.2f} below threshold {self.config.min_quality_threshold}",
                        "mining_time": time.time() - start_time
                    }
                
                # Calculate mining reward
                reward = self._calculate_reward(energy, mining_time)
                
                return {
                    "status": "success",
                    "span": mined_span,
                    "mining_time": mining_time,
                    "reward": reward
                }
            
            # Increment nonce for next attempt
            nonce += 1
            
            # Yield to other threads occasionally
            if nonce % 50 == 0:
                time.sleep(0.001)
        
        # Mining failed after maximum attempts
        return {
            "status": "failed",
            "reason": "Exceeded maximum mining attempts",
            "mining_time": time.time() - start_time
        }
    
    def _calculate_base_energy(self, span: Dict[str, Any]) -> float:
        """Calculate base energy for a span"""
        # Start with base value
        base = 10.0
        
        # Content complexity factor
        content = span["content"]
        if isinstance(content, dict) and "text" in content:
            text = content["text"]
            # Text length factor (with diminishing returns)
            length_factor = min(3.0, len(text) / 500)
            
            # Vocabulary richness
            words = text.lower().split()
            unique_ratio = len(set(words)) / max(1, len(words))
            vocab_factor = 1.0 + unique_ratio  # 1.0-2.0 based on vocabulary richness
            
            content_factor = length_factor * vocab_factor
        else:
            # Use JSON complexity for non-text content
            content_size = len(json.dumps(content))
            content_factor = min(3.0, content_size / 1000)
        
        # Parent relationships factor
        parent_count = len(span["parent_ids"])
        parent_factor = 1.0 + (parent_count * 0.15)
        
        # Metadata richness factor
        metadata_factor = min(1.5, len(json.dumps(span["metadata"])) / 200)
        
        # Priority factor (higher priority = more energy)
        priority = span["metadata"].get("mining_priority", 1.0)
        priority_factor = 1.0 + max(0, (1.0 - priority) * 0.5)  # 1.0-1.5 based on priority
        
        # Calculate final base energy
        energy = base * content_factor * parent_factor * metadata_factor * priority_factor
        
        # Cap at reasonable values
        return min(100.0, energy)
    
    def _calculate_quality_score(self, span: Dict[str, Any]) -> float:
        """Calculate quality score for a span"""
        # Start with base score
        score = 5.0
        
        # Content quality
        content = span["content"]
        if isinstance(content, dict) and "text" in content:
            text = content["text"]
            
            # Length factor (with diminishing returns)
            length_score = min(2.0, len(text) / 1000)
            
            # Vocabulary richness
            words = text.lower().split()
            unique_ratio = len(set(words)) / max(1, len(words))
            vocab_score = unique_ratio * 3.0
            
            # Combine text scores
            content_score = length_score + vocab_score
        else:
            # Non-text content uses complexity score
            content_score = min(3.0, len(json.dumps(content)) / 500)
        
        # Factor 2: Causal richness (parent relationships)
        parent_score = min(2.0, len(span["parent_ids"]) * 0.5)
        
        # Factor 3: Energy level
        energy_score = min(3.0, span["energy"] / 30)
        
        # Combine all scores
        final_score = score + content_score + parent_score + energy_score
        
        # Cap at 10
        return min(10.0, final_score)
    
    def _generate_proof(self, span_id: str, content: Any, nonce: int) -> str:
        """Generate a proof of work hash"""
        # Check cache first
        cache_key = f"{span_id}:{nonce}"
        if cache_key in self.proof_cache:
            return self.proof_cache[cache_key]
        
        # Create a deterministic string representation of the content
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)
        
        # Generate hash
        hash_input = f"{span_id}:{content_str}:{nonce}".encode()
        proof = hashlib.sha256(hash_input).hexdigest()
        
        # Cache the result
        self.proof_cache[cache_key] = proof
        
        # Limit cache size
        if len(self.proof_cache) > 10000:
            # Remove random entries to keep size in check
            keys_to_remove = random.sample(list(self.proof_cache.keys()), 1000)
            for key in keys_to_remove:
                del self.proof_cache[key]
        
        return proof
    
    def _check_proof_meets_difficulty(self, proof: str, difficulty: float) -> bool:
        """Check if a proof meets the difficulty target"""
        # Convert first 16 hex characters (64 bits) to integer
        proof_int = int(proof[:16], 16)
        
        # Calculate target threshold based on difficulty
        # Higher difficulty = lower threshold = harder to mine
        max_value = 2 ** 64  # Maximum value for 64 bits
        threshold = int(max_value / difficulty)
        
        # Check if proof is below threshold
        return proof_int < threshold
    
    def _generate_signature(self, span_id: str, nonce: int) -> str:
        """Generate a signature for a mined span"""
        timestamp = datetime.utcnow().isoformat()
        sig_input = f"{span_id}:{nonce}:{timestamp}".encode()
        return hashlib.sha256(sig_input).hexdigest()
    
    def _calculate_reward(self, energy: float, mining_time: float) -> float:
        """Calculate mining reward for a span"""
        # Base reward proportional to energy
        base_reward = energy * 0.1
        
        # Adjust based on mining time
        time_factor = 1.0
        if mining_time > 0:
            # Effort bonus for longer mining times (up to 2x)
            time_factor = min(2.0, 1.0 + (mining_time / 10.0))
        
        # Apply reward schema
        if self.config.reward_schema == "logarithmic":
            # Logarithmic rewards grow more slowly at higher energies
            import math
            reward = base_reward * (1 + math.log10(1 + energy / 10)) * time_factor
        elif self.config.reward_schema == "linear":
            # Linear rewards scale directly with energy
            reward = base_reward * time_factor
        else:
            # Default quadratic rewards
            reward = base_reward * (1 + (energy / 100)) * time_factor
        
        return reward
    
    def _verify_parent_spans(self, parent_ids: List[str]) -> bool:
        """Verify that parent spans exist and are valid"""
        # If farm is available, check with it
        if self.farm:
            for parent_id in parent_ids:
                if parent_id not in self.farm.spans:
                    return False
        return True
    
    def _apply_energy_mutation(self, energy: float) -> float:
        """Apply random mutations to energy values"""
        if random.random() < self.config.mutation_rate:
            # Apply a mutation (±10%)
            mutation_factor = 0.9 + (random.random() * 0.2)
            return energy * mutation_factor
        return energy
    
    def _estimate_mining_time(self, priority: float) -> float:
        """Estimate mining time based on queue size and priority"""
        queue_size = self.pending_spans.qsize()
        avg_time = self.stats.average_mining_time or 1.0
        
        # Calculate position penalty (higher queue = longer wait)
        position_penalty = queue_size / max(1, self.config.threads)
        
        # Calculate priority bonus (lower priority value = shorter wait)
        priority_bonus = priority
        
        # Estimate time in seconds
        estimated_time = avg_time * position_penalty * priority_bonus
        
        return estimated_time
    
    def _start_difficulty_adjuster(self) -> None:
        """Start thread for automatic difficulty adjustment"""
        adjuster = threading.Thread(
            target=self._difficulty_adjustment_worker,
            name="difficulty-adjuster"
        )
        adjuster.daemon = True
        adjuster.start()
        logger.info("Started difficulty adjustment thread")
    
    def _difficulty_adjustment_worker(self) -> None:
        """Worker thread that adjusts mining difficulty"""
        while self.mining_active:
            try:
                # Sleep for adjustment interval
                time.sleep(10)  # Check every 10 seconds
                
                # Skip if not enough mining history
                if self.stats.total_attempts < self.config.difficulty_adjustment_interval:
                    continue
                
                # Calculate current mining rate
                mining_time = time.time() - self.stats.start_time
                minutes = mining_time / 60
                
                if minutes < 0.5:  # Need at least 30 seconds of data
                    continue
                
                current_rate = self.stats.successful_mines / minutes
                target_rate = self.config.target_mining_rate
                
                # Adjust difficulty based on rate
                if current_rate > target_rate * 1.2:
                    # Too fast, increase difficulty
                    new_difficulty = self.stats.current_difficulty * 1.1
                elif current_rate < target_rate * 0.8:
                    # Too slow, decrease difficulty
                    new_difficulty = self.stats.current_difficulty * 0.9
                else:
                    # Within acceptable range
                    continue
                
                # Apply new difficulty with limits
                with self.lock:
                    self.stats.current_difficulty = max(0.1, min(100.0, new_difficulty))
                
                # Record in history
                self.difficulty_history.append(self.stats.current_difficulty)
                if len(self.difficulty_history) > 100:
                    self.difficulty_history.pop(0)
                
                logger.info(f"Adjusted mining difficulty to {self.stats.current_difficulty:.2f} "
                           f"(mining rate: {current_rate:.2f}/min, target: {target_rate}/min)")
                
            except Exception as e:
                logger.error(f"Error in difficulty adjuster: {str(e)}", exc_info=True)
                time.sleep(30)  # Longer sleep on error
    
    def _start_checkpointing_thread(self) -> None:
        """Start thread for periodic state checkpointing"""
        checkpointer = threading.Thread(
            target=self._checkpointing_worker,
            name="checkpointer"
        )
        checkpointer.daemon = True
        checkpointer.start()
        logger.info("Started checkpointing thread")
    
    def _checkpointing_worker(self) -> None:
        """Worker thread that periodically saves state"""
        while self.mining_active:
            try:
                # Sleep for checkpoint interval
                time.sleep(self.config.checkpoint_interval)
                
                # Save state
                self._save_state()
                
            except Exception as e:
                logger.error(f"Error in checkpointing: {str(e)}", exc_info=True)
                time.sleep(60)  # Longer sleep on error
    
    def _save_state(self) -> None:
        """Save miner state to disk"""
        try:
            # Create state dictionary
            state = {
                "stats": {
                    "total_attempts": self.stats.total_attempts,
                    "successful_mines": self.stats.successful_mines,
                    "failed_mines": self.stats.failed_mines,
                    "total_energy": self.stats.total_energy,
                    "highest_energy": self.stats.highest_energy,
                    "average_mining_time": self.stats.average_mining_time,
                    "current_difficulty": self.stats.current_difficulty,
                    "start_time": self.stats.start_time,
                    "total_rewards": self.stats.total_rewards
                },
                "config": {
                    "threads": self.config.threads,
                    "target_difficulty": self.config.target_difficulty,
                    "energy_factor": self.config.energy_factor,
                    "reward_schema": self.config.reward_schema
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Ensure directory exists
            os.makedirs("data/miner", exist_ok=True)
            
            # Save to file
            with open("data/miner/state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.debug("Saved miner state")
            
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}", exc_info=True)
    
    def _load_state(self) -> None:
        """Load miner state from disk"""
        try:
            state_file = "data/miner/state.json"
            if not os.path.exists(state_file):
                return
            
            with open(state_file, "r") as f:
                state = json.load(f)
            
            # Restore statistics
            stats = state.get("stats", {})
            self.stats.total_attempts = stats.get("total_attempts", 0)
            self.stats.successful_mines = stats.get("successful_mines", 0)
            self.stats.failed_mines = stats.get("failed_mines", 0)
            self.stats.total_energy = stats.get("total_energy", 0.0)
            self.stats.highest_energy = stats.get("highest_energy", 0.0)
            self.stats.average_mining_time = stats.get("average_mining_time", 0.0)
            self.stats.current_difficulty = stats.get("current_difficulty", self.config.target_difficulty)
            self.stats.total_rewards = stats.get("total_rewards", 0.0)
            
            logger.info(f"Loaded miner state: {self.stats.successful_mines} spans, "
                       f"difficulty: {self.stats.current_difficulty:.2f}")
            
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}", exc_info=True)