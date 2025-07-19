"""
Span Marketplace implementation for LogLineOS/DiamondSpan
Provides trading, auctions, and value tracking for spans
"""
import time
import asyncio
import json
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpanMarketplace")

class SpanMarketplace:
    """
    Diamond Span Marketplace implementation
    """
    
    def __init__(self, farm=None):
        self.farm = farm
        self.listings: Dict[str, Dict] = {}  # Active listings by span_id
        self.auctions: Dict[str, Dict] = {}  # Active auctions by span_id
        self.bids: Dict[str, List[Dict]] = {}  # Bids by auction_id
        self.trade_history: List[Dict] = []  # History of all trades
        self.price_history: Dict[str, List[Dict]] = {}  # Price history by span_id
        self.market_stats = {
            "volume_24h": 0.0,
            "trade_count_24h": 0,
            "highest_sale": 0.0,
            "total_volume": 0.0,
            "updated_at": datetime.utcnow().isoformat()
        }
    
    async def create_listing(self, span_id: str, seller: str, price: float, 
                            duration_hours: float = 24.0) -> Dict[str, Any]:
        """Create a new listing for a span"""
        # Validate the span exists
        if self.farm and span_id not in self.farm.spans:
            return {"status": "error", "message": "Span not found"}
        
        if span_id in self.listings:
            return {"status": "error", "message": "Span already listed"}
        
        # Create the listing
        expires_at = datetime.utcnow() + timedelta(hours=duration_hours)
        listing_id = f"list-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        
        listing = {
            "id": listing_id,
            "span_id": span_id,
            "seller": seller,
            "price": price,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "status": "active"
        }
        
        self.listings[span_id] = listing
        logger.info(f"Listing created: {listing_id} for span {span_id} at price {price}")
        
        return {"status": "success", "listing": listing}
    
    async def buy_span(self, span_id: str, buyer: str) -> Dict[str, Any]:
        """Buy a span at its listed price"""
        if span_id not in self.listings:
            return {"status": "error", "message": "Span not listed"}
        
        listing = self.listings[span_id]
        
        # Check if listing is still active
        if listing["status"] != "active":
            return {"status": "error", "message": "Listing is not active"}
        
        # Check if listing has expired
        expires_at = datetime.fromisoformat(listing["expires_at"])
        if datetime.utcnow() > expires_at:
            listing["status"] = "expired"
            return {"status": "error", "message": "Listing has expired"}
        
        # Process the purchase
        price = listing["price"]
        seller = listing["seller"]
        
        # Record the trade
        trade = {
            "id": f"trade-{int(time.time())}-{uuid.uuid4().hex[:8]}",
            "span_id": span_id,
            "listing_id": listing["id"],
            "buyer": buyer,
            "seller": seller,
            "price": price,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "direct_sale"
        }
        
        self.trade_history.append(trade)
        
        # Update price history
        if span_id not in self.price_history:
            self.price_history[span_id] = []
        self.price_history[span_id].append({
            "price": price,
            "timestamp": trade["timestamp"],
            "trade_id": trade["id"]
        })
        
        # Update market stats
        self.market_stats["volume_24h"] += price
        self.market_stats["trade_count_24h"] += 1
        self.market_stats["total_volume"] += price
        if price > self.market_stats["highest_sale"]:
            self.market_stats["highest_sale"] = price
        self.market_stats["updated_at"] = datetime.utcnow().isoformat()
        
        # Update listing status
        listing["status"] = "sold"
        
        logger.info(f"Span {span_id} sold to {buyer} for {price}")
        
        return {
            "status": "success", 
            "trade": trade,
            "message": f"Successfully purchased span {span_id} for {price}"
        }
    
    async def create_auction(self, span_id: str, seller: str, min_bid: float,
                            duration_hours: float = 24.0) -> Dict[str, Any]:
        """Create an auction for a span"""
        # Validate the span exists
        if self.farm and span_id not in self.farm.spans:
            return {"status": "error", "message": "Span not found"}
        
        if span_id in self.auctions:
            return {"status": "error", "message": "Span already in auction"}
        
        # Create the auction
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        auction_id = f"auc-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        
        auction = {
            "id": auction_id,
            "span_id": span_id,
            "seller": seller,
            "min_bid": min_bid,
            "current_bid": 0.0,
            "current_bidder": None,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": end_time.isoformat(),
            "status": "active",
            "bids_count": 0
        }
        
        self.auctions[span_id] = auction
        self.bids[auction_id] = []
        
        logger.info(f"Auction created: {auction_id} for span {span_id} with min bid {min_bid}")
        
        return {"status": "success", "auction": auction}
    
    async def place_bid(self, span_id: str, bidder: str, amount: float) -> Dict[str, Any]:
        """Place a bid on an auction"""
        if span_id not in self.auctions:
            return {"status": "error", "message": "Span not in auction"}
        
        auction = self.auctions[span_id]
        
        # Check if auction is still active
        if auction["status"] != "active":
            return {"status": "error", "message": "Auction is not active"}
        
        # Check if auction has ended
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
        bid = {
            "bidder": bidder,
            "amount": amount,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.bids[auction["id"]].append(bid)
        
        auction["current_bid"] = amount
        auction["current_bidder"] = bidder
        auction["bids_count"] += 1
        
        logger.info(f"Bid placed on span {span_id}: {amount} by {bidder}")
        
        return {"status": "success", "auction": auction, "bid": bid}
    
    async def finalize_auction(self, span_id: str) -> Dict[str, Any]:
        """Finalize an auction"""
        if span_id not in self.auctions:
            return {"status": "error", "message": "Span not in auction"}
        
        auction = self.auctions[span_id]
        
        # Check if auction should be finalized
        end_time = datetime.fromisoformat(auction["end_time"])
        if datetime.utcnow() < end_time and auction["status"] != "force_end":
            return {"status": "error", "message": "Auction still in progress"}
        
        # Finalize the auction
        if auction["current_bidder"]:
            # Auction successful
            auction["status"] = "completed"
            
            # Record the trade
            trade = {
                "id": f"trade-{int(time.time())}-{uuid.uuid4().hex[:8]}",
                "span_id": span_id,
                "auction_id": auction["id"],
                "buyer": auction["current_bidder"],
                "seller": auction["seller"],
                "price": auction["current_bid"],
                "timestamp": datetime.utcnow().isoformat(),
                "type": "auction_sale"
            }
            
            self.trade_history.append(trade)
            
            # Update price history
            if span_id not in self.price_history:
                self.price_history[span_id] = []
            self.price_history[span_id].append({
                "price": auction["current_bid"],
                "timestamp": trade["timestamp"],
                "trade_id": trade["id"]
            })
            
            # Update market stats
            self.market_stats["volume_24h"] += auction["current_bid"]
            self.market_stats["trade_count_24h"] += 1
            self.market_stats["total_volume"] += auction["current_bid"]
            if auction["current_bid"] > self.market_stats["highest_sale"]:
                self.market_stats["highest_sale"] = auction["current_bid"]
            self.market_stats["updated_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Auction completed for span {span_id}: sold to {auction['current_bidder']} for {auction['current_bid']}")
            
            return {
                "status": "success", 
                "auction": auction, 
                "trade": trade,
                "message": "Auction completed successfully"
            }
            
        else:
            # No bids
            auction["status"] = "ended_no_bids"
            logger.info(f"Auction ended with no bids for span {span_id}")
            
            return {
                "status": "success", 
                "auction": auction,
                "message": "Auction ended with no bids"
            }
    
    async def get_market_stats(self) -> Dict[str, Any]:
        """Get market statistics"""
        # Calculate 24h metrics
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=24)
        
        # Reset 24h stats
        volume_24h = 0.0
        trade_count_24h = 0
        
        # Calculate from trade history
        for trade in self.trade_history:
            trade_time = datetime.fromisoformat(trade["timestamp"])
            if trade_time >= cutoff:
                volume_24h += trade["price"]
                trade_count_24h += 1
        
        # Update stats
        self.market_stats["volume_24h"] = volume_24h
        self.market_stats["trade_count_24h"] = trade_count_24h
        self.market_stats["updated_at"] = now.isoformat()
        
        # Add counts
        stats = {
            **self.market_stats,
            "active_listings": len([l for l in self.listings.values() if l["status"] == "active"]),
            "active_auctions": len([a for a in self.auctions.values() if a["status"] == "active"]),
            "total_trades": len(self.trade_history)
        }
        
        return stats
    
    async def get_price_history(self, span_id: str) -> Dict[str, Any]:
        """Get price history for a span"""
        if span_id not in self.price_history:
            return {"status": "error", "message": "No price history for this span"}
        
        return {
            "status": "success",
            "span_id": span_id,
            "history": self.price_history[span_id]
        }
    
    async def clean_expired_listings(self):
        """Clean expired listings"""
        now = datetime.utcnow()
        expired_count = 0
        
        for span_id, listing in self.listings.items():
            if listing["status"] == "active":
                expires_at = datetime.fromisoformat(listing["expires_at"])
                if now > expires_at:
                    listing["status"] = "expired"
                    expired_count += 1
        
        return {"status": "success", "expired_count": expired_count}
    
    async def get_active_listings(self, page=1, page_size=20) -> Dict[str, Any]:
        """Get active listings with pagination"""
        active_listings = [l for l in self.listings.values() if l["status"] == "active"]
        
        # Sort by creation time (newest first)
        active_listings.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        page_listings = active_listings[start:end]
        
        return {
            "status": "success",
            "listings": page_listings,
            "total": len(active_listings),
            "page": page,
            "page_size": page_size,
            "total_pages": (len(active_listings) + page_size - 1) // page_size
        }
    
    async def get_active_auctions(self, page=1, page_size=20) -> Dict[str, Any]:
        """Get active auctions with pagination"""
        active_auctions = [a for a in self.auctions.values() if a["status"] == "active"]
        
        # Sort by end time (soonest first)
        active_auctions.sort(key=lambda x: x["end_time"])
        
        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        page_auctions = active_auctions[start:end]
        
        return {
            "status": "success",
            "auctions": page_auctions,
            "total": len(active_auctions),
            "page": page,
            "page_size": page_size,
            "total_pages": (len(active_auctions) + page_size - 1) // page_size
        }