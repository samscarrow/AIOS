"""
Attention Management System - Managing cognitive resources as a scarce commodity
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import heapq
import logging
from .fault_tolerance import ResourceExhaustedError

logger = logging.getLogger(__name__)


@dataclass
class AttentionRequest:
    """Request for attention resources"""
    model_id: str
    tokens_required: int
    priority: float
    timestamp: float = field(default_factory=time.time)
    callback: Optional[asyncio.Future] = None
    
    def __lt__(self, other):
        # Higher priority comes first (negative for min heap)
        return -self.priority < -other.priority


class AttentionManager:
    """
    Manages attention as a finite resource that models compete for
    """
    
    def __init__(self, total_tokens: int = 100):
        self.total_tokens = total_tokens
        self.available_tokens = total_tokens
        self.allocated: Dict[str, int] = {}
        self.pending_requests: List[AttentionRequest] = []
        self.attention_history: List[Tuple[float, str, int]] = []
        self._lock = asyncio.Lock()
        
        # Resource bounds
        self.max_pending_requests = 50
        self.max_tokens_per_model = total_tokens // 2  # No single model can use more than 50%
        self.request_timeout = 30.0  # seconds
        
    async def request_attention(self, model_id: str, tokens: int, 
                               priority: float = 1.0) -> asyncio.Future:
        """Request attention tokens with priority"""
        # Validate request bounds
        if tokens <= 0:
            raise ValueError("Token request must be positive")
        if tokens > self.max_tokens_per_model:
            raise ResourceExhaustedError(f"Requested {tokens} tokens exceeds limit {self.max_tokens_per_model}")
        
        # Check if model would exceed individual limit
        current_allocation = self.allocated.get(model_id, 0)
        if current_allocation + tokens > self.max_tokens_per_model:
            raise ResourceExhaustedError(f"Model {model_id} would exceed token limit")
            
        async with self._lock:
            # Check pending queue limit
            if len(self.pending_requests) >= self.max_pending_requests:
                raise ResourceExhaustedError("Too many pending attention requests")
                
            future = asyncio.Future()
            request = AttentionRequest(
                model_id=model_id,
                tokens_required=tokens,
                priority=priority,
                callback=future
            )
            
            # Try immediate allocation
            if self.available_tokens >= tokens:
                self._allocate(model_id, tokens)
                future.set_result(True)
                return future
                
            # Queue the request with timeout
            heapq.heappush(self.pending_requests, request)
            logger.info(f"Queued attention request from {model_id} for {tokens} tokens")
            
            # Set timeout for request
            asyncio.create_task(self._timeout_request(request, self.request_timeout))
            
            # Try to preempt lower priority allocations
            await self._try_preemption(request)
            
            return future
            
    def _allocate(self, model_id: str, tokens: int):
        """Internal allocation method"""
        self.available_tokens -= tokens
        self.allocated[model_id] = self.allocated.get(model_id, 0) + tokens
        self.attention_history.append((time.time(), model_id, tokens))
        logger.debug(f"Allocated {tokens} tokens to {model_id}")
        
    async def release_attention(self, model_id: str, tokens: Optional[int] = None):
        """Release attention tokens back to the pool"""
        async with self._lock:
            if model_id not in self.allocated:
                return
                
            tokens_to_release = tokens or self.allocated[model_id]
            tokens_to_release = min(tokens_to_release, self.allocated[model_id])
            
            self.allocated[model_id] -= tokens_to_release
            if self.allocated[model_id] == 0:
                del self.allocated[model_id]
                
            self.available_tokens += tokens_to_release
            logger.debug(f"Released {tokens_to_release} tokens from {model_id}")
            
            # Process pending requests
            await self._process_pending()
            
    async def _process_pending(self):
        """Process pending attention requests"""
        satisfied = []
        
        while self.pending_requests and self.available_tokens > 0:
            request = heapq.heappop(self.pending_requests)
            
            if self.available_tokens >= request.tokens_required:
                self._allocate(request.model_id, request.tokens_required)
                if request.callback and not request.callback.done():
                    request.callback.set_result(True)
                satisfied.append(request.model_id)
            else:
                # Put it back if we can't satisfy it
                heapq.heappush(self.pending_requests, request)
                break
                
        if satisfied:
            logger.info(f"Satisfied attention requests for: {satisfied}")
            
    async def _try_preemption(self, request: AttentionRequest):
        """Try to preempt lower priority allocations"""
        # Find models with lower priority that we could preempt
        preemption_candidates = []
        
        for model_id, allocated_tokens in self.allocated.items():
            # In a real system, we'd track priorities of allocated models
            # For now, we'll use a simple heuristic
            if allocated_tokens >= request.tokens_required:
                preemption_candidates.append((model_id, allocated_tokens))
                
        # For now, we don't actually preempt - this is a placeholder
        # In a full implementation, we'd suspend lower priority models
        
    def get_attention_distribution(self) -> Dict[str, float]:
        """Get current attention distribution as percentages"""
        total_allocated = sum(self.allocated.values())
        if total_allocated == 0:
            return {}
            
        return {
            model_id: (tokens / self.total_tokens) * 100
            for model_id, tokens in self.allocated.items()
        }
        
    def get_status(self) -> Dict[str, any]:
        """Get current attention manager status"""
        return {
            "total_tokens": self.total_tokens,
            "available_tokens": self.available_tokens,
            "allocated": dict(self.allocated),
            "pending_requests": len(self.pending_requests),
            "distribution": self.get_attention_distribution(),
            "max_tokens_per_model": self.max_tokens_per_model,
            "max_pending_requests": self.max_pending_requests
        }
        
    async def _timeout_request(self, request: AttentionRequest, timeout: float):
        """Timeout a pending request"""
        await asyncio.sleep(timeout)
        
        # Check if request is still pending
        if request in self.pending_requests:
            self.pending_requests.remove(request)
            if request.callback and not request.callback.done():
                request.callback.set_exception(
                    TimeoutError(f"Attention request timed out for {request.model_id}")
                )
                logger.warning(f"Attention request timed out for {request.model_id}")
                
    def cleanup_stale_requests(self):
        """Clean up stale or invalid requests"""
        current_time = time.time()
        stale_requests = []
        
        for request in self.pending_requests:
            if (current_time - request.timestamp > self.request_timeout or
                request.callback.done()):
                stale_requests.append(request)
                
        for request in stale_requests:
            if request in self.pending_requests:
                self.pending_requests.remove(request)
                
        if stale_requests:
            logger.info(f"Cleaned up {len(stale_requests)} stale attention requests")