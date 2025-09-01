#!/usr/bin/env python3
"""
Production Router API Server
FastAPI endpoint with monitoring, caching, and graceful degradation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import numpy as np

# Import production router
from production_task_router import (
    create_production_router,
    UnifiedProductionRouter,
    RoutingDecision
)

# ============================================================================
# API Models
# ============================================================================

class TaskRequest(BaseModel):
    """Incoming task routing request"""
    task: str = Field(..., description="Task description to route")
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")
    priority: Optional[int] = Field(5, description="Priority 1-10")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TaskResponse(BaseModel):
    """Task routing response"""
    request_id: Optional[str]
    task_type: Optional[str]
    complexity: str
    domain: Optional[str]
    confidence: float
    abstain: bool
    routing_time_ms: float
    routing_path: str
    recommended_models: List[str]
    attention_weight: float
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    uptime_seconds: float
    total_requests: int
    avg_latency_ms: float
    cache_hit_rate: float
    abstention_rate: float

class MetricsResponse(BaseModel):
    """Detailed metrics response"""
    total_requests: int
    requests_per_minute: float
    latency_percentiles: Dict[str, float]
    routing_path_distribution: Dict[str, int]
    task_type_distribution: Dict[str, int]
    complexity_distribution: Dict[str, int]
    cache_stats: Dict[str, Any]
    drift_score: float

# ============================================================================
# Router Service
# ============================================================================

class RouterService:
    """Production router service with caching and monitoring"""
    
    def __init__(
        self,
        proto_path: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_size: int = 10000,
        cache_ttl_seconds: int = 3600
    ):
        # Load router
        self.router = create_production_router(
            proto_path=proto_path,
            config_path=config_path
        )
        
        # Cache setup
        self.cache = {}
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl_seconds
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Metrics
        self.start_time = time.time()
        self.request_count = 0
        self.request_times = deque(maxlen=1000)  # Last 1000 request times
        self.latencies = deque(maxlen=1000)
        self.task_type_counts = {}
        self.complexity_counts = {}
        self.routing_path_counts = {}
        
        # Model recommendations based on task type
        self.model_recommendations = {
            'code_generation': ['qwen/qwen2.5-coder-14b', 'deepseek-coder-v2'],
            'analysis': ['qwen/qwen2.5-14b', 'llama-3.2-3b'],
            'debugging': ['qwen/qwen2.5-coder-14b', 'codestral-22b'],
            'system_design': ['qwen/qwen2.5-14b', 'llama-3.2-8b'],
            'data_analysis': ['qwen/qwen2.5-14b', 'llama-3.2-3b'],
            'logical_reasoning': ['qwen/qwen3-4b-thinking', 'llama-3.2-8b'],
            'documentation': ['llama-3.2-3b', 'qwen/qwen2.5-7b']
        }
        
        # Attention weights based on complexity
        self.attention_weights = {
            'trivial': 0.2,
            'simple': 0.3,
            'medium': 0.5,
            'complex': 0.7,
            'very_complex': 0.9
        }
    
    def _get_cache_key(self, task: str) -> str:
        """Generate cache key for task"""
        # Use first 200 chars for cache key
        return hash(task[:200])
    
    def _check_cache(self, task: str) -> Optional[Dict]:
        """Check if task result is cached"""
        key = self._get_cache_key(task)
        if key in self.cache:
            entry = self.cache[key]
            # Check if not expired
            if time.time() - entry['timestamp'] < self.cache_ttl:
                self.cache_hits += 1
                return entry['result']
            else:
                # Expired, remove from cache
                del self.cache[key]
        
        self.cache_misses += 1
        return None
    
    def _update_cache(self, task: str, result: Dict):
        """Update cache with new result"""
        key = self._get_cache_key(task)
        
        # LRU eviction if cache is full
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    async def route_task(self, request: TaskRequest) -> TaskResponse:
        """Route a task with caching and monitoring"""
        start_time = time.perf_counter()
        
        # Check cache first
        cached_result = self._check_cache(request.task)
        if cached_result:
            return TaskResponse(**cached_result, request_id=request.request_id)
        
        # Route task
        try:
            decision = await self.router.route(request.task)
        except Exception as e:
            # Fallback to basic routing on error
            decision = RoutingDecision(
                task_type=None,
                complexity='medium',
                domain=None,
                confidence=0.0,
                abstain=True,
                routing_time_ms=0.0,
                routing_path='error',
                metadata={'error': str(e)}
            )
        
        # Get model recommendations
        recommended_models = self.model_recommendations.get(
            decision.task_type,
            ['qwen/qwen2.5-7b']  # Default fallback
        )
        
        # Get attention weight
        attention_weight = self.attention_weights.get(
            decision.complexity,
            0.5  # Default
        )
        
        # Update metrics
        self.request_count += 1
        self.request_times.append(time.time())
        latency = (time.perf_counter() - start_time) * 1000
        self.latencies.append(latency)
        
        # Update counters
        if decision.task_type:
            self.task_type_counts[decision.task_type] = \
                self.task_type_counts.get(decision.task_type, 0) + 1
        
        self.complexity_counts[decision.complexity] = \
            self.complexity_counts.get(decision.complexity, 0) + 1
        
        self.routing_path_counts[decision.routing_path] = \
            self.routing_path_counts.get(decision.routing_path, 0) + 1
        
        # Build response
        response_data = {
            'task_type': decision.task_type,
            'complexity': decision.complexity,
            'domain': decision.domain,
            'confidence': decision.confidence,
            'abstain': decision.abstain,
            'routing_time_ms': latency,
            'routing_path': decision.routing_path,
            'recommended_models': recommended_models,
            'attention_weight': attention_weight,
            'metadata': {
                **decision.metadata,
                **request.metadata,
                'cached': False
            }
        }
        
        # Cache the result
        self._update_cache(request.task, response_data)
        
        return TaskResponse(
            request_id=request.request_id,
            **response_data
        )
    
    def get_health(self) -> HealthResponse:
        """Get service health status"""
        uptime = time.time() - self.start_time
        
        # Calculate metrics
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(total_cache_requests, 1)
        
        avg_latency = np.mean(self.latencies) if self.latencies else 0.0
        
        # Get abstention rate from router
        router_metrics = self.router.get_metrics()
        abstention_rate = router_metrics.get('abstention_rate', 0.0)
        
        return HealthResponse(
            status='healthy',
            uptime_seconds=uptime,
            total_requests=self.request_count,
            avg_latency_ms=avg_latency,
            cache_hit_rate=cache_hit_rate,
            abstention_rate=abstention_rate
        )
    
    def get_metrics(self) -> MetricsResponse:
        """Get detailed metrics"""
        # Calculate RPM
        now = time.time()
        recent_requests = sum(1 for t in self.request_times 
                            if now - t < 60)
        rpm = recent_requests
        
        # Calculate latency percentiles
        if self.latencies:
            latencies_array = np.array(self.latencies)
            percentiles = {
                'p50': float(np.percentile(latencies_array, 50)),
                'p75': float(np.percentile(latencies_array, 75)),
                'p90': float(np.percentile(latencies_array, 90)),
                'p95': float(np.percentile(latencies_array, 95)),
                'p99': float(np.percentile(latencies_array, 99))
            }
        else:
            percentiles = {'p50': 0, 'p75': 0, 'p90': 0, 'p95': 0, 'p99': 0}
        
        # Cache stats
        cache_stats = {
            'size': len(self.cache),
            'max_size': self.cache_size,
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            'ttl_seconds': self.cache_ttl
        }
        
        # Get drift score if available
        drift_score = 0.0
        if hasattr(self.router, 'semantic_router') and self.router.semantic_router:
            drift_score = self.router.semantic_router._calculate_drift()
        
        return MetricsResponse(
            total_requests=self.request_count,
            requests_per_minute=rpm,
            latency_percentiles=percentiles,
            routing_path_distribution=self.routing_path_counts,
            task_type_distribution=self.task_type_counts,
            complexity_distribution=self.complexity_counts,
            cache_stats=cache_stats,
            drift_score=drift_score
        )
    
    async def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Task Router API",
    description="Production-ready task routing service with multi-tier routing strategies",
    version="1.0.0"
)

# Initialize service
service = None

@app.on_event("startup")
async def startup_event():
    """Initialize router service on startup"""
    global service
    
    # Check for calibration files
    proto_path = "router_calibration/prototypes.npy"
    config_path = "router_calibration/router_config.json"
    
    if not Path(proto_path).exists():
        print("Warning: No prototype file found. Using default configuration.")
        proto_path = None
    
    if not Path(config_path).exists():
        print("Warning: No config file found. Using default configuration.")
        config_path = None
    
    service = RouterService(
        proto_path=proto_path,
        config_path=config_path,
        cache_size=10000,
        cache_ttl_seconds=3600
    )
    
    print("Router service initialized successfully")

@app.post("/route", response_model=TaskResponse)
async def route_task(request: TaskRequest) -> TaskResponse:
    """
    Route a task to determine its type, complexity, and recommended models.
    
    The router uses a multi-tier strategy:
    1. Fast semantic similarity matching (<10ms)
    2. Zero-shot classification for ambiguous cases
    3. Fine-tuned model for maximum accuracy (if available)
    """
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await service.route_task(request)

@app.post("/route/batch", response_model=List[TaskResponse])
async def route_batch(requests: List[TaskRequest]) -> List[TaskResponse]:
    """
    Route multiple tasks in a single request.
    More efficient for bulk processing.
    """
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Process in parallel for better throughput
    tasks = [service.route_task(req) for req in requests]
    return await asyncio.gather(*tasks)

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Get service health status"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return service.get_health()

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get detailed service metrics"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return service.get_metrics()

@app.post("/cache/clear")
async def clear_cache():
    """Clear the routing cache"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    await service.clear_cache()
    return {"status": "success", "message": "Cache cleared"}

@app.get("/")
async def root():
    """API documentation"""
    return {
        "name": "Task Router API",
        "version": "1.0.0",
        "endpoints": {
            "/route": "Route a single task",
            "/route/batch": "Route multiple tasks",
            "/health": "Health check",
            "/metrics": "Detailed metrics",
            "/cache/clear": "Clear routing cache",
            "/docs": "Interactive API documentation"
        }
    }


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Task Router API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--proto-path", help="Path to prototype embeddings")
    parser.add_argument("--config-path", help="Path to router configuration")
    
    args = parser.parse_args()
    
    # Run server
    uvicorn.run(
        "router_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )