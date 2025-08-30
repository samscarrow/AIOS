"""
Core GAIA Kernel - Orchestrates intelligence as the primary computational substrate
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
from .fault_tolerance import FaultToleranceManager, ErrorSeverity, ResourceExhaustedError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelState(Enum):
    """States for AI models in the kernel"""
    DORMANT = "dormant"
    LOADING = "loading"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    YIELDING = "yielding"


@dataclass
class CognitiveContext:
    """Represents the cognitive context at any point in time"""
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    semantic_embedding: List[float] = field(default_factory=list)
    active_associations: Set[str] = field(default_factory=set)
    attention_distribution: Dict[str, float] = field(default_factory=dict)
    temporal_window: float = 0.0
    parent_context: Optional[str] = None
    child_contexts: List[str] = field(default_factory=list)
    
    def spawn_child(self) -> 'CognitiveContext':
        """Create a child context for branching thoughts"""
        child = CognitiveContext(
            parent_context=self.context_id,
            semantic_embedding=self.semantic_embedding.copy(),
            temporal_window=self.temporal_window
        )
        self.child_contexts.append(child.context_id)
        return child


@dataclass
class ModelInstance:
    """Represents an AI model instance in the kernel"""
    model_id: str
    model_type: str
    state: ModelState = ModelState.DORMANT
    memory_footprint: int = 0  # in MB
    vram_required: int = 0  # in MB
    last_accessed: float = 0.0
    access_frequency: int = 0
    associations: Set[str] = field(default_factory=set)
    semantic_tags: Set[str] = field(default_factory=set)
    
    def calculate_priority(self, context: CognitiveContext) -> float:
        """Calculate model priority based on current context"""
        base_priority = self.access_frequency * 0.3
        recency_factor = 1.0 / (time.time() - self.last_accessed + 1)
        association_bonus = len(self.associations & context.active_associations) * 0.2
        return base_priority + recency_factor * 0.5 + association_bonus


class GAIAKernel:
    """
    The core kernel that manages cognitive resources and model orchestration
    """
    
    def __init__(self):
        self.models: Dict[str, ModelInstance] = {}
        self.contexts: Dict[str, CognitiveContext] = {}
        self.active_context: Optional[CognitiveContext] = None
        self.attention_pool: int = 100  # Total attention tokens
        self.available_attention: int = 100
        self.association_graph: Dict[str, Set[str]] = {}
        self.thought_streams: Dict[str, asyncio.Task] = {}
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Fault tolerance
        self.fault_manager = FaultToleranceManager()
        self.max_models = 50  # Resource bounds
        self.max_contexts = 100
        self.max_thought_streams = 20
        
    async def initialize(self):
        """Initialize the kernel and start the event loop"""
        logger.info("Initializing GAIA Kernel...")
        self.event_loop = asyncio.get_event_loop()
        self.active_context = CognitiveContext()
        self.contexts[self.active_context.context_id] = self.active_context
        
        # Register core components with fault tolerance
        self.fault_manager.register_component(
            "kernel", 
            recovery_handler=self._recover_kernel,
            degradation_handler=self._degrade_kernel
        )
        
        logger.info(f"GAIA Kernel initialized with context {self.active_context.context_id}")
        
    def register_model(self, model_id: str, model_type: str, 
                      memory_footprint: int, vram_required: int,
                      semantic_tags: Optional[Set[str]] = None):
        """Register a new model with the kernel"""
        # Check resource bounds
        if len(self.models) >= self.max_models:
            raise ResourceExhaustedError(f"Cannot register model: max models ({self.max_models}) exceeded")
            
        model = ModelInstance(
            model_id=model_id,
            model_type=model_type,
            memory_footprint=memory_footprint,
            vram_required=vram_required,
            semantic_tags=semantic_tags or set()
        )
        self.models[model_id] = model
        
        # Register model with fault tolerance
        self.fault_manager.register_component(model_id)
        
        logger.info(f"Registered model {model_id} of type {model_type}")
        
    def create_association(self, model_a: str, model_b: str, strength: float = 1.0):
        """Create an association between two models"""
        if model_a not in self.association_graph:
            self.association_graph[model_a] = set()
        if model_b not in self.association_graph:
            self.association_graph[model_b] = set()
            
        self.association_graph[model_a].add(model_b)
        self.association_graph[model_b].add(model_a)
        
        # Update model associations
        if model_a in self.models:
            self.models[model_a].associations.add(model_b)
        if model_b in self.models:
            self.models[model_b].associations.add(model_a)
            
        logger.info(f"Created association between {model_a} and {model_b}")
        
    async def spawn_thought(self, model_id: str, input_data: Any,
                          context: Optional[CognitiveContext] = None) -> str:
        """Spawn a new thought stream asynchronously"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")
            
        thought_id = f"thought_{uuid.uuid4().hex[:8]}"
        context = context or self.active_context
        
        # Create async task for this thought
        task = asyncio.create_task(
            self._execute_thought(thought_id, model_id, input_data, context)
        )
        self.thought_streams[thought_id] = task
        
        logger.info(f"Spawned thought {thought_id} with model {model_id}")
        return thought_id
        
    async def _execute_thought(self, thought_id: str, model_id: str,
                              input_data: Any, context: CognitiveContext):
        """Execute a thought asynchronously"""
        model = self.models[model_id]
        model.state = ModelState.ACTIVE
        model.last_accessed = time.time()
        model.access_frequency += 1
        
        try:
            # Simulate model execution
            await asyncio.sleep(0.1)  # Placeholder for actual model execution
            
            # Check for associations to spawn
            associations = self.association_graph.get(model_id, set())
            for associated_model in associations:
                if associated_model in context.active_associations:
                    # Spawn associated thought without blocking
                    child_context = context.spawn_child()
                    await self.spawn_thought(
                        associated_model, 
                        {"parent": input_data, "association": True},
                        child_context
                    )
                    
            model.state = ModelState.DORMANT
            return {"thought_id": thought_id, "result": f"Processed by {model_id}"}
            
        except Exception as e:
            logger.error(f"Error in thought {thought_id}: {e}")
            model.state = ModelState.DORMANT
            raise
            
    async def allocate_attention(self, model_id: str, tokens_required: int) -> bool:
        """Allocate attention tokens to a model"""
        if self.available_attention >= tokens_required:
            self.available_attention -= tokens_required
            logger.debug(f"Allocated {tokens_required} attention tokens to {model_id}")
            return True
        return False
        
    async def release_attention(self, model_id: str, tokens: int):
        """Release attention tokens back to the pool"""
        self.available_attention = min(self.attention_pool, self.available_attention + tokens)
        logger.debug(f"Released {tokens} attention tokens from {model_id}")
        
    def get_kernel_status(self) -> Dict[str, Any]:
        """Get current kernel status"""
        status = {
            "active_models": sum(1 for m in self.models.values() if m.state == ModelState.ACTIVE),
            "total_models": len(self.models),
            "active_thoughts": len([t for t in self.thought_streams.values() if not t.done()]),
            "available_attention": self.available_attention,
            "total_attention": self.attention_pool,
            "contexts": len(self.contexts)
        }
        
        # Add fault tolerance info
        health_report = self.fault_manager.get_system_health_report()
        status["health"] = health_report["overall_health"]
        status["component_health"] = health_report["components"]
        
        return status
        
    async def _recover_kernel(self):
        """Recover kernel from error state"""
        logger.info("Attempting kernel recovery...")
        
        # Clear failed thought streams
        failed_streams = [tid for tid, task in self.thought_streams.items() if task.done() and task.exception()]
        for tid in failed_streams:
            del self.thought_streams[tid]
            
        # Reset attention pool if corrupted
        if self.available_attention < 0 or self.available_attention > self.attention_pool:
            self.available_attention = self.attention_pool
            logger.info("Reset attention pool")
            
        # Cleanup orphaned contexts
        active_context_ids = {self.active_context.context_id} if self.active_context else set()
        orphaned = set(self.contexts.keys()) - active_context_ids
        for context_id in orphaned:
            if len(self.contexts[context_id].child_contexts) == 0:  # No children
                del self.contexts[context_id]
                
        logger.info("Kernel recovery completed")
        
    async def _degrade_kernel(self, reason: str):
        """Gracefully degrade kernel functionality"""
        logger.warning(f"Kernel degradation triggered: {reason}")
        
        if reason == "shutdown":
            # Cancel all active thought streams
            for task in self.thought_streams.values():
                if not task.done():
                    task.cancel()
            
            # Clear all contexts except active
            if self.active_context:
                self.contexts = {self.active_context.context_id: self.active_context}
            else:
                self.contexts.clear()
                
            logger.info("Kernel shutdown completed")
        else:
            # Reduce resource limits
            self.max_models = max(10, self.max_models // 2)
            self.max_contexts = max(20, self.max_contexts // 2)
            self.max_thought_streams = max(5, self.max_thought_streams // 2)
            
            logger.info(f"Reduced resource limits: models={self.max_models}, contexts={self.max_contexts}")
            
    async def execute_with_fault_tolerance(self, operation_name: str, operation: callable, *args, **kwargs):
        """Execute operation with fault tolerance wrapper"""
        return await self.fault_manager.execute_with_fault_tolerance(
            "kernel", operation, *args, **kwargs
        )