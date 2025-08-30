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
        
        # Cognitive capabilities (initialized in initialize() to avoid circular imports)
        self.meta_cognitive = None
        self.introspector = None
        self.cognitive_strategist = None
        
    async def initialize(self):
        """Initialize the kernel and start the event loop"""
        logger.info("Initializing GAIA Kernel...")
        self.event_loop = asyncio.get_event_loop()
        self.active_context = CognitiveContext()
        self.contexts[self.active_context.context_id] = self.active_context
        
        # Initialize cognitive capabilities
        try:
            from ..models.metacognitive import MetaCognitiveReasoner
            from ..models.introspection import CognitiveIntrospector
            from ..models.cognitive_strategies import AdaptiveCognitiveStrategist
            self.meta_cognitive = MetaCognitiveReasoner(kernel=self)
            self.introspector = CognitiveIntrospector(kernel=self)
            self.cognitive_strategist = AdaptiveCognitiveStrategist(kernel=self)
        except ImportError as e:
            logger.warning(f"Could not initialize cognitive capabilities: {e}")
            # Fallback to None values
            pass
        
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
        """Spawn a new thought stream asynchronously with cognitive tracing"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")
            
        thought_id = f"thought_{uuid.uuid4().hex[:8]}"
        context = context or self.active_context
        
        # Start cognitive trace for meta-analysis (if available)
        trace_id = None
        optimal_strategy = None
        
        if self.meta_cognitive and self.cognitive_strategist:
            task_context = {
                'complexity': len(str(input_data)) / 1000,  # Rough complexity estimate
                'novelty': 0.5 if model_id in self.association_graph else 0.8,
                'time_pressure': 0.3,  # Low default pressure
                'resources': self.available_attention / self.attention_pool
            }
            trace_id = await self.meta_cognitive.start_cognitive_trace(task_context)
            optimal_strategy = await self.cognitive_strategist.select_optimal_strategy(task_context)
        
        # Create async task for this thought
        task = asyncio.create_task(
            self._execute_thought(thought_id, model_id, input_data, context, trace_id, optimal_strategy)
        )
        self.thought_streams[thought_id] = task
        
        logger.info(f"Spawned thought {thought_id} with model {model_id}, trace {trace_id}")
        return thought_id
        
    async def _execute_thought(self, thought_id: str, model_id: str,
                              input_data: Any, context: CognitiveContext, trace_id: str, strategy):
        """Execute a thought asynchronously with cognitive monitoring and adaptive strategies"""
        model = self.models[model_id]
        model.state = ModelState.ACTIVE
        model.last_accessed = time.time()
        model.access_frequency += 1
        
        attention_used = 0
        start_time = time.time()
        
        try:
            # Allocate attention based on strategy (if available)
            if strategy:
                strategy_attention = strategy.attention_allocation.get(model_id, 0.3)
                required_attention = int(strategy_attention * 50)  # Scale to available attention
            else:
                required_attention = 15  # Default attention requirement
            
            if await self.allocate_attention(model_id, required_attention):
                attention_used = required_attention
            else:
                # Insufficient attention - request cognitive strategy adjustment
                if self.meta_cognitive and trace_id:
                    await self.meta_cognitive.observe_cognitive_step(
                        trace_id, model_id, 0, {'urgency_increased': True}
                    )
                # Wait briefly and try again with reduced requirement
                await asyncio.sleep(0.05)
                reduced_attention = max(5, required_attention // 2)
                if await self.allocate_attention(model_id, reduced_attention):
                    attention_used = reduced_attention
                else:
                    raise ResourceExhaustedError("Insufficient attention for thought execution")
            
            # Record cognitive step (if available)
            if self.meta_cognitive and trace_id:
                await self.meta_cognitive.observe_cognitive_step(
                    trace_id, model_id, attention_used, context.attention_distribution
                )
            
            # Simulate model execution with introspection (if available)
            mental_snapshot = None
            if self.introspector:
                mental_snapshot = await self.introspector.take_mental_snapshot()
            await asyncio.sleep(0.1)  # Placeholder for actual model execution
            
            # Check for associations to spawn (adaptive based on strategy)
            associations = self.association_graph.get(model_id, set())
            spawned_associations = 0
            max_associations = min(strategy.branching_factor, len(associations)) if strategy else min(2, len(associations))
            
            for associated_model in list(associations)[:max_associations]:
                branching_limit = strategy.branching_factor if strategy else 2
                if (associated_model in context.active_associations or 
                    spawned_associations < branching_limit):
                    
                    # Get cognitive approach recommendation (if available)
                    should_spawn = True
                    if self.meta_cognitive:
                        approach = await self.meta_cognitive.suggest_cognitive_approach({
                            'current_model': model_id,
                            'association_target': associated_model,
                            'complexity': len(str(input_data)) / 1000,
                            'strategy_type': strategy.strategy_type.value if strategy else 'default'
                        })
                        
                        # Use strategy-specific confidence threshold
                        confidence_threshold = 0.4 if (strategy and strategy.strategy_type.name == "EXPLORATORY") else 0.6
                        should_spawn = approach['confidence'] > confidence_threshold
                    
                    if should_spawn:
                        child_context = context.spawn_child()
                        await self.spawn_thought(
                            associated_model, 
                            {"parent": input_data, "association": True, "strategy_guided": bool(strategy)},
                            child_context
                        )
                        spawned_associations += 1
            
            # Complete processing
            processing_time = time.time() - start_time
            result = {
                "thought_id": thought_id, 
                "result": f"Processed by {model_id}",
                "processing_time": processing_time,
                "attention_used": attention_used,
                "associations_spawned": spawned_associations,
                "mental_state": mental_snapshot.dominant_state if mental_snapshot else "unknown"
            }
            
            # Complete cognitive trace (if available)
            performance_metrics = {
                'efficiency': 1.0 / max(processing_time, 0.01),
                'attention_efficiency': required_attention / max(attention_used, 1),
                'association_success': spawned_associations / max(len(associations), 1)
            }
            
            if self.meta_cognitive and trace_id:
                await self.meta_cognitive.complete_cognitive_trace(
                    trace_id, 
                    success=True, 
                    performance_metrics=performance_metrics
                )
            
            # Record strategy performance for adaptation (if available)
            if self.cognitive_strategist and strategy:
                task_context = {
                    'complexity': len(str(input_data)) / 1000,
                    'novelty': 0.5 if model_id in self.association_graph else 0.8,
                    'time_pressure': 0.3,
                    'resources': self.available_attention / self.attention_pool
                }
                
                outcome = {
                    'success': True,
                    'processing_time': processing_time,
                    'attention_used': attention_used,
                    'efficiency': performance_metrics['efficiency'],
                    'attention_efficiency': performance_metrics['attention_efficiency']
                }
                
                await self.cognitive_strategist.record_strategy_performance(
                    strategy.strategy_id, task_context, outcome
                )
            
            await self.release_attention(model_id, attention_used)
            model.state = ModelState.DORMANT
            return result
            
        except Exception as e:
            # Handle failure in cognitive trace
            processing_time = time.time() - start_time
            
            if self.meta_cognitive and trace_id:
                await self.meta_cognitive.complete_cognitive_trace(
                    trace_id, 
                    success=False, 
                    performance_metrics={
                        'efficiency': 0.1,
                        'processing_time': processing_time,
                        'error': str(e)
                    }
                )
            
            # Record strategy failure for adaptation (if available)
            if self.cognitive_strategist and strategy:
                task_context = {
                    'complexity': len(str(input_data)) / 1000,
                    'novelty': 0.5 if model_id in self.association_graph else 0.8,
                    'time_pressure': 0.3,
                    'resources': self.available_attention / self.attention_pool
                }
                
                outcome = {
                    'success': False,
                    'processing_time': processing_time,
                    'attention_used': attention_used,
                    'efficiency': 0.1,
                    'error': str(e)
                }
                
                await self.cognitive_strategist.record_strategy_performance(
                    strategy.strategy_id, task_context, outcome
                )
            
            if attention_used > 0:
                await self.release_attention(model_id, attention_used)
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
        """Get current kernel status including cognitive insights"""
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
        
        # Add cognitive insights (if available)
        if self.meta_cognitive:
            status["cognitive_insights"] = self.meta_cognitive.get_meta_cognitive_insights()
            status["current_cognitive_state"] = self.meta_cognitive.cognitive_state.value
            status["active_cognitive_traces"] = len(self.meta_cognitive.active_traces)
        if self.cognitive_strategist:
            status["strategy_analytics"] = self.cognitive_strategist.get_strategy_analytics()
        
        return status
        
    async def get_cognitive_approach_recommendation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendation for cognitive approach based on current context"""
        if not self.meta_cognitive:
            return {"approach": "unavailable", "confidence": 0.0, "reasoning": "Meta-cognitive capabilities not initialized"}
        return await self.meta_cognitive.suggest_cognitive_approach(context)
        
    async def perform_cognitive_introspection(self) -> Dict[str, Any]:
        """Perform deep cognitive introspection on current mental state"""
        if not self.introspector:
            return {"error": "Introspection capabilities not initialized"}
            
        snapshot = await self.introspector.take_mental_snapshot()
        deep_insights = await self.introspector.deep_introspection(depth=2)
        
        return {
            "mental_snapshot": {
                "dominant_state": snapshot.dominant_state,
                "attention_distribution": snapshot.attention_distribution,
                "active_processes": snapshot.active_processes,
                "resource_pressure": snapshot.resource_pressure,
                "temporal_focus": snapshot.temporal_focus
            },
            "deep_insights": deep_insights,
            "cognitive_observations": snapshot.cognitive_observations
        }
        
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