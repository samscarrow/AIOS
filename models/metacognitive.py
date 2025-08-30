"""
Meta-Cognitive Reasoning - Models that think about thinking
Enables GAIA to reason about its own cognitive processes
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CognitiveState(Enum):
    """Different cognitive states the system can be in"""
    EXPLORING = "exploring"      # Divergent thinking, broad search
    FOCUSING = "focusing"        # Convergent thinking, narrow search
    REFLECTING = "reflecting"    # Analyzing own processes
    LEARNING = "learning"        # Updating internal models
    PLANNING = "planning"        # Strategic thinking
    MONITORING = "monitoring"    # Observing performance


class MetaStrategy(Enum):
    """Meta-level strategies for cognitive processing"""
    DEPTH_FIRST = "depth_first"          # Deep dive into single thread
    BREADTH_FIRST = "breadth_first"      # Explore many options
    ITERATIVE_DEEPENING = "iterative_deepening"  # Gradually increase depth
    BEST_FIRST = "best_first"            # Follow most promising leads
    REFLECTION_DRIVEN = "reflection_driven"      # Reflect then act
    ADAPTIVE = "adaptive"                # Change strategy based on context


@dataclass
class CognitiveTrace:
    """Records a trace of cognitive processing"""
    trace_id: str
    model_chain: List[str] = field(default_factory=list)
    attention_flow: List[Tuple[str, float]] = field(default_factory=list)  # (model, attention)
    decision_points: List[Dict[str, Any]] = field(default_factory=list)
    context_transitions: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    success: Optional[bool] = None
    meta_observations: List[str] = field(default_factory=list)


@dataclass
class CognitivePattern:
    """A learned pattern of cognitive processing"""
    pattern_id: str
    trigger_conditions: Dict[str, Any]
    model_sequence: List[str]
    expected_performance: float
    context_requirements: Dict[str, Any]
    success_rate: float = 0.0
    usage_count: int = 0
    adaptations: List[Dict[str, Any]] = field(default_factory=list)


class MetaCognitiveReasoner:
    """
    Meta-cognitive reasoning engine that thinks about GAIA's thinking processes
    """
    
    def __init__(self, kernel=None):
        self.kernel = kernel
        self.cognitive_state = CognitiveState.MONITORING
        self.current_strategy = MetaStrategy.ADAPTIVE
        
        # Cognitive traces and patterns
        self.active_traces: Dict[str, CognitiveTrace] = {}
        self.completed_traces: List[CognitiveTrace] = []
        self.learned_patterns: Dict[str, CognitivePattern] = {}
        
        # Meta-cognitive knowledge
        self.strategy_performance: Dict[MetaStrategy, float] = {}
        self.context_strategy_map: Dict[str, MetaStrategy] = {}
        self.cognitive_biases: List[str] = []
        
        # Self-monitoring
        self.performance_history: List[Tuple[float, Dict[str, Any]]] = []
        self.reflection_triggers: Set[str] = {
            "poor_performance", "repeated_failures", "context_change", "resource_pressure"
        }
        
    async def start_cognitive_trace(self, task_context: Dict[str, Any]) -> str:
        """Start tracing a cognitive process"""
        trace_id = f"trace_{int(time.time() * 1000)}"
        trace = CognitiveTrace(trace_id=trace_id)
        
        # Analyze initial context and set strategy
        trace.meta_observations.append("Starting cognitive trace")
        strategy = await self._select_cognitive_strategy(task_context)
        trace.meta_observations.append(f"Selected strategy: {strategy.value}")
        
        self.active_traces[trace_id] = trace
        self.current_strategy = strategy
        
        logger.info(f"Started cognitive trace {trace_id} with strategy {strategy.value}")
        return trace_id
        
    async def _select_cognitive_strategy(self, context: Dict[str, Any]) -> MetaStrategy:
        """Meta-reasoning to select optimal cognitive strategy"""
        
        # Analyze context characteristics
        complexity = context.get('complexity', 0.5)
        time_pressure = context.get('time_pressure', 0.5)
        novelty = context.get('novelty', 0.5)
        resource_availability = context.get('resources', 1.0)
        
        # Strategy selection based on meta-analysis
        if novelty > 0.7:
            # High novelty - explore broadly first
            return MetaStrategy.BREADTH_FIRST
        elif complexity > 0.8 and time_pressure < 0.3:
            # High complexity, low time pressure - go deep
            return MetaStrategy.DEPTH_FIRST
        elif time_pressure > 0.7:
            # Time pressure - use best known approach
            return MetaStrategy.BEST_FIRST
        elif resource_availability < 0.3:
            # Low resources - be strategic
            return MetaStrategy.ITERATIVE_DEEPENING
        else:
            # Default to adaptive approach
            return MetaStrategy.ADAPTIVE
            
    async def observe_cognitive_step(self, trace_id: str, model_id: str, 
                                   attention_used: float, context: Dict[str, Any]):
        """Observe and record a cognitive processing step"""
        if trace_id not in self.active_traces:
            return
            
        trace = self.active_traces[trace_id]
        trace.model_chain.append(model_id)
        trace.attention_flow.append((model_id, attention_used))
        
        # Meta-observation: analyze this step
        meta_observation = await self._analyze_cognitive_step(
            model_id, attention_used, context, trace
        )
        trace.meta_observations.append(meta_observation)
        
        # Check if strategy adjustment needed
        if await self._should_adjust_strategy(trace, context):
            new_strategy = await self._select_cognitive_strategy(context)
            if new_strategy != self.current_strategy:
                trace.meta_observations.append(
                    f"Strategy adjusted: {self.current_strategy.value} -> {new_strategy.value}"
                )
                self.current_strategy = new_strategy
                
    async def _analyze_cognitive_step(self, model_id: str, attention_used: float,
                                    context: Dict[str, Any], trace: CognitiveTrace) -> str:
        """Analyze a single cognitive step"""
        
        # Analyze attention efficiency
        if attention_used > 30:  # High attention usage
            if len(trace.model_chain) < 3:
                return f"High attention usage ({attention_used}) early in process - may indicate complexity"
            else:
                return f"High attention usage ({attention_used}) late in process - possible inefficiency"
        
        # Analyze model selection patterns
        if len(trace.model_chain) > 1:
            prev_model = trace.model_chain[-2]
            if prev_model == model_id:
                return f"Repeated model usage ({model_id}) - may indicate stuck pattern"
                
        # Analyze context transitions
        if 'context_shift' in context:
            return f"Context shift detected - cognitive flexibility demonstrated"
            
        return f"Normal cognitive step: {model_id} using {attention_used} attention"
        
    async def _should_adjust_strategy(self, trace: CognitiveTrace, context: Dict[str, Any]) -> bool:
        """Determine if cognitive strategy should be adjusted"""
        
        # Check for stuck patterns
        if len(trace.model_chain) > 5:
            recent_models = trace.model_chain[-3:]
            if len(set(recent_models)) == 1:  # Same model repeated
                return True
                
        # Check attention efficiency
        if len(trace.attention_flow) > 3:
            avg_attention = np.mean([att for _, att in trace.attention_flow[-3:]])
            if avg_attention > 40:  # High attention usage
                return True
                
        # Check for context changes
        if context.get('urgency_increased', False):
            return True
            
        return False
        
    async def complete_cognitive_trace(self, trace_id: str, success: bool, 
                                     performance_metrics: Dict[str, float]):
        """Complete a cognitive trace and learn from it"""
        if trace_id not in self.active_traces:
            return
            
        trace = self.active_traces[trace_id]
        trace.end_time = time.time()
        trace.success = success
        trace.performance_metrics = performance_metrics
        
        # Meta-reflection on the completed trace
        reflection = await self._reflect_on_trace(trace)
        trace.meta_observations.append(f"Final reflection: {reflection}")
        
        # Learn patterns from this trace
        await self._extract_cognitive_patterns(trace)
        
        # Update strategy performance
        duration = trace.end_time - trace.start_time
        efficiency = performance_metrics.get('efficiency', 0.5)
        strategy_score = efficiency / max(duration, 0.1)  # Efficiency per second
        
        if self.current_strategy in self.strategy_performance:
            old_score = self.strategy_performance[self.current_strategy]
            self.strategy_performance[self.current_strategy] = (old_score + strategy_score) / 2
        else:
            self.strategy_performance[self.current_strategy] = strategy_score
            
        # Store completed trace
        self.completed_traces.append(trace)
        del self.active_traces[trace_id]
        
        logger.info(f"Completed cognitive trace {trace_id}: success={success}, "
                   f"strategy={self.current_strategy.value}, score={strategy_score:.3f}")
                   
    async def _reflect_on_trace(self, trace: CognitiveTrace) -> str:
        """Reflect on a completed cognitive trace"""
        
        # Analyze model usage patterns
        model_diversity = len(set(trace.model_chain)) / len(trace.model_chain) if trace.model_chain else 0
        
        # Analyze attention efficiency
        total_attention = sum(att for _, att in trace.attention_flow)
        avg_attention = total_attention / len(trace.attention_flow) if trace.attention_flow else 0
        
        # Generate reflection
        if trace.success and model_diversity > 0.7:
            return f"Successful diverse thinking (diversity={model_diversity:.2f})"
        elif trace.success and avg_attention < 20:
            return f"Efficient successful processing (avg_attention={avg_attention:.1f})"
        elif not trace.success and model_diversity < 0.3:
            return f"Failed - insufficient cognitive diversity (diversity={model_diversity:.2f})"
        elif not trace.success and avg_attention > 40:
            return f"Failed - inefficient attention usage (avg_attention={avg_attention:.1f})"
        else:
            return f"Mixed results - model_diversity={model_diversity:.2f}, avg_attention={avg_attention:.1f}"
            
    async def _extract_cognitive_patterns(self, trace: CognitiveTrace):
        """Extract reusable cognitive patterns from successful traces"""
        
        if not trace.success or len(trace.model_chain) < 3:
            return
            
        # Create pattern from successful sequence
        pattern_id = f"pattern_{len(self.learned_patterns)}"
        
        # Identify trigger conditions (simplified)
        trigger_conditions = {
            'model_chain_length': len(trace.model_chain),
            'total_attention': sum(att for _, att in trace.attention_flow),
            'strategy_used': self.current_strategy.value
        }
        
        # Extract core model sequence
        model_sequence = trace.model_chain
        
        # Calculate expected performance
        expected_performance = trace.performance_metrics.get('efficiency', 0.5)
        
        pattern = CognitivePattern(
            pattern_id=pattern_id,
            trigger_conditions=trigger_conditions,
            model_sequence=model_sequence,
            expected_performance=expected_performance,
            context_requirements={},
            success_rate=1.0,  # First success
            usage_count=1
        )
        
        self.learned_patterns[pattern_id] = pattern
        logger.info(f"Extracted cognitive pattern {pattern_id} with performance {expected_performance:.3f}")
        
    async def suggest_cognitive_approach(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal cognitive approach based on learned patterns"""
        
        # Find matching patterns
        matching_patterns = []
        for pattern in self.learned_patterns.values():
            if pattern.success_rate > 0.6:  # Only consider successful patterns
                matching_patterns.append(pattern)
                
        if not matching_patterns:
            # No learned patterns - use meta-strategy
            strategy = await self._select_cognitive_strategy(context)
            return {
                'approach': 'meta_strategy',
                'strategy': strategy.value,
                'confidence': 0.5,
                'reasoning': 'No learned patterns available, using meta-strategy'
            }
            
        # Rank patterns by expected performance
        matching_patterns.sort(key=lambda p: p.expected_performance, reverse=True)
        best_pattern = matching_patterns[0]
        
        return {
            'approach': 'learned_pattern',
            'pattern_id': best_pattern.pattern_id,
            'model_sequence': best_pattern.model_sequence,
            'expected_performance': best_pattern.expected_performance,
            'confidence': best_pattern.success_rate,
            'reasoning': f'Using pattern with {best_pattern.success_rate:.1%} success rate'
        }
        
    def get_meta_cognitive_insights(self) -> Dict[str, Any]:
        """Get insights about GAIA's cognitive processes"""
        
        if not self.completed_traces:
            return {'status': 'No cognitive traces completed yet'}
            
        # Analyze completed traces
        successful_traces = [t for t in self.completed_traces if t.success]
        success_rate = len(successful_traces) / len(self.completed_traces)
        
        # Analyze model usage patterns
        all_models = []
        for trace in self.completed_traces:
            all_models.extend(trace.model_chain)
            
        model_usage = {}
        for model in all_models:
            model_usage[model] = model_usage.get(model, 0) + 1
            
        # Strategy effectiveness
        best_strategy = max(self.strategy_performance.items(), 
                          key=lambda x: x[1]) if self.strategy_performance else None
        
        return {
            'cognitive_traces_completed': len(self.completed_traces),
            'overall_success_rate': success_rate,
            'learned_patterns': len(self.learned_patterns),
            'model_usage_frequency': model_usage,
            'best_performing_strategy': best_strategy[0].value if best_strategy else None,
            'best_strategy_score': best_strategy[1] if best_strategy else None,
            'current_cognitive_state': self.cognitive_state.value,
            'active_traces': len(self.active_traces)
        }