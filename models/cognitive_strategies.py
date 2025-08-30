"""
Adaptive Cognitive Strategies - System learns to optimize its own thinking patterns
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of cognitive strategies"""
    SEQUENTIAL = "sequential"           # Process thoughts one after another
    PARALLEL = "parallel"               # Process multiple thoughts simultaneously  
    HIERARCHICAL = "hierarchical"       # Deep dive into specific areas
    EXPLORATORY = "exploratory"         # Broad search and discovery
    REFLECTIVE = "reflective"           # Self-examination and analysis
    ADAPTIVE_HYBRID = "adaptive_hybrid" # Dynamically combines strategies


@dataclass
class StrategyPerformance:
    """Tracks performance metrics for a cognitive strategy"""
    strategy_type: StrategyType
    success_count: int = 0
    failure_count: int = 0
    total_processing_time: float = 0.0
    total_attention_used: int = 0
    context_matches: int = 0
    recent_successes: deque = field(default_factory=lambda: deque(maxlen=10))
    performance_trend: float = 0.0  # -1 to 1, negative means declining
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def efficiency(self) -> float:
        """Calculate efficiency (success per unit time)"""
        if self.total_processing_time == 0:
            return 0.0
        return self.success_count / self.total_processing_time
    
    @property
    def attention_efficiency(self) -> float:
        """Calculate attention efficiency"""
        if self.total_attention_used == 0:
            return 0.0
        return self.success_count / self.total_attention_used


@dataclass
class CognitiveStrategy:
    """Represents a specific cognitive processing strategy"""
    strategy_id: str
    strategy_type: StrategyType
    model_sequence: List[str]
    attention_allocation: Dict[str, float]
    branching_factor: int = 1
    depth_limit: int = 3
    parallel_streams: int = 1
    context_conditions: Dict[str, Any] = field(default_factory=dict)
    adaptation_rules: List[str] = field(default_factory=list)
    performance: StrategyPerformance = field(init=False)
    
    def __post_init__(self):
        self.performance = StrategyPerformance(self.strategy_type)


class AdaptiveCognitiveStrategist:
    """
    Learns and adapts cognitive strategies based on performance feedback
    """
    
    def __init__(self, kernel=None):
        self.kernel = kernel
        
        # Strategy management
        self.available_strategies: Dict[str, CognitiveStrategy] = {}
        self.active_strategies: Dict[str, CognitiveStrategy] = {}
        self.strategy_history: List[Tuple[str, Dict[str, Any], float]] = []
        
        # Learning parameters
        self.adaptation_threshold = 0.3  # When to consider strategy change
        self.exploration_rate = 0.15     # How often to try new strategies
        self.performance_window = 20     # Number of recent performances to track
        
        # Context analysis
        self.context_clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.strategy_context_map: Dict[str, Set[str]] = defaultdict(set)
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
    def _initialize_default_strategies(self):
        """Initialize a set of default cognitive strategies"""
        
        # Sequential processing strategy
        sequential = CognitiveStrategy(
            strategy_id="sequential_default",
            strategy_type=StrategyType.SEQUENTIAL,
            model_sequence=["reasoning", "analysis", "synthesis"],
            attention_allocation={"reasoning": 0.4, "analysis": 0.4, "synthesis": 0.2},
            branching_factor=1,
            parallel_streams=1,
            context_conditions={"complexity": "low", "time_pressure": "low"}
        )
        
        # Parallel processing strategy  
        parallel = CognitiveStrategy(
            strategy_id="parallel_default",
            strategy_type=StrategyType.PARALLEL,
            model_sequence=["reasoning", "intuition", "analysis"],
            attention_allocation={"reasoning": 0.3, "intuition": 0.3, "analysis": 0.4},
            branching_factor=3,
            parallel_streams=3,
            context_conditions={"complexity": "medium", "resources": "high"}
        )
        
        # Exploratory strategy
        exploratory = CognitiveStrategy(
            strategy_id="exploratory_default", 
            strategy_type=StrategyType.EXPLORATORY,
            model_sequence=["creative", "associative", "reasoning", "synthesis"],
            attention_allocation={"creative": 0.3, "associative": 0.3, "reasoning": 0.2, "synthesis": 0.2},
            branching_factor=4,
            depth_limit=2,
            parallel_streams=2,
            context_conditions={"novelty": "high", "time_pressure": "low"}
        )
        
        # Reflective strategy
        reflective = CognitiveStrategy(
            strategy_id="reflective_default",
            strategy_type=StrategyType.REFLECTIVE,
            model_sequence=["introspection", "reasoning", "meta_analysis"],
            attention_allocation={"introspection": 0.4, "reasoning": 0.3, "meta_analysis": 0.3},
            branching_factor=1,
            depth_limit=4,
            parallel_streams=1,
            context_conditions={"self_analysis": "required", "depth": "high"}
        )
        
        # Store strategies
        for strategy in [sequential, parallel, exploratory, reflective]:
            self.available_strategies[strategy.strategy_id] = strategy
            
        logger.info(f"Initialized {len(self.available_strategies)} default cognitive strategies")
        
    async def select_optimal_strategy(self, context: Dict[str, Any]) -> CognitiveStrategy:
        """Select the optimal cognitive strategy for given context"""
        
        # Get context characteristics
        complexity = context.get('complexity', 0.5)
        novelty = context.get('novelty', 0.5) 
        time_pressure = context.get('time_pressure', 0.3)
        resources = context.get('resources', 1.0)
        
        # Score all available strategies
        strategy_scores = {}
        
        for strategy_id, strategy in self.available_strategies.items():
            score = await self._score_strategy_for_context(strategy, context)
            strategy_scores[strategy_id] = score
            
        # Select best strategy, with some exploration
        if np.random.random() < self.exploration_rate:
            # Explore: try a random strategy
            selected_id = np.random.choice(list(strategy_scores.keys()))
            logger.info(f"Exploring with strategy {selected_id}")
        else:
            # Exploit: use best scoring strategy
            selected_id = max(strategy_scores, key=strategy_scores.get)
            logger.info(f"Selected optimal strategy {selected_id} (score: {strategy_scores[selected_id]:.3f})")
            
        return self.available_strategies[selected_id]
        
    async def _score_strategy_for_context(self, strategy: CognitiveStrategy, context: Dict[str, Any]) -> float:
        """Score how well a strategy matches the current context"""
        score = 0.0
        
        # Base performance score
        if strategy.performance.success_count > 0:
            score += strategy.performance.success_rate * 0.4
            score += min(strategy.performance.efficiency / 10, 0.3)  # Cap efficiency bonus
            score += strategy.performance.attention_efficiency * 0.2
        else:
            score += 0.3  # Neutral score for untested strategies
            
        # Context matching
        complexity = context.get('complexity', 0.5)
        novelty = context.get('novelty', 0.5)
        time_pressure = context.get('time_pressure', 0.3)
        resources = context.get('resources', 1.0)
        
        # Strategy-specific scoring
        if strategy.strategy_type == StrategyType.SEQUENTIAL:
            score += 0.2 if complexity < 0.4 else -0.1
            score += 0.1 if time_pressure < 0.5 else -0.2
            
        elif strategy.strategy_type == StrategyType.PARALLEL:
            score += 0.3 if resources > 0.7 else -0.2
            score += 0.2 if complexity > 0.6 else -0.1
            
        elif strategy.strategy_type == StrategyType.EXPLORATORY:
            score += 0.3 if novelty > 0.6 else -0.1
            score += 0.1 if time_pressure < 0.4 else -0.3
            
        elif strategy.strategy_type == StrategyType.REFLECTIVE:
            score += 0.2 if complexity > 0.5 else 0.0
            score -= 0.3 if time_pressure > 0.7 else 0.0
            
        # Performance trend adjustment
        score += strategy.performance.performance_trend * 0.1
        
        return max(0.0, score)  # Ensure non-negative score
        
    async def record_strategy_performance(self, strategy_id: str, context: Dict[str, Any], 
                                        outcome: Dict[str, Any]):
        """Record performance outcome for strategy adaptation"""
        if strategy_id not in self.available_strategies:
            return
            
        strategy = self.available_strategies[strategy_id]
        success = outcome.get('success', False)
        processing_time = outcome.get('processing_time', 0.0)
        attention_used = outcome.get('attention_used', 0)
        
        # Update performance metrics
        if success:
            strategy.performance.success_count += 1
            strategy.performance.recent_successes.append(1)
        else:
            strategy.performance.failure_count += 1
            strategy.performance.recent_successes.append(0)
            
        strategy.performance.total_processing_time += processing_time
        strategy.performance.total_attention_used += attention_used
        strategy.performance.context_matches += 1
        
        # Update performance trend
        if len(strategy.performance.recent_successes) >= 5:
            recent = list(strategy.performance.recent_successes)
            first_half = np.mean(recent[:len(recent)//2])
            second_half = np.mean(recent[len(recent)//2:])
            strategy.performance.performance_trend = second_half - first_half
            
        # Store in history
        self.strategy_history.append((strategy_id, context.copy(), outcome.get('efficiency', 0.0)))
        
        # Trigger adaptation if needed
        await self._consider_strategy_adaptation(strategy, context, outcome)
        
        logger.debug(f"Recorded performance for {strategy_id}: success={success}, "
                    f"success_rate={strategy.performance.success_rate:.3f}")
        
    async def _consider_strategy_adaptation(self, strategy: CognitiveStrategy, 
                                          context: Dict[str, Any], outcome: Dict[str, Any]):
        """Consider adapting strategy based on performance"""
        
        # Only adapt strategies that have enough data
        total_attempts = strategy.performance.success_count + strategy.performance.failure_count
        if total_attempts < 5:
            return
            
        # Check if performance is declining
        if (strategy.performance.success_rate < self.adaptation_threshold or
            strategy.performance.performance_trend < -0.3):
            
            logger.info(f"Strategy {strategy.strategy_id} underperforming, considering adaptation")
            await self._adapt_strategy(strategy, context, outcome)
            
    async def _adapt_strategy(self, strategy: CognitiveStrategy, context: Dict[str, Any], 
                            outcome: Dict[str, Any]):
        """Adapt an underperforming strategy"""
        
        # Create adapted version
        adapted_id = f"{strategy.strategy_id}_adapted_{int(time.time())}"
        
        # Clone original strategy
        adapted = CognitiveStrategy(
            strategy_id=adapted_id,
            strategy_type=strategy.strategy_type,
            model_sequence=strategy.model_sequence.copy(),
            attention_allocation=strategy.attention_allocation.copy(),
            branching_factor=strategy.branching_factor,
            depth_limit=strategy.depth_limit,
            parallel_streams=strategy.parallel_streams,
            context_conditions=strategy.context_conditions.copy()
        )
        
        # Apply adaptations based on failure patterns
        if outcome.get('attention_efficiency', 0) < 0.1:
            # Poor attention efficiency - reduce attention allocation
            for model in adapted.attention_allocation:
                adapted.attention_allocation[model] *= 0.8
                
        if outcome.get('processing_time', 0) > 2.0:
            # Too slow - reduce complexity
            adapted.depth_limit = max(1, adapted.depth_limit - 1)
            adapted.parallel_streams = max(1, adapted.parallel_streams - 1)
            
        if strategy.performance.success_rate < 0.3:
            # Very poor performance - try different model sequence
            if len(adapted.model_sequence) > 1:
                # Shuffle model sequence
                np.random.shuffle(adapted.model_sequence)
                
        # Add adaptation rule
        adapted.adaptation_rules.append(f"Adapted from {strategy.strategy_id} due to poor performance")
        
        # Store adapted strategy
        self.available_strategies[adapted_id] = adapted
        
        logger.info(f"Created adapted strategy {adapted_id} from {strategy.strategy_id}")
        
    def create_custom_strategy(self, strategy_type: StrategyType, model_sequence: List[str],
                              attention_allocation: Dict[str, float], **kwargs) -> str:
        """Create a custom cognitive strategy"""
        strategy_id = f"custom_{strategy_type.value}_{int(time.time())}"
        
        strategy = CognitiveStrategy(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            model_sequence=model_sequence,
            attention_allocation=attention_allocation,
            **kwargs
        )
        
        self.available_strategies[strategy_id] = strategy
        logger.info(f"Created custom strategy {strategy_id}")
        
        return strategy_id
        
    def get_strategy_analytics(self) -> Dict[str, Any]:
        """Get analytics about strategy performance"""
        analytics = {
            "total_strategies": len(self.available_strategies),
            "strategy_performances": {},
            "best_strategies": [],
            "adaptation_count": len([s for s in self.available_strategies.values() if s.adaptation_rules])
        }
        
        # Calculate performance for each strategy
        for strategy_id, strategy in self.available_strategies.items():
            perf = strategy.performance
            analytics["strategy_performances"][strategy_id] = {
                "success_rate": perf.success_rate,
                "efficiency": perf.efficiency,
                "attention_efficiency": perf.attention_efficiency,
                "total_uses": perf.success_count + perf.failure_count,
                "performance_trend": perf.performance_trend
            }
            
        # Find best performing strategies
        if self.available_strategies:
            best_by_success = max(self.available_strategies.values(), 
                                key=lambda s: s.performance.success_rate)
            best_by_efficiency = max(self.available_strategies.values(),
                                   key=lambda s: s.performance.efficiency)
            
            analytics["best_strategies"] = [
                {"metric": "success_rate", "strategy": best_by_success.strategy_id, 
                 "value": best_by_success.performance.success_rate},
                {"metric": "efficiency", "strategy": best_by_efficiency.strategy_id,
                 "value": best_by_efficiency.performance.efficiency}
            ]
            
        return analytics
        
    async def optimize_all_strategies(self):
        """Perform optimization pass on all strategies"""
        optimized_count = 0
        
        for strategy in self.available_strategies.values():
            if (strategy.performance.success_count + strategy.performance.failure_count) >= 10:
                if strategy.performance.success_rate < 0.6:
                    await self._adapt_strategy(strategy, {}, {"success": False})
                    optimized_count += 1
                    
        logger.info(f"Optimized {optimized_count} underperforming strategies")
        return optimized_count