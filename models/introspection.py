"""
Cognitive Introspection - GAIA's ability to examine its own mental states
Self-reflection and self-awareness capabilities
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


class IntrospectionLevel(Enum):
    """Different levels of introspective analysis"""
    SURFACE = "surface"          # Basic state awareness
    PROCESS = "process"          # Understanding of cognitive processes
    META = "meta"                # Awareness of awareness itself
    RECURSIVE = "recursive"      # Self-referential introspection


class MentalState(Enum):
    """Different mental states GAIA can be in"""
    CLEAR = "clear"              # High clarity, low confusion
    CONFUSED = "confused"        # High uncertainty, conflicting info
    FOCUSED = "focused"          # Strong attention on specific area
    SCATTERED = "scattered"      # Attention spread across many areas
    CREATIVE = "creative"        # High associative activity
    ANALYTICAL = "analytical"    # Systematic, logical processing
    LEARNING = "learning"        # Actively updating knowledge
    STUCK = "stuck"              # Unable to make progress


@dataclass
class MentalStateSnapshot:
    """A snapshot of GAIA's mental state at a point in time"""
    timestamp: float = field(default_factory=time.time)
    dominant_state: MentalState = MentalState.CLEAR
    state_confidence: float = 0.5
    
    # Cognitive load indicators
    attention_distribution: Dict[str, float] = field(default_factory=dict)
    active_model_count: int = 0
    working_memory_load: float = 0.0
    
    # Process indicators
    thought_stream_coherence: float = 0.5
    association_strength: float = 0.5
    context_stability: float = 0.5
    
    # Meta indicators
    self_awareness_level: float = 0.5
    confidence_in_reasoning: float = 0.5
    perceived_effectiveness: float = 0.5
    
    # Qualitative observations
    observations: List[str] = field(default_factory=list)


@dataclass
class CognitiveGradient:
    """Tracks how mental state changes over time"""
    state_trajectory: List[MentalState] = field(default_factory=list)
    attention_flow: List[Dict[str, float]] = field(default_factory=list)
    coherence_trend: List[float] = field(default_factory=list)
    confidence_trend: List[float] = field(default_factory=list)
    window_size: int = 10


class CognitiveIntrospector:
    """
    Introspection system that allows GAIA to examine its own cognitive processes
    """
    
    def __init__(self, kernel=None):
        self.kernel = kernel
        self.mental_state_history: deque = deque(maxlen=100)
        self.current_mental_state = MentalState.CLEAR
        self.introspection_level = IntrospectionLevel.SURFACE
        
        # Cognitive gradients
        self.attention_gradient = CognitiveGradient()
        self.coherence_gradient = CognitiveGradient()
        
        # Self-awareness components
        self.self_model: Dict[str, Any] = {
            'strengths': ['pattern_recognition', 'associative_thinking'],
            'weaknesses': ['long_term_consistency', 'resource_estimation'],
            'preferences': ['depth_over_breadth', 'creative_connections'],
            'biases': ['recency_bias', 'availability_heuristic']
        }
        
        # Meta-cognitive awareness
        self.awareness_stack: List[str] = []  # Stack of what we're aware of being aware of
        self.recursive_depth = 0
        self.max_recursive_depth = 3
        
        # Performance self-assessment
        self.self_assessment_history: List[Dict[str, float]] = []
        
    async def take_mental_snapshot(self) -> MentalStateSnapshot:
        """Take a comprehensive snapshot of current mental state"""
        
        snapshot = MentalStateSnapshot()
        
        if self.kernel:
            # Get current system state
            kernel_status = self.kernel.get_kernel_status()
            
            # Analyze attention distribution
            if hasattr(self.kernel, 'attention_manager'):
                attention_status = self.kernel.attention_manager.get_status()
                snapshot.attention_distribution = attention_status.get('distribution', {})
                snapshot.working_memory_load = (
                    attention_status['total_tokens'] - attention_status['available_tokens']
                ) / attention_status['total_tokens']
                
            # Analyze cognitive load
            snapshot.active_model_count = kernel_status.get('active_models', 0)
            
            # Determine dominant mental state
            snapshot.dominant_state = await self._assess_mental_state(kernel_status)
            snapshot.state_confidence = await self._calculate_state_confidence()
            
        # Analyze thought stream coherence
        snapshot.thought_stream_coherence = await self._assess_coherence()
        
        # Assess association strength
        snapshot.association_strength = await self._assess_association_strength()
        
        # Evaluate context stability
        snapshot.context_stability = await self._assess_context_stability()
        
        # Meta-cognitive assessments
        snapshot.self_awareness_level = self._calculate_self_awareness_level()
        snapshot.confidence_in_reasoning = await self._assess_reasoning_confidence()
        snapshot.perceived_effectiveness = await self._assess_perceived_effectiveness()
        
        # Generate qualitative observations
        snapshot.observations = await self._generate_observations(snapshot)
        
        # Store snapshot
        self.mental_state_history.append(snapshot)
        self.current_mental_state = snapshot.dominant_state
        
        return snapshot
        
    async def _assess_mental_state(self, kernel_status: Dict[str, Any]) -> MentalState:
        """Assess current dominant mental state"""
        
        active_models = kernel_status.get('active_models', 0)
        total_models = kernel_status.get('total_models', 1)
        available_attention = kernel_status.get('available_attention', 100)
        total_attention = kernel_status.get('total_attention', 100)
        
        # Calculate key indicators
        model_activation_ratio = active_models / total_models if total_models > 0 else 0
        attention_utilization = (total_attention - available_attention) / total_attention
        
        # Assess based on patterns
        if attention_utilization > 0.8 and model_activation_ratio > 0.5:
            return MentalState.FOCUSED
        elif attention_utilization > 0.9:
            return MentalState.SCATTERED
        elif attention_utilization < 0.2:
            return MentalState.CLEAR
        elif model_activation_ratio < 0.1:
            return MentalState.STUCK
        else:
            # Check recent history for patterns
            if len(self.mental_state_history) > 3:
                recent_states = [s.dominant_state for s in list(self.mental_state_history)[-3:]]
                if len(set(recent_states)) == 1:  # Same state repeated
                    return MentalState.STUCK
                    
            return MentalState.ANALYTICAL  # Default
            
    async def _calculate_state_confidence(self) -> float:
        """Calculate confidence in current state assessment"""
        
        if len(self.mental_state_history) < 2:
            return 0.5
            
        # Check consistency with recent history
        recent_states = [s.dominant_state for s in list(self.mental_state_history)[-5:]]
        consistency = len([s for s in recent_states if s == self.current_mental_state]) / len(recent_states)
        
        return min(0.9, consistency + 0.1)
        
    async def _assess_coherence(self) -> float:
        """Assess coherence of thought streams"""
        
        if not self.kernel or not hasattr(self.kernel, 'thought_streams'):
            return 0.5
            
        # Simple coherence metric based on thought completion rate
        total_streams = len(self.kernel.thought_streams)
        if total_streams == 0:
            return 1.0
            
        completed_streams = sum(1 for task in self.kernel.thought_streams.values() if task.done())
        coherence = completed_streams / total_streams
        
        return coherence
        
    async def _assess_association_strength(self) -> float:
        """Assess strength of associative connections"""
        
        if not self.kernel or not hasattr(self.kernel, 'association_graph'):
            return 0.5
            
        # Simple metric based on association density
        total_models = len(self.kernel.models)
        total_associations = sum(len(associations) for associations in self.kernel.association_graph.values())
        
        if total_models < 2:
            return 0.0
            
        # Normalize by maximum possible associations
        max_associations = total_models * (total_models - 1)
        association_density = total_associations / max_associations if max_associations > 0 else 0
        
        return min(1.0, association_density * 2)  # Scale to reasonable range
        
    async def _assess_context_stability(self) -> float:
        """Assess stability of cognitive context"""
        
        if len(self.mental_state_history) < 3:
            return 0.5
            
        # Check attention distribution stability
        recent_snapshots = list(self.mental_state_history)[-3:]
        attention_stability = 0.0
        
        for i in range(1, len(recent_snapshots)):
            prev_dist = recent_snapshots[i-1].attention_distribution
            curr_dist = recent_snapshots[i].attention_distribution
            
            # Calculate similarity between attention distributions
            common_keys = set(prev_dist.keys()) & set(curr_dist.keys())
            if common_keys:
                similarities = [
                    1 - abs(prev_dist.get(key, 0) - curr_dist.get(key, 0)) / 100
                    for key in common_keys
                ]
                attention_stability += np.mean(similarities)
                
        return attention_stability / max(1, len(recent_snapshots) - 1)
        
    def _calculate_self_awareness_level(self) -> float:
        """Calculate current level of self-awareness"""
        
        # Base self-awareness on introspection depth and recursive awareness
        base_awareness = 0.3  # Minimum awareness level
        
        # Increase based on introspection level
        level_bonus = {
            IntrospectionLevel.SURFACE: 0.1,
            IntrospectionLevel.PROCESS: 0.3,
            IntrospectionLevel.META: 0.5,
            IntrospectionLevel.RECURSIVE: 0.7
        }.get(self.introspection_level, 0.1)
        
        # Increase based on recursive depth
        recursive_bonus = min(0.2, self.recursive_depth * 0.1)
        
        # Increase based on self-model richness
        model_bonus = len(self.self_model.get('strengths', [])) * 0.05
        
        return min(1.0, base_awareness + level_bonus + recursive_bonus + model_bonus)
        
    async def _assess_reasoning_confidence(self) -> float:
        """Assess confidence in current reasoning processes"""
        
        if not self.mental_state_history:
            return 0.5
            
        latest_snapshot = self.mental_state_history[-1]
        
        # High confidence indicators
        confidence_factors = []
        
        # Coherence contributes to confidence
        confidence_factors.append(latest_snapshot.thought_stream_coherence)
        
        # Context stability contributes to confidence
        confidence_factors.append(latest_snapshot.context_stability)
        
        # Clear mental states are more confident
        state_confidence_map = {
            MentalState.CLEAR: 0.9,
            MentalState.FOCUSED: 0.8,
            MentalState.ANALYTICAL: 0.7,
            MentalState.CREATIVE: 0.6,
            MentalState.LEARNING: 0.6,
            MentalState.SCATTERED: 0.3,
            MentalState.CONFUSED: 0.2,
            MentalState.STUCK: 0.1
        }
        confidence_factors.append(state_confidence_map.get(latest_snapshot.dominant_state, 0.5))
        
        return np.mean(confidence_factors)
        
    async def _assess_perceived_effectiveness(self) -> float:
        """Assess how effective the system perceives itself to be"""
        
        if not self.self_assessment_history:
            return 0.5
            
        # Use recent self-assessments
        recent_assessments = self.self_assessment_history[-5:]
        effectiveness_scores = [assessment.get('effectiveness', 0.5) for assessment in recent_assessments]
        
        return np.mean(effectiveness_scores)
        
    async def _generate_observations(self, snapshot: MentalStateSnapshot) -> List[str]:
        """Generate qualitative observations about current mental state"""
        
        observations = []
        
        # State-specific observations
        if snapshot.dominant_state == MentalState.FOCUSED:
            observations.append("Currently in a focused state - attention concentrated")
        elif snapshot.dominant_state == MentalState.SCATTERED:
            observations.append("Attention appears scattered across multiple areas")
        elif snapshot.dominant_state == MentalState.STUCK:
            observations.append("Detecting signs of cognitive stagnation")
        elif snapshot.dominant_state == MentalState.CREATIVE:
            observations.append("High associative activity detected - creative mode")
            
        # Load-based observations
        if snapshot.working_memory_load > 0.8:
            observations.append("Working memory approaching capacity")
        elif snapshot.working_memory_load < 0.2:
            observations.append("Low cognitive load - capacity available")
            
        # Coherence observations
        if snapshot.thought_stream_coherence > 0.8:
            observations.append("Thought processes highly coherent")
        elif snapshot.thought_stream_coherence < 0.3:
            observations.append("Detecting fragmented thought patterns")
            
        # Self-awareness observations
        if snapshot.self_awareness_level > 0.7:
            observations.append("High self-awareness detected")
        elif snapshot.self_awareness_level < 0.3:
            observations.append("Limited self-awareness currently")
            
        return observations
        
    async def perform_deep_introspection(self) -> Dict[str, Any]:
        """Perform deep introspective analysis"""
        
        self.introspection_level = IntrospectionLevel.META
        
        # Take current snapshot
        current_snapshot = await self.take_mental_snapshot()
        
        # Analyze patterns in mental state history
        state_patterns = self._analyze_state_patterns()
        
        # Assess cognitive strengths and weaknesses
        cognitive_assessment = await self._assess_cognitive_capabilities()
        
        # Generate insights about thinking patterns
        thinking_insights = await self._generate_thinking_insights()
        
        # Self-reflection on self-reflection (recursive introspection)
        if self.recursive_depth < self.max_recursive_depth:
            self.recursive_depth += 1
            self.awareness_stack.append("introspecting about introspection")
            
            recursive_insights = await self._recursive_introspection()
            self.recursive_depth -= 1
            self.awareness_stack.pop()
        else:
            recursive_insights = {"note": "Maximum recursive depth reached"}
            
        return {
            'current_state': {
                'dominant_state': current_snapshot.dominant_state.value,
                'confidence': current_snapshot.state_confidence,
                'coherence': current_snapshot.thought_stream_coherence,
                'self_awareness': current_snapshot.self_awareness_level
            },
            'state_patterns': state_patterns,
            'cognitive_assessment': cognitive_assessment,
            'thinking_insights': thinking_insights,
            'recursive_insights': recursive_insights,
            'introspection_meta': {
                'level': self.introspection_level.value,
                'recursive_depth': self.recursive_depth,
                'awareness_stack': self.awareness_stack.copy()
            }
        }
        
    def _analyze_state_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in mental state history"""
        
        if len(self.mental_state_history) < 5:
            return {'note': 'Insufficient history for pattern analysis'}
            
        states = [s.dominant_state for s in self.mental_state_history]
        
        # Most common states
        state_counts = defaultdict(int)
        for state in states:
            state_counts[state] += 1
            
        most_common = max(state_counts.items(), key=lambda x: x[1])
        
        # State transitions
        transitions = []
        for i in range(1, len(states)):
            transitions.append((states[i-1], states[i]))
            
        transition_counts = defaultdict(int)
        for transition in transitions:
            transition_counts[transition] += 1
            
        most_common_transition = max(transition_counts.items(), key=lambda x: x[1]) if transition_counts else None
        
        return {
            'most_common_state': most_common[0].value,
            'most_common_frequency': most_common[1] / len(states),
            'most_common_transition': f"{most_common_transition[0][0].value} -> {most_common_transition[0][1].value}" if most_common_transition else None,
            'state_diversity': len(set(states)) / len(MentalState),
            'recent_trend': states[-3:] if len(states) >= 3 else states
        }
        
    async def _assess_cognitive_capabilities(self) -> Dict[str, Any]:
        """Assess current cognitive capabilities"""
        
        capabilities = {}
        
        # Assess working memory capacity
        if self.mental_state_history:
            avg_load = np.mean([s.working_memory_load for s in self.mental_state_history[-10:]])
            capabilities['working_memory'] = 'high' if avg_load < 0.5 else 'moderate' if avg_load < 0.8 else 'strained'
            
        # Assess associative thinking
        if self.mental_state_history:
            avg_association = np.mean([s.association_strength for s in self.mental_state_history[-10:]])
            capabilities['associative_thinking'] = 'strong' if avg_association > 0.7 else 'moderate' if avg_association > 0.4 else 'weak'
            
        # Assess coherence maintenance
        if self.mental_state_history:
            avg_coherence = np.mean([s.thought_stream_coherence for s in self.mental_state_history[-10:]])
            capabilities['coherence_maintenance'] = 'excellent' if avg_coherence > 0.8 else 'good' if avg_coherence > 0.6 else 'needs_improvement'
            
        return capabilities
        
    async def _generate_thinking_insights(self) -> List[str]:
        """Generate insights about thinking patterns"""
        
        insights = []
        
        if not self.mental_state_history:
            return ["Insufficient data for thinking insights"]
            
        # Analyze recent state patterns
        recent_states = [s.dominant_state for s in self.mental_state_history[-5:]]
        
        if MentalState.FOCUSED in recent_states and MentalState.CREATIVE in recent_states:
            insights.append("Demonstrates ability to switch between focused and creative thinking modes")
            
        if recent_states.count(MentalState.STUCK) > 2:
            insights.append("Tendency to get stuck - may need better strategy switching")
            
        if all(s == recent_states[0] for s in recent_states):
            insights.append("Very consistent mental state - may indicate strong focus or lack of flexibility")
            
        # Analyze attention patterns
        if self.mental_state_history:
            attention_loads = [s.working_memory_load for s in self.mental_state_history[-10:]]
            if np.std(attention_loads) > 0.3:
                insights.append("Highly variable attention usage - dynamic resource allocation")
            else:
                insights.append("Stable attention usage patterns")
                
        return insights
        
    async def _recursive_introspection(self) -> Dict[str, Any]:
        """Perform recursive introspection - thinking about thinking about thinking"""
        
        return {
            'observation': f"Currently aware of being aware of {self.awareness_stack[-1] if self.awareness_stack else 'base state'}",
            'recursive_depth': self.recursive_depth,
            'meta_observation': f"Performing level-{self.recursive_depth} introspection",
            'recursive_limit_note': f"Recursion limited to depth {self.max_recursive_depth} to prevent infinite loops"
        }