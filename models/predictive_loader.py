"""
Predictive Model Loader - Anticipates model needs based on learned patterns
Similar to how brain regions recruit other regions
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelUsagePattern:
    """Tracks usage patterns for predictive loading"""
    model_id: str
    context_vector: np.ndarray
    timestamp: float
    followed_by: List[Tuple[str, float]] = field(default_factory=list)  # (model_id, time_delta)
    preceded_by: List[Tuple[str, float]] = field(default_factory=list)
    co_occurred_with: Set[str] = field(default_factory=set)
    performance_score: float = 1.0


@dataclass
class PredictionCandidate:
    """Candidate model for predictive loading"""
    model_id: str
    confidence: float
    urgency: float  # How soon it's likely needed
    memory_required: int
    context_similarity: float
    
    @property
    def priority_score(self) -> float:
        """Calculate overall priority for loading"""
        return (self.confidence * 0.4 + 
                (1.0 / (self.urgency + 1)) * 0.3 +
                self.context_similarity * 0.3)


class PredictiveModelLoader:
    """
    Learns patterns over time and predictively loads models
    before they're explicitly needed - creating neural plasticity
    """
    
    def __init__(self, kernel=None, memory_limit_mb: int = 8192):
        self.kernel = kernel
        self.memory_limit = memory_limit_mb
        self.current_memory_usage = 0
        
        # Pattern tracking
        self.usage_history: deque = deque(maxlen=1000)
        self.sequence_patterns: Dict[str, List[ModelUsagePattern]] = {}
        self.transition_matrix: Dict[Tuple[str, str], float] = {}
        self.context_associations: Dict[str, List[np.ndarray]] = {}
        
        # Predictive state
        self.loaded_models: Dict[str, float] = {}  # model_id -> load_time
        self.pending_loads: Set[str] = set()
        self.prediction_accuracy: deque = deque(maxlen=100)
        
        # Learning parameters
        self.learning_rate = 0.01
        self.context_window = 5  # Look at last 5 models
        self.prediction_horizon = 3  # Predict next 3 models
        
    async def initialize(self):
        """Initialize the predictive loader"""
        logger.info("Initializing Predictive Model Loader")
        # Start background tasks
        asyncio.create_task(self._pattern_learning_loop())
        asyncio.create_task(self._predictive_loading_loop())
        
    def record_model_usage(self, model_id: str, context: Dict[str, Any]):
        """Record a model usage event for pattern learning"""
        timestamp = time.time()
        
        # Create context vector from current state
        context_vector = self._extract_context_vector(context)
        
        # Create usage pattern
        pattern = ModelUsagePattern(
            model_id=model_id,
            context_vector=context_vector,
            timestamp=timestamp
        )
        
        # Update sequence patterns
        if len(self.usage_history) > 0:
            last_pattern = self.usage_history[-1]
            time_delta = timestamp - last_pattern.timestamp
            
            # Update followed_by for last pattern
            last_pattern.followed_by.append((model_id, time_delta))
            
            # Update preceded_by for current pattern
            pattern.preceded_by.append((last_pattern.model_id, time_delta))
            
            # Update transition matrix
            transition_key = (last_pattern.model_id, model_id)
            if transition_key in self.transition_matrix:
                # Exponential moving average
                old_prob = self.transition_matrix[transition_key]
                self.transition_matrix[transition_key] = (
                    old_prob * (1 - self.learning_rate) + self.learning_rate
                )
            else:
                self.transition_matrix[transition_key] = self.learning_rate
                
        # Add to history
        self.usage_history.append(pattern)
        
        # Update context associations
        if model_id not in self.context_associations:
            self.context_associations[model_id] = []
        self.context_associations[model_id].append(context_vector)
        
        # Track co-occurrences within window
        window_start = max(0, len(self.usage_history) - self.context_window)
        for i in range(window_start, len(self.usage_history) - 1):
            self.usage_history[i].co_occurred_with.add(model_id)
            pattern.co_occurred_with.add(self.usage_history[i].model_id)
            
        logger.debug(f"Recorded usage of {model_id}")
        
    def _extract_context_vector(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from context"""
        # This is a simplified version - in practice would use embeddings
        features = []
        
        # Add basic features
        features.append(context.get('task_type', 0))
        features.append(context.get('data_size', 0))
        features.append(context.get('urgency', 0.5))
        features.append(time.time() % 86400 / 86400)  # Time of day
        
        # Pad to standard size
        while len(features) < 128:
            features.append(0.0)
            
        vector = np.array(features[:128], dtype=np.float32)
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
        
    async def predict_next_models(self, current_context: Dict[str, Any],
                                 n_predictions: int = 3) -> List[PredictionCandidate]:
        """Predict which models will be needed next"""
        predictions = []
        context_vector = self._extract_context_vector(current_context)
        
        # Get recent history for sequence prediction
        recent_models = [p.model_id for p in list(self.usage_history)[-self.context_window:]]
        
        # Calculate predictions based on multiple factors
        model_scores: Dict[str, float] = {}
        
        # 1. Transition probability from recent models
        for recent_model in recent_models[-2:]:  # Look at last 2 models
            for candidate_model, trans_prob in self.transition_matrix.items():
                if candidate_model[0] == recent_model:
                    next_model = candidate_model[1]
                    if next_model not in model_scores:
                        model_scores[next_model] = 0
                    model_scores[next_model] += trans_prob * 0.4
                    
        # 2. Context similarity
        for model_id, context_history in self.context_associations.items():
            if context_history:
                # Average similarity to historical contexts
                similarities = [
                    np.dot(context_vector, hist_context)
                    for hist_context in context_history[-10:]  # Last 10 contexts
                ]
                avg_similarity = np.mean(similarities)
                
                if model_id not in model_scores:
                    model_scores[model_id] = 0
                model_scores[model_id] += avg_similarity * 0.3
                
        # 3. Co-occurrence patterns
        if recent_models:
            recent_set = set(recent_models)
            for pattern in list(self.usage_history)[-20:]:
                if pattern.model_id in recent_set:
                    for co_model in pattern.co_occurred_with:
                        if co_model not in model_scores:
                            model_scores[co_model] = 0
                        model_scores[co_model] += 0.1
                        
        # 4. Temporal patterns (time-based predictions)
        current_time = time.time()
        for pattern in list(self.usage_history)[-50:]:
            time_diff = current_time - pattern.timestamp
            if time_diff < 300:  # Within 5 minutes
                weight = 1.0 - (time_diff / 300)
                if pattern.model_id not in model_scores:
                    model_scores[pattern.model_id] = 0
                model_scores[pattern.model_id] += weight * 0.2
                
        # Convert to prediction candidates
        for model_id, score in model_scores.items():
            if model_id not in self.loaded_models:  # Don't predict already loaded
                candidate = PredictionCandidate(
                    model_id=model_id,
                    confidence=min(1.0, score),
                    urgency=1.0,  # Simplified - would calculate based on patterns
                    memory_required=self._estimate_memory(model_id),
                    context_similarity=self._calculate_context_similarity(
                        context_vector, model_id
                    )
                )
                predictions.append(candidate)
                
        # Sort by priority and return top N
        predictions.sort(key=lambda x: x.priority_score, reverse=True)
        return predictions[:n_predictions]
        
    def _calculate_context_similarity(self, context_vector: np.ndarray,
                                     model_id: str) -> float:
        """Calculate similarity between current context and model's historical contexts"""
        if model_id not in self.context_associations:
            return 0.0
            
        contexts = self.context_associations[model_id]
        if not contexts:
            return 0.0
            
        # Average similarity to recent contexts
        recent_contexts = contexts[-5:]
        similarities = [np.dot(context_vector, ctx) for ctx in recent_contexts]
        return np.mean(similarities)
        
    def _estimate_memory(self, model_id: str) -> int:
        """Estimate memory requirement for a model"""
        # In practice, would look up actual model sizes
        # For now, return a dummy value
        return 512  # MB
        
    async def _pattern_learning_loop(self):
        """Background task for continuous pattern learning"""
        while True:
            await asyncio.sleep(10)  # Learn every 10 seconds
            
            # Analyze recent patterns
            if len(self.usage_history) > 10:
                self._update_patterns()
                
    def _update_patterns(self):
        """Update learned patterns from recent history"""
        # Decay old transition probabilities
        for key in self.transition_matrix:
            self.transition_matrix[key] *= 0.99
            
        # Reinforce recent transitions
        for i in range(len(self.usage_history) - 1):
            curr = self.usage_history[i].model_id
            next_model = self.usage_history[i + 1].model_id
            key = (curr, next_model)
            
            if key in self.transition_matrix:
                self.transition_matrix[key] = min(
                    1.0,
                    self.transition_matrix[key] + self.learning_rate
                )
                
    async def _predictive_loading_loop(self):
        """Background task for predictive model loading"""
        while True:
            await asyncio.sleep(2)  # Check every 2 seconds
            
            # Get current context from kernel if available
            context = {}
            if self.kernel:
                context = {
                    'active_models': len([m for m in self.kernel.models.values() 
                                        if m.state == 'ACTIVE']),
                    'attention_available': self.kernel.available_attention
                }
                
            # Get predictions
            predictions = await self.predict_next_models(context)
            
            # Load high-confidence predictions
            for prediction in predictions:
                if prediction.confidence > 0.6:
                    await self._preload_model(prediction.model_id)
                    
    async def _preload_model(self, model_id: str):
        """Preload a model into memory"""
        if model_id in self.loaded_models or model_id in self.pending_loads:
            return
            
        self.pending_loads.add(model_id)
        logger.info(f"Preloading model {model_id}")
        
        # Simulate loading
        await asyncio.sleep(0.5)
        
        self.loaded_models[model_id] = time.time()
        self.pending_loads.remove(model_id)
        
    def evaluate_prediction_accuracy(self, model_id: str) -> float:
        """Evaluate how accurate our prediction was"""
        if model_id in self.loaded_models:
            load_time = self.loaded_models[model_id]
            current_time = time.time()
            time_since_load = current_time - load_time
            
            # Good prediction if used within 10 seconds of loading
            if time_since_load < 10:
                accuracy = 1.0
            elif time_since_load < 30:
                accuracy = 0.5
            else:
                accuracy = 0.0
                
            self.prediction_accuracy.append(accuracy)
            return accuracy
        return 0.0
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictive loading statistics"""
        avg_accuracy = np.mean(self.prediction_accuracy) if self.prediction_accuracy else 0
        
        return {
            'loaded_models': len(self.loaded_models),
            'pending_loads': len(self.pending_loads),
            'patterns_learned': len(self.transition_matrix),
            'prediction_accuracy': avg_accuracy,
            'memory_usage': self.current_memory_usage,
            'history_size': len(self.usage_history)
        }