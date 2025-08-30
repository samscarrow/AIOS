"""
Pattern Learner - Reinforcement learning for model loading patterns
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import time
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class State:
    """Represents the current state for RL"""
    active_models: Set[str]
    context_vector: np.ndarray
    available_memory: int
    available_attention: int
    time_of_day: float
    recent_sequence: List[str]
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector"""
        features = []
        
        # Binary encoding of active models (simplified)
        model_features = [1 if m in self.active_models else 0 
                         for m in list(self.active_models)[:10]]
        features.extend(model_features)
        
        # Context vector
        features.extend(self.context_vector[:32])
        
        # Resources
        features.append(self.available_memory / 10000)
        features.append(self.available_attention / 100)
        features.append(self.time_of_day)
        
        # Pad to fixed size
        while len(features) < 64:
            features.append(0)
            
        return np.array(features[:64], dtype=np.float32)


@dataclass  
class Action:
    """Action to take - which model(s) to load"""
    models_to_load: List[str]
    models_to_unload: List[str]
    priority_adjustments: Dict[str, float]


class PatternLearner:
    """
    Uses reinforcement learning to learn optimal model loading patterns
    """
    
    def __init__(self, learning_rate: float = 0.01, gamma: float = 0.95):
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        
        # Q-table approximation using function approximation
        self.weights = np.random.randn(64, 32) * 0.01  # State features -> action values
        
        # Experience replay buffer
        self.experience_buffer = []
        self.max_buffer_size = 1000
        
        # Exploration parameters
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Reward tracking
        self.total_reward = 0
        self.episode_rewards = []
        
        # Model performance tracking
        self.model_performance: Dict[str, float] = defaultdict(float)
        
    def get_state_value(self, state: State) -> float:
        """Get value of current state"""
        state_vector = state.to_vector()
        value = np.max(np.dot(state_vector, self.weights))
        return value
        
    def select_action(self, state: State, available_models: List[str]) -> Action:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore: random action
            n_models = np.random.randint(0, min(3, len(available_models)))
            models_to_load = np.random.choice(available_models, n_models, replace=False).tolist()
            return Action(
                models_to_load=models_to_load,
                models_to_unload=[],
                priority_adjustments={}
            )
        else:
            # Exploit: choose best action based on Q-values
            state_vector = state.to_vector()
            action_values = np.dot(state_vector, self.weights)
            
            # Map action values to models
            model_scores = {}
            for i, model in enumerate(available_models[:32]):
                if i < len(action_values):
                    model_scores[model] = action_values[i]
                    
            # Select top scoring models
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            models_to_load = [m for m, s in sorted_models[:3] if s > 0]
            
            return Action(
                models_to_load=models_to_load,
                models_to_unload=[],
                priority_adjustments={m: s for m, s in sorted_models[:5]}
            )
            
    def update_q_values(self, state: State, action: Action, reward: float,
                       next_state: State):
        """Update Q-values using TD learning"""
        state_vector = state.to_vector()
        next_state_vector = next_state.to_vector()
        
        # Calculate TD error
        current_q = np.max(np.dot(state_vector, self.weights))
        next_q = np.max(np.dot(next_state_vector, self.weights))
        td_error = reward + self.gamma * next_q - current_q
        
        # Update weights
        gradient = np.outer(state_vector, np.ones(32))
        self.weights += self.learning_rate * td_error * gradient
        
        # Store experience
        self.experience_buffer.append((state, action, reward, next_state))
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
            
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def calculate_reward(self, action: Action, outcome: Dict[str, Any]) -> float:
        """Calculate reward based on action outcome"""
        reward = 0.0
        
        # Positive reward for models that were used soon after loading
        for model in action.models_to_load:
            if model in outcome.get('models_used', []):
                time_to_use = outcome.get('time_to_use', {}).get(model, float('inf'))
                if time_to_use < 5:  # Used within 5 seconds
                    reward += 10.0
                elif time_to_use < 15:
                    reward += 5.0
                elif time_to_use < 30:
                    reward += 1.0
                else:
                    reward -= 1.0  # Penalty for premature loading
            else:
                reward -= 2.0  # Penalty for unnecessary load
                
        # Penalty for memory waste
        memory_waste = outcome.get('memory_waste', 0)
        reward -= memory_waste / 1000
        
        # Bonus for maintaining good performance
        performance = outcome.get('system_performance', 0.5)
        reward += performance * 5
        
        self.total_reward += reward
        return reward
        
    def replay_experience(self, batch_size: int = 32):
        """Experience replay for more stable learning"""
        if len(self.experience_buffer) < batch_size:
            return
            
        # Sample random batch
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        
        for idx in indices:
            state, action, reward, next_state = self.experience_buffer[idx]
            self.update_q_values(state, action, reward, next_state)
            
    def update_model_performance(self, model_id: str, performance: float):
        """Track individual model performance"""
        # Exponential moving average
        alpha = 0.1
        old_perf = self.model_performance[model_id]
        self.model_performance[model_id] = old_perf * (1 - alpha) + performance * alpha
        
    def get_model_recommendations(self, state: State, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """Get model loading recommendations with confidence scores"""
        state_vector = state.to_vector()
        action_values = np.dot(state_vector, self.weights)
        
        # Create recommendations
        recommendations = []
        available_models = ['model_' + str(i) for i in range(min(32, len(action_values)))]
        
        for i, model in enumerate(available_models):
            if i < len(action_values):
                confidence = 1 / (1 + np.exp(-action_values[i]))  # Sigmoid
                recommendations.append((model, confidence))
                
        # Sort by confidence
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Add performance-based adjustments
        for i, (model, conf) in enumerate(recommendations):
            if model in self.model_performance:
                perf_adjustment = self.model_performance[model] * 0.2
                recommendations[i] = (model, min(1.0, conf + perf_adjustment))
                
        return recommendations[:n_recommendations]
        
    def save_model(self, filepath: str):
        """Save learned weights"""
        np.save(filepath, self.weights)
        logger.info(f"Saved pattern learner model to {filepath}")
        
    def load_model(self, filepath: str):
        """Load learned weights"""
        self.weights = np.load(filepath)
        logger.info(f"Loaded pattern learner model from {filepath}")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'total_reward': self.total_reward,
            'epsilon': self.epsilon,
            'buffer_size': len(self.experience_buffer),
            'avg_episode_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'model_performances': dict(self.model_performance)
        }