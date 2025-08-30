"""
GAIA Demo - Showcasing cognitive architecture capabilities
"""

import asyncio
import numpy as np
import time
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import GAIA components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel.core import GAIAKernel
from kernel.attention import AttentionManager
from orchestrator.async_executor import AsyncThoughtExecutor
from memory.semantic_graph import SemanticGraph
from models.predictive_loader import PredictiveModelLoader
from models.pattern_learner import PatternLearner, State


class GAIADemo:
    """Demonstration of GAIA's cognitive capabilities"""
    
    def __init__(self):
        self.kernel = GAIAKernel()
        self.attention_manager = AttentionManager(total_tokens=100)
        self.executor = AsyncThoughtExecutor(kernel=self.kernel)
        self.semantic_graph = SemanticGraph()
        self.predictive_loader = PredictiveModelLoader(kernel=self.kernel)
        self.pattern_learner = PatternLearner()
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("=" * 60)
        logger.info("Initializing GAIA - General AI Architecture")
        logger.info("=" * 60)
        
        await self.kernel.initialize()
        await self.predictive_loader.initialize()
        
        # Register some demo models
        models = [
            ("language_model", "language", 2048, 4096),
            ("vision_model", "vision", 1024, 2048),
            ("reasoning_model", "reasoning", 512, 1024),
            ("memory_model", "memory", 256, 512),
            ("planning_model", "planning", 512, 1024),
            ("analysis_model", "analysis", 384, 768)
        ]
        
        for model_id, model_type, memory, vram in models:
            self.kernel.register_model(model_id, model_type, memory, vram, {model_type})
            
            # Add to semantic graph with random embeddings
            embedding = np.random.randn(128)
            embedding = embedding / np.linalg.norm(embedding)
            self.semantic_graph.add_node(model_id, model_type, embedding)
            
        logger.info(f"Registered {len(models)} models")
        
    async def demo_async_branching(self):
        """Demonstrate async thought branching"""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO 1: Asynchronous Thought Branching")
        logger.info("=" * 60)
        
        # Start main thought
        main_thought = await self.executor.spawn_thought(
            "reasoning_model",
            {"task": "Analyze climate change impacts", "depth": "comprehensive"}
        )
        logger.info(f"Spawned main thought: {main_thought}")
        
        # Let it run and spawn associations
        await asyncio.sleep(1)
        
        # Check thought graph
        graph_viz = self.executor.visualize_thought_graph()
        logger.info(f"Thought graph has {len(graph_viz['nodes'])} nodes")
        
        for node in graph_viz['nodes']:
            logger.info(f"  - {node['id']}: {node['label']} [{node['state']}]")
            
        # Read partial outputs from running thoughts
        for thought_id in self.executor.thought_graph:
            partial = await self.executor.read_partial_output(thought_id, timeout=0.1)
            if partial:
                logger.info(f"Partial output from {thought_id}: {partial}")
                
        # Wait for completion
        result = await self.executor.wait_for_thought(main_thought)
        logger.info(f"Main thought completed with result: {result}")
        
    async def demo_attention_competition(self):
        """Demonstrate attention as scarce resource"""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO 2: Attention Competition")
        logger.info("=" * 60)
        
        # Multiple models compete for attention
        requests = []
        
        # High priority request
        high_priority = await self.attention_manager.request_attention(
            "reasoning_model", 50, priority=0.9
        )
        logger.info("High priority model requested 50 tokens")
        
        # Medium priority requests
        for i in range(3):
            req = await self.attention_manager.request_attention(
                f"analysis_model_{i}", 30, priority=0.5
            )
            requests.append(req)
            logger.info(f"Medium priority model {i} requested 30 tokens")
            
        # Check status
        status = self.attention_manager.get_status()
        logger.info(f"\nAttention status: {status}")
        
        # Release some attention
        await self.attention_manager.release_attention("reasoning_model", 25)
        logger.info("Released 25 tokens from reasoning_model")
        
        # Check which pending requests got satisfied
        await asyncio.sleep(0.1)
        status = self.attention_manager.get_status()
        logger.info(f"Updated status: {status}")
        
    async def demo_semantic_associations(self):
        """Demonstrate semantic graph associations"""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO 3: Semantic Associations")
        logger.info("=" * 60)
        
        # Create associations
        self.kernel.create_association("language_model", "reasoning_model", 0.8)
        self.kernel.create_association("reasoning_model", "planning_model", 0.7)
        self.kernel.create_association("vision_model", "analysis_model", 0.6)
        
        # Activate a node and watch propagation
        logger.info("\nActivating 'reasoning_model' with propagation...")
        self.semantic_graph.activate_node("reasoning_model", strength=1.0, propagate=True)
        
        # Check activation pattern
        activation = self.semantic_graph.get_activation_pattern()
        for node_id, level in activation.items():
            if level > 0:
                logger.info(f"  {node_id}: activation level {level:.3f}")
                
        # Find associative path
        path = self.semantic_graph.find_path("language_model", "planning_model")
        if path:
            logger.info(f"\nAssociative path: {' -> '.join(path)}")
            
        # Create temporal association
        self.semantic_graph.create_temporal_association("vision_model", "memory_model")
        logger.info("Created temporal association: vision_model <-> memory_model")
        
    async def demo_predictive_loading(self):
        """Demonstrate predictive model loading"""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO 4: Predictive Model Loading")
        logger.info("=" * 60)
        
        # Simulate usage pattern
        sequence = ["language_model", "reasoning_model", "planning_model", 
                   "language_model", "analysis_model"]
        
        for model_id in sequence:
            context = {
                "task_type": 1 if "language" in model_id else 2,
                "data_size": np.random.randint(100, 1000),
                "urgency": np.random.random()
            }
            
            self.predictive_loader.record_model_usage(model_id, context)
            logger.info(f"Recorded usage: {model_id}")
            await asyncio.sleep(0.1)
            
        # Get predictions
        current_context = {
            "task_type": 1,
            "data_size": 500,
            "urgency": 0.7
        }
        
        predictions = await self.predictive_loader.predict_next_models(current_context, 3)
        logger.info("\nPredicted next models:")
        for pred in predictions:
            logger.info(f"  - {pred.model_id}: confidence={pred.confidence:.3f}, "
                       f"priority={pred.priority_score:.3f}")
            
        # Check pattern learning statistics
        stats = self.predictive_loader.get_statistics()
        logger.info(f"\nPredictive loader stats: {stats}")
        
    async def demo_reinforcement_learning(self):
        """Demonstrate reinforcement learning for patterns"""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO 5: Reinforcement Learning")
        logger.info("=" * 60)
        
        # Create state
        state = State(
            active_models={"language_model", "reasoning_model"},
            context_vector=np.random.randn(128),
            available_memory=4096,
            available_attention=50,
            time_of_day=0.5,
            recent_sequence=["language_model"]
        )
        
        # Get action recommendation
        available_models = ["planning_model", "analysis_model", "memory_model"]
        action = self.pattern_learner.select_action(state, available_models)
        
        logger.info(f"RL recommended loading: {action.models_to_load}")
        
        # Simulate outcome
        outcome = {
            "models_used": action.models_to_load[:1],  # First model was used
            "time_to_use": {action.models_to_load[0]: 3.0} if action.models_to_load else {},
            "memory_waste": 100,
            "system_performance": 0.8
        }
        
        # Calculate reward
        reward = self.pattern_learner.calculate_reward(action, outcome)
        logger.info(f"Action received reward: {reward:.2f}")
        
        # Update Q-values
        next_state = State(
            active_models={"planning_model"},
            context_vector=np.random.randn(128),
            available_memory=3000,
            available_attention=30,
            time_of_day=0.51,
            recent_sequence=["reasoning_model", "planning_model"]
        )
        
        self.pattern_learner.update_q_values(state, action, reward, next_state)
        
        # Get recommendations
        recommendations = self.pattern_learner.get_model_recommendations(state, 3)
        logger.info("\nRL model recommendations:")
        for model, confidence in recommendations:
            logger.info(f"  - {model}: confidence={confidence:.3f}")
            
    async def demo_thought_merging(self):
        """Demonstrate merging parallel thought streams"""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO 6: Thought Stream Merging")
        logger.info("=" * 60)
        
        # Spawn multiple parallel thoughts
        thoughts = []
        topics = [
            ("language_model", {"analyze": "syntax"}),
            ("reasoning_model", {"analyze": "logic"}),
            ("analysis_model", {"analyze": "patterns"})
        ]
        
        for model, input_data in topics:
            thought_id = await self.executor.spawn_thought(model, input_data)
            thoughts.append(thought_id)
            logger.info(f"Spawned thought {thought_id} for {model}")
            
        # Let them run in parallel
        await asyncio.sleep(0.5)
        
        # Merge results
        merged = await self.executor.merge_thought_streams(thoughts)
        logger.info(f"\nMerged {len(thoughts)} thought streams")
        logger.info(f"Merged result: {merged}")
        
    async def run_full_demo(self):
        """Run all demonstrations"""
        await self.initialize()
        
        # Run each demo
        await self.demo_async_branching()
        await self.demo_attention_competition()
        await self.demo_semantic_associations()
        await self.demo_predictive_loading()
        await self.demo_reinforcement_learning()
        await self.demo_thought_merging()
        
        # Final statistics
        logger.info("\n" + "=" * 60)
        logger.info("GAIA Demo Complete - Final Statistics")
        logger.info("=" * 60)
        
        kernel_status = self.kernel.get_kernel_status()
        logger.info(f"Kernel status: {kernel_status}")
        
        logger.info("\nGAIA has demonstrated:")
        logger.info("  ✓ Async parallel thought branching")
        logger.info("  ✓ Attention as scarce resource")
        logger.info("  ✓ Semantic associations & activation propagation")
        logger.info("  ✓ Predictive model loading")
        logger.info("  ✓ Reinforcement learning for patterns")
        logger.info("  ✓ Thought stream merging")
        

async def main():
    """Main entry point"""
    demo = GAIADemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())