#!/usr/bin/env python3
"""
Demonstration and testing script for ModelMemoryManager
Shows integration with cognitive platform and intelligent model management
"""

import asyncio
import logging
import time
from typing import Dict, Any
import json

from .model_memory_manager import ModelMemoryManager, MemoryPressure
from .integration_utils import CognitiveModelIntegrator, ModelMemoryMiddleware, ModelHealthMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockCognitiveKernel:
    """Mock kernel for demonstration purposes"""
    
    def __init__(self):
        self.requests_made = 0
        
    async def generate(self, provider_name: str, prompt: str, model: str = None, **kwargs):
        """Mock generate method"""
        self.requests_made += 1
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return type('MockResponse', (), {
            'content': f"Mock response for {prompt[:30]}...",
            'tokens_used': len(prompt.split()) + 10,
            'model': model or 'default-model',
            'provider': provider_name
        })()


class MockLearningStrategist:
    """Mock learning strategist for demonstration"""
    
    async def select_optimal_strategy(self, context: Dict[str, Any]):
        """Mock strategy selection"""
        complexity = context.get('complexity', 0.5)
        
        if complexity > 0.7:
            model_sequence = ["reasoning", "analysis", "synthesis"]
        elif complexity > 0.4:
            model_sequence = ["reasoning", "synthesis"]
        else:
            model_sequence = ["reasoning"]
        
        return type('MockStrategy', (), {
            'strategy_id': f"mock_strategy_{int(time.time())}",
            'model_sequence': model_sequence,
            'complexity_score': complexity
        })()


async def demonstrate_basic_functionality():
    """Demonstrate basic ModelMemoryManager functionality"""
    logger.info("=== Basic Functionality Demo ===")
    
    # Initialize memory manager
    memory_manager = ModelMemoryManager(
        max_memory_usage_percent=75.0,
        model_timeout_hours=1.0,
        min_free_memory_gb=2.0
    )
    
    try:
        # Start monitoring
        await memory_manager.start_monitoring()
        
        # Get initial stats
        stats = memory_manager.get_memory_stats()
        logger.info(f"Initial memory stats: {json.dumps(stats, indent=2)}")
        
        # Discover available models
        logger.info("Discovering loaded models...")
        loaded_models = await memory_manager.get_loaded_models()
        logger.info(f"Currently loaded models: {loaded_models}")
        
        # Test model loading
        if loaded_models:
            test_model = loaded_models[0]
            logger.info(f"Testing with model: {test_model}")
            
            # Record some usage
            for i in range(5):
                latency = 100 + (i * 20)  # Simulate varying latency
                success = i < 4  # Simulate one failure
                memory_manager.record_model_usage(test_model, latency, success, tokens_used=50)
                await asyncio.sleep(0.1)
            
            # Get updated stats
            stats = memory_manager.get_memory_stats()
            logger.info(f"Updated stats after usage: {json.dumps(stats, indent=2)}")
        
        # Test memory optimization
        logger.info("Testing memory optimization...")
        optimization_result = await memory_manager.optimize_memory_usage()
        logger.info(f"Optimization result: {json.dumps(optimization_result, indent=2)}")
        
        # Test model recommendations
        test_context = {
            "task_type": "coding",
            "complexity": 0.8,
            "urgency": 0.3,
            "available_memory_gb": 8.0
        }
        
        recommendations = memory_manager.get_model_recommendations(test_context)
        logger.info(f"Model recommendations for context {test_context}:")
        logger.info(json.dumps(recommendations, indent=2))
        
    finally:
        await memory_manager.stop_monitoring()
        await memory_manager.cleanup()


async def demonstrate_integration():
    """Demonstrate integration with cognitive platform components"""
    logger.info("=== Integration Demo ===")
    
    # Initialize components
    memory_manager = ModelMemoryManager(max_memory_usage_percent=80.0)
    mock_kernel = MockCognitiveKernel()
    mock_strategist = MockLearningStrategist()
    
    # Create integrator
    integrator = CognitiveModelIntegrator(memory_manager)
    integrator.integrate_with_kernel(mock_kernel)
    integrator.integrate_with_strategist(mock_strategist)
    
    try:
        await memory_manager.start_monitoring()
        
        # Test integrated kernel generation
        logger.info("Testing integrated kernel generation...")
        response = await mock_kernel.generate(
            "lmstudio",
            "Write a Python function to calculate fibonacci numbers",
            model="qwen/qwen2.5-coder-14b"
        )
        logger.info(f"Generated response: {response.content}")
        
        # Test integrated strategy selection
        logger.info("Testing integrated strategy selection...")
        context = {
            "complexity": 0.9,
            "task_type": "reasoning",
            "time_pressure": 0.2
        }
        
        strategy = await mock_strategist.select_optimal_strategy(context)
        logger.info(f"Selected strategy: {strategy.strategy_id}, models: {strategy.model_sequence}")
        
        # Check memory stats after integration
        stats = memory_manager.get_memory_stats()
        logger.info(f"Memory stats after integration test: {json.dumps(stats, indent=2)}")
        
    finally:
        await memory_manager.stop_monitoring()
        await memory_manager.cleanup()


async def demonstrate_middleware():
    """Demonstrate middleware functionality"""
    logger.info("=== Middleware Demo ===")
    
    memory_manager = ModelMemoryManager()
    middleware = ModelMemoryMiddleware(memory_manager)
    
    async def mock_request_handler(context: Dict[str, Any]):
        """Mock request handler"""
        await asyncio.sleep(0.2)  # Simulate processing
        context["models_used"] = ["qwen/qwen3-8b", "qwen/qwen2.5-coder-14b"]
        context["processing_time"] = 0.15
        return {"result": "Mock processing complete"}
    
    try:
        await memory_manager.start_monitoring()
        
        # Process several requests through middleware
        for i in range(3):
            request_context = {
                "task_type": "coding" if i % 2 == 0 else "reasoning",
                "complexity": 0.3 + (i * 0.2),
                "request_id": f"req_{i+1}"
            }
            
            logger.info(f"Processing request {i+1}: {request_context}")
            
            result = await middleware(request_context, mock_request_handler)
            logger.info(f"Request {i+1} result: {result}")
            
            await asyncio.sleep(0.5)
        
        # Check final stats
        stats = memory_manager.get_memory_stats()
        logger.info(f"Final middleware stats: {json.dumps(stats, indent=2)}")
        
    finally:
        await memory_manager.stop_monitoring()
        await memory_manager.cleanup()


async def demonstrate_health_monitoring():
    """Demonstrate health monitoring functionality"""
    logger.info("=== Health Monitoring Demo ===")
    
    memory_manager = ModelMemoryManager()
    health_monitor = ModelHealthMonitor(memory_manager)
    
    try:
        # Start both monitoring systems
        await memory_manager.start_monitoring()
        await health_monitor.start_monitoring(interval=5.0)  # Check every 5 seconds
        
        # Simulate some activity
        loaded_models = await memory_manager.get_loaded_models()
        if loaded_models:
            test_model = loaded_models[0]
            
            # Simulate various usage patterns
            for i in range(10):
                # Vary success rates and latency
                success = i < 8  # 80% success rate
                latency = 50 + (i * 10) + (0 if success else 200)  # Higher latency for failures
                
                memory_manager.record_model_usage(test_model, latency, success, tokens_used=30)
                await asyncio.sleep(0.8)
        
        # Get health report
        health_report = health_monitor.get_health_report()
        logger.info("Health Report:")
        logger.info(json.dumps(health_report, indent=2))
        
        # Wait a bit more to see monitoring in action
        logger.info("Monitoring for 10 more seconds...")
        await asyncio.sleep(10)
        
        # Final health report
        final_report = health_monitor.get_health_report()
        logger.info("Final Health Report:")
        logger.info(json.dumps(final_report, indent=2))
        
    finally:
        await health_monitor.stop_monitoring()
        await memory_manager.stop_monitoring()
        await memory_manager.cleanup()


async def demonstrate_advanced_scenarios():
    """Demonstrate advanced usage scenarios"""
    logger.info("=== Advanced Scenarios Demo ===")
    
    memory_manager = ModelMemoryManager(
        max_memory_usage_percent=60.0,  # Lower threshold for demo
        model_timeout_hours=0.1,  # Short timeout for demo
        min_free_memory_gb=1.0
    )
    
    try:
        await memory_manager.start_monitoring()
        
        # Scenario 1: Multi-agent task requiring multiple models
        logger.info("Scenario 1: Multi-agent task")
        multi_agent_models = ["qwen/qwen3-8b", "qwen/qwen2.5-coder-14b", "deepseek/deepseek-r1-0528-qwen3-8b"]
        
        success = await memory_manager.ensure_models_available(multi_agent_models, priority=9)
        logger.info(f"Multi-agent models availability: {success}")
        
        # Simulate concurrent usage
        tasks = []
        for model in multi_agent_models:
            task = asyncio.create_task(
                simulate_model_usage(memory_manager, model, requests=5)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Scenario 2: Memory pressure response
        logger.info("Scenario 2: Memory pressure response")
        
        # Simulate memory pressure by loading many models (mock)
        from .model_memory_manager import ModelStatus, ModelMetrics
        
        for i in range(5):
            mock_model = f"mock_model_{i}"
            memory_manager.model_status[mock_model] = ModelStatus.LOADED
            memory_manager.model_metrics[mock_model] = ModelMetrics(mock_model)
            memory_manager.model_metrics[mock_model].estimated_memory_mb = 2000  # 2GB each
        
        # Force optimization
        optimization = await memory_manager.optimize_memory_usage()
        logger.info(f"Memory pressure optimization: {json.dumps(optimization, indent=2)}")
        
        # Scenario 3: Adaptive model selection
        logger.info("Scenario 3: Adaptive model selection")
        
        contexts = [
            {"task_type": "coding", "complexity": 0.9, "urgency": 0.8},
            {"task_type": "reasoning", "complexity": 0.3, "urgency": 0.2},
            {"task_type": "creative", "complexity": 0.7, "urgency": 0.5}
        ]
        
        for i, context in enumerate(contexts):
            recommendations = memory_manager.get_model_recommendations(context)
            logger.info(f"Context {i+1} recommendations: {json.dumps(recommendations, indent=2)}")
        
    finally:
        await memory_manager.stop_monitoring()
        await memory_manager.cleanup()


async def simulate_model_usage(memory_manager: ModelMemoryManager, model_name: str, requests: int = 5):
    """Simulate usage of a specific model"""
    for i in range(requests):
        # Vary latency and success rate
        latency = 80 + (i * 15)
        success = i < requests - 1  # One failure
        tokens = 20 + (i * 10)
        
        memory_manager.record_model_usage(model_name, latency, success, tokens)
        await asyncio.sleep(0.3)


async def main():
    """Run all demonstrations"""
    logger.info("Starting ModelMemoryManager Comprehensive Demo")
    logger.info("=" * 60)
    
    demos = [
        ("Basic Functionality", demonstrate_basic_functionality),
        ("Integration", demonstrate_integration),
        ("Middleware", demonstrate_middleware),
        ("Health Monitoring", demonstrate_health_monitoring),
        ("Advanced Scenarios", demonstrate_advanced_scenarios)
    ]
    
    for name, demo_func in demos:
        try:
            logger.info(f"\n{'='*20} {name} {'='*20}")
            await demo_func()
            logger.info(f"✓ {name} demo completed successfully")
            
            # Brief pause between demos
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"✗ {name} demo failed: {e}", exc_info=True)
    
    logger.info("\n" + "="*60)
    logger.info("ModelMemoryManager Demo Complete!")


if __name__ == "__main__":
    asyncio.run(main())