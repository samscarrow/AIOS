#!/usr/bin/env python3
"""
Integration utilities for connecting ModelMemoryManager with existing cognitive platform
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from .model_memory_manager import ModelMemoryManager, ModelStatus, MemoryPressure

logger = logging.getLogger(__name__)


class CognitiveModelIntegrator:
    """
    Integrates ModelMemoryManager with the cognitive platform components
    """
    
    def __init__(self, memory_manager: ModelMemoryManager):
        self.memory_manager = memory_manager
        self.kernel = None
        self.learning_strategist = None
        
    def integrate_with_kernel(self, kernel):
        """Integrate with UniversalProviderKernel"""
        self.kernel = kernel
        self.memory_manager.set_kernel(kernel)
        
        # Hook into kernel's generation methods to track model usage
        if hasattr(kernel, 'generate'):
            original_generate = kernel.generate
            kernel.generate = self._wrap_kernel_generate(original_generate)
            logger.info("Integrated ModelMemoryManager with UniversalProviderKernel")
    
    def integrate_with_strategist(self, strategist):
        """Integrate with EnhancedLearningStrategist"""
        self.learning_strategist = strategist
        self.memory_manager.set_learning_strategist(strategist)
        
        # Hook into strategist's strategy selection to ensure required models
        if hasattr(strategist, 'select_optimal_strategy'):
            original_select = strategist.select_optimal_strategy
            strategist.select_optimal_strategy = self._wrap_strategy_selection(original_select)
            logger.info("Integrated ModelMemoryManager with EnhancedLearningStrategist")
    
    def _wrap_kernel_generate(self, original_generate):
        """Wrap kernel generate method to track model usage"""
        async def wrapped_generate(provider_name: str, prompt: str, model: Optional[str] = None, **kwargs):
            import time
            start_time = time.perf_counter()
            
            try:
                # Ensure model is available before generation
                if model and provider_name == "lmstudio":
                    await self.memory_manager.ensure_models_available([model])
                
                # Call original generate method
                result = await original_generate(provider_name, prompt, model, **kwargs)
                
                # Record successful usage
                if model and provider_name == "lmstudio":
                    latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
                    tokens_used = getattr(result, 'tokens_used', 0)
                    self.memory_manager.record_model_usage(model, latency, True, tokens_used)
                
                return result
                
            except Exception as e:
                # Record failed usage
                if model and provider_name == "lmstudio":
                    latency = (time.perf_counter() - start_time) * 1000
                    self.memory_manager.record_model_usage(model, latency, False, 0)
                raise e
        
        return wrapped_generate
    
    def _wrap_strategy_selection(self, original_select):
        """Wrap strategy selection to ensure required models are available"""
        async def wrapped_select(context: Dict[str, Any]):
            # Get model recommendations from memory manager
            recommendations = self.memory_manager.get_model_recommendations(context)
            
            # Preload recommended models if system resources allow
            if recommendations["models_to_preload"]:
                memory_stats = self.memory_manager.get_memory_stats()
                if memory_stats["system_memory"]["pressure_level"] in ["low", "medium"]:
                    await self.memory_manager.ensure_models_available(
                        recommendations["models_to_preload"], 
                        priority=3
                    )
            
            # Call original strategy selection
            strategy = await original_select(context)
            
            # Ensure models required by strategy are available
            if hasattr(strategy, 'model_sequence'):
                required_models = []
                for model_role in strategy.model_sequence:
                    # Map model roles to actual model names
                    actual_model = self._map_role_to_model(model_role, context)
                    if actual_model:
                        required_models.append(actual_model)
                
                if required_models:
                    await self.memory_manager.ensure_models_available(required_models, priority=8)
            
            return strategy
        
        return wrapped_select
    
    def _map_role_to_model(self, model_role: str, context: Dict[str, Any]) -> Optional[str]:
        """Map cognitive model roles to actual model names"""
        # This mapping should be configurable and context-aware
        role_mappings = {
            "reasoning": "qwen/qwen3-8b",
            "analysis": "deepseek/deepseek-r1-0528-qwen3-8b",
            "synthesis": "qwen/qwen3-8b",
            "creative": "mistralai/devstral-small-2507",
            "associative": "qwen/qwen3-8b",
            "introspection": "deepseek/deepseek-r1-0528-qwen3-8b",
            "meta_analysis": "qwen/qwen3-32b",
            "coding": "qwen/qwen2.5-coder-14b",
            "intuition": "qwen/qwen3-8b"
        }
        
        # Context-aware model selection
        complexity = context.get('complexity', 0.5)
        if complexity > 0.8:
            # Use larger models for complex tasks
            high_complexity_mappings = {
                "reasoning": "qwen/qwen3-32b",
                "analysis": "hermes-4-70b@q4_k_m",
                "coding": "qwen/qwen3-coder-30b"
            }
            return high_complexity_mappings.get(model_role) or role_mappings.get(model_role)
        
        return role_mappings.get(model_role)


class ModelMemoryMiddleware:
    """
    Middleware for automatic model management in request processing
    """
    
    def __init__(self, memory_manager: ModelMemoryManager):
        self.memory_manager = memory_manager
        
    async def __call__(self, request_context: Dict[str, Any], next_handler):
        """Process request with automatic model management"""
        try:
            # Pre-request: Analyze and prepare models
            await self._pre_request_optimization(request_context)
            
            # Process request
            result = await next_handler(request_context)
            
            # Post-request: Update metrics and optimize
            await self._post_request_cleanup(request_context, result, success=True)
            
            return result
            
        except Exception as e:
            # Handle errors and update metrics
            await self._post_request_cleanup(request_context, None, success=False)
            raise e
    
    async def _pre_request_optimization(self, context: Dict[str, Any]):
        """Optimize models before processing request"""
        # Get recommendations
        recommendations = self.memory_manager.get_model_recommendations(context)
        
        # Ensure recommended models are available
        if recommendations["recommended_models"]:
            await self.memory_manager.ensure_models_available(
                recommendations["recommended_models"],
                priority=6
            )
        
        # Check memory pressure and optimize if needed
        memory_stats = self.memory_manager.get_memory_stats()
        if memory_stats["system_memory"]["pressure_level"] == "high":
            await self.memory_manager.optimize_memory_usage()
    
    async def _post_request_cleanup(self, context: Dict[str, Any], result: Any, success: bool):
        """Cleanup and optimize after processing request"""
        # Record usage for any models used
        used_models = context.get("models_used", [])
        latency = context.get("processing_time", 0) * 1000  # Convert to ms
        
        for model_name in used_models:
            self.memory_manager.record_model_usage(model_name, latency, success)
        
        # Periodic optimization (every 10th request)
        if hasattr(self, '_request_count'):
            self._request_count += 1
        else:
            self._request_count = 1
            
        if self._request_count % 10 == 0:
            # Background optimization
            asyncio.create_task(self.memory_manager.optimize_memory_usage())


class ModelHealthMonitor:
    """
    Monitors model health and performance
    """
    
    def __init__(self, memory_manager: ModelMemoryManager):
        self.memory_manager = memory_manager
        self.monitoring_task = None
        
    async def start_monitoring(self, interval: float = 60.0):
        """Start health monitoring"""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(
                self._monitor_health(interval)
            )
            logger.info("Started model health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            logger.info("Stopped model health monitoring")
    
    async def _monitor_health(self, interval: float):
        """Background health monitoring task"""
        while True:
            try:
                await asyncio.sleep(interval)
                await self._check_model_health()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in model health monitoring: {e}")
    
    async def _check_model_health(self):
        """Check health of all loaded models"""
        loaded_models = await self.memory_manager.get_loaded_models()
        
        for model_name in loaded_models:
            try:
                # Test model with a simple query
                # In real implementation, would use MCP tools:
                # response = await asyncio.wait_for(
                #     mcp__lmstudio_omnibus__query(
                #         prompt="Test",
                #         context={"preferred_model": model_name}
                #     ),
                #     timeout=10.0
                # )
                
                # Simulate health check for demo
                await asyncio.sleep(0.1)
                response = "test_response"  # Simulate successful response
                
                if response is None:
                    logger.warning(f"Model {model_name} may be unhealthy - no response")
                    # Mark as potential issue
                    if model_name in self.memory_manager.model_status:
                        # Could mark for investigation or reload
                        pass
                        
            except asyncio.TimeoutError:
                logger.warning(f"Model {model_name} health check timed out")
            except Exception as e:
                logger.error(f"Health check failed for {model_name}: {e}")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        memory_stats = self.memory_manager.get_memory_stats()
        
        report = {
            "timestamp": asyncio.get_event_loop().time(),
            "system_health": {
                "memory_pressure": memory_stats["system_memory"]["pressure_level"],
                "available_memory_gb": memory_stats["system_memory"]["available_gb"],
                "memory_usage_percent": memory_stats["system_memory"]["used_percent"]
            },
            "model_health": {
                "loaded_models": memory_stats["model_counts"]["loaded"],
                "error_models": memory_stats["model_counts"]["error"],
                "total_tracked": memory_stats["model_counts"]["total_tracked"]
            },
            "performance_summary": {
                "top_performers": memory_stats["top_models_by_usage"],
                "recommendations": []
            }
        }
        
        # Add recommendations based on health status
        if memory_stats["system_memory"]["used_percent"] > 85:
            report["performance_summary"]["recommendations"].append(
                "High memory usage detected - consider unloading unused models"
            )
        
        if memory_stats["model_counts"]["error"] > 0:
            report["performance_summary"]["recommendations"].append(
                f"{memory_stats['model_counts']['error']} models in error state - investigate"
            )
        
        return report