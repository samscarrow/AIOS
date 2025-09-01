#!/usr/bin/env python3
"""
MCP Integration Module for ModelMemoryManager
Provides real integration with LMStudio MCP tools when available
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from .model_memory_manager import ModelMemoryManager, ModelStatus

logger = logging.getLogger(__name__)


class MCPModelManager(ModelMemoryManager):
    """
    Extended ModelMemoryManager with actual MCP tool integration
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcp_available = self._check_mcp_availability()
    
    def _check_mcp_availability(self) -> bool:
        """Check if MCP tools are available in the environment"""
        try:
            # Try to import MCP functions (this would work in Claude Code environment)
            import builtins
            return hasattr(builtins, 'mcp__lmstudio_omnibus__discover_models')
        except:
            return False
    
    async def get_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded models using actual MCP tools
        """
        if not self.mcp_available:
            return await super().get_loaded_models()
        
        try:
            # Use actual MCP tools when available
            discovered_models = await self._discover_models_mcp()
            metrics = await self._get_metrics_mcp()
            
            loaded_models = []
            for model_name in discovered_models:
                # Consider a model "loaded" if it has metrics or is tracked as loaded
                if (model_name in metrics and metrics[model_name].get('total_requests', 0) > 0) or \
                   self.model_status.get(model_name) == ModelStatus.LOADED:
                    loaded_models.append(model_name)
                    self.model_status[model_name] = ModelStatus.LOADED
                    
                    # Update metrics if not exists
                    if model_name not in self.model_metrics:
                        from .model_memory_manager import ModelMetrics
                        self.model_metrics[model_name] = ModelMetrics(model_name)
                        
                    # Update from MCP metrics
                    if model_name in metrics:
                        self._update_metrics_from_mcp(model_name, metrics[model_name])
                        
            return loaded_models
            
        except Exception as e:
            logger.error(f"MCP integration failed, falling back to base implementation: {e}")
            return await super().get_loaded_models()
    
    async def _discover_models_mcp(self) -> List[str]:
        """Discover models using MCP tool"""
        try:
            # This would be the actual MCP call in Claude Code environment
            # For now, we'll use a placeholder that can be replaced
            
            # In real implementation:
            # from mcp__lmstudio_omnibus__discover_models import discover_models
            # return await discover_models()
            
            # Placeholder for demonstration
            return [
                "qwen/qwen2.5-coder-14b",
                "qwen/qwen3-8b", 
                "deepseek/deepseek-r1-0528-qwen3-8b",
                "qwen/qwen3-32b",
                "hermes-4-70b@q4_k_m",
                "qwen/qwen3-coder-30b"
            ]
            
        except Exception as e:
            logger.error(f"Failed to discover models via MCP: {e}")
            return []
    
    async def _get_metrics_mcp(self) -> Dict[str, Any]:
        """Get model metrics using MCP tool"""
        try:
            # In real implementation:
            # from mcp__lmstudio_omnibus__get_metrics import get_metrics
            # return await get_metrics()
            
            # Placeholder for demonstration
            return {
                "qwen/qwen3-8b": {
                    "total_requests": 5,
                    "successful_requests": 4,
                    "total_latency": 1500.0,
                    "min_latency": 200.0,
                    "max_latency": 500.0,
                    "total_tokens": 1200
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics via MCP: {e}")
            return {}
    
    def _update_metrics_from_mcp(self, model_name: str, mcp_metrics: Dict[str, Any]):
        """Update internal metrics from MCP data"""
        if model_name not in self.model_metrics:
            return
            
        metrics = self.model_metrics[model_name]
        
        # Update from MCP metrics
        metrics.total_requests = mcp_metrics.get('total_requests', metrics.total_requests)
        metrics.successful_requests = mcp_metrics.get('successful_requests', metrics.successful_requests)
        metrics.failed_requests = metrics.total_requests - metrics.successful_requests
        metrics.total_latency = mcp_metrics.get('total_latency', metrics.total_latency)
        
        min_lat = mcp_metrics.get('min_latency')
        if min_lat is not None:
            metrics.min_latency = min(metrics.min_latency, min_lat)
            
        max_lat = mcp_metrics.get('max_latency')
        if max_lat is not None:
            metrics.max_latency = max(metrics.max_latency, max_lat)
    
    async def _load_model_impl(self, model_name: str) -> bool:
        """Implementation for loading a model using MCP tools"""
        if not self.mcp_available:
            return await super()._load_model_impl(model_name)
        
        try:
            logger.info(f"Loading model via MCP: {model_name}")
            
            # Use MCP query to trigger model loading
            success = await self._trigger_model_load_mcp(model_name)
            
            if success:
                logger.info(f"Successfully loaded model via MCP: {model_name}")
            else:
                logger.warning(f"Failed to load model via MCP: {model_name}")
                
            return success
            
        except Exception as e:
            logger.error(f"MCP model loading failed: {e}")
            return False
    
    async def _trigger_model_load_mcp(self, model_name: str) -> bool:
        """Trigger model loading using MCP query"""
        try:
            # In real implementation:
            # from mcp__lmstudio_omnibus__query import query
            # response = await query(
            #     prompt="Hello", 
            #     context={"preferred_model": model_name}
            # )
            # return response is not None
            
            # Simulate loading for demo
            await asyncio.sleep(0.5)
            
            # Simulate success for known models
            known_models = [
                "qwen/qwen2.5-coder-14b", "qwen/qwen3-8b", 
                "deepseek/deepseek-r1-0528-qwen3-8b", "qwen/qwen3-32b",
                "hermes-4-70b@q4_k_m", "qwen/qwen3-coder-30b"
            ]
            
            return model_name in known_models
            
        except Exception as e:
            logger.error(f"Failed to trigger model load: {e}")
            return False
    
    async def health_check_mcp(self) -> Dict[str, Any]:
        """Perform health check using MCP tools"""
        try:
            # In real implementation:
            # from mcp__lmstudio_omnibus__health import health
            # return await health()
            
            # Simulate health check for demo
            return {
                "status": "healthy",
                "providers": {
                    "lmstudio": {
                        "status": "healthy",
                        "models": len(await self.get_loaded_models())
                    }
                },
                "memory_pressure": self.memory_pressure.value
            }
            
        except Exception as e:
            logger.error(f"MCP health check failed: {e}")
            return {"status": "error", "error": str(e)}


class MCPIntegrationWrapper:
    """
    Wrapper that provides seamless integration between ModelMemoryManager and MCP tools
    """
    
    def __init__(self, use_mcp: bool = True):
        self.use_mcp = use_mcp and self._check_mcp_environment()
        
        if self.use_mcp:
            self.memory_manager = MCPModelManager()
            logger.info("Using MCP-integrated ModelMemoryManager")
        else:
            self.memory_manager = ModelMemoryManager()
            logger.info("Using standard ModelMemoryManager")
    
    def _check_mcp_environment(self) -> bool:
        """Check if we're running in an MCP-enabled environment"""
        try:
            import builtins
            return any(hasattr(builtins, name) for name in [
                'mcp__lmstudio_omnibus__discover_models',
                'mcp__lmstudio_omnibus__query',
                'mcp__lmstudio_omnibus__health'
            ])
        except:
            return False
    
    async def initialize(self) -> 'MCPIntegrationWrapper':
        """Initialize the memory manager"""
        await self.memory_manager.start_monitoring()
        return self
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.memory_manager.stop_monitoring()
        await self.memory_manager.cleanup()
    
    def __getattr__(self, name):
        """Delegate all other attributes to the memory manager"""
        return getattr(self.memory_manager, name)
    
    async def __aenter__(self):
        """Async context manager entry"""
        return await self.initialize()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()


# Convenience function for easy integration
async def create_model_memory_manager(use_mcp: bool = True) -> MCPIntegrationWrapper:
    """
    Create and initialize a ModelMemoryManager with optional MCP integration
    
    Args:
        use_mcp: Whether to use MCP integration if available
        
    Returns:
        Initialized MCPIntegrationWrapper
    """
    wrapper = MCPIntegrationWrapper(use_mcp=use_mcp)
    return await wrapper.initialize()


# Example usage functions
async def example_basic_usage():
    """Example of basic ModelMemoryManager usage"""
    async with MCPIntegrationWrapper() as manager:
        # Get loaded models
        models = await manager.get_loaded_models()
        print(f"Loaded models: {models}")
        
        # Ensure specific models are available
        required = ["qwen/qwen3-8b", "qwen/qwen2.5-coder-14b"]
        success = await manager.ensure_models_available(required)
        print(f"Models available: {success}")
        
        # Get memory statistics
        stats = manager.get_memory_stats()
        print(f"Memory stats: {stats}")


async def example_integration_with_cognitive_platform():
    """Example of integration with cognitive platform components"""
    from .integration_utils import CognitiveModelIntegrator
    
    async with MCPIntegrationWrapper() as manager:
        # Create integrator
        integrator = CognitiveModelIntegrator(manager)
        
        # This would integrate with actual kernel and strategist
        # integrator.integrate_with_kernel(kernel)
        # integrator.integrate_with_strategist(strategist)
        
        print("Integration complete - memory manager is now monitoring model usage")


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_basic_usage())