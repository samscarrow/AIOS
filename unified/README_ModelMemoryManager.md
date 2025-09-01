# Intelligent Model Memory Management System

## Overview

The `ModelMemoryManager` is an intelligent memory management system designed for cognitive platforms that use multiple AI models. It automatically optimizes model loading and unloading based on usage patterns, system resources, and cognitive strategies.

## Key Features

### 🧠 Intelligent Memory Management
- **Automatic model discovery** from LMStudio and other providers
- **Smart unloading** of unused models based on priority scoring
- **Memory pressure detection** with automatic optimization
- **Usage pattern learning** for predictive model loading

### 📊 Comprehensive Metrics Tracking
- Real-time model performance monitoring
- Success rates, latency tracking, and token usage
- Usage frequency analysis and trend detection
- Memory consumption estimates

### 🔗 Seamless Integration
- Integration with `UniversalProviderKernel` for automatic model management
- Integration with `EnhancedLearningStrategist` for cognitive strategy optimization
- Middleware support for request-level optimization
- Health monitoring with automated alerts

### ⚡ Performance Optimization
- Predictive model preloading based on context
- Concurrent model loading/unloading
- Background memory optimization
- Configurable memory thresholds and timeouts

## Quick Start

```python
from unified.model_memory_manager import ModelMemoryManager
from unified.integration_utils import CognitiveModelIntegrator

# Basic usage
async def basic_example():
    manager = ModelMemoryManager(
        max_memory_usage_percent=80.0,
        model_timeout_hours=2.0,
        min_free_memory_gb=4.0
    )
    
    await manager.start_monitoring()
    
    try:
        # Ensure required models are available
        required_models = ["qwen/qwen3-8b", "qwen/qwen2.5-coder-14b"]
        success = await manager.ensure_models_available(required_models)
        
        # Get memory statistics
        stats = manager.get_memory_stats()
        print(f"Memory usage: {stats['system_memory']['used_percent']:.1f}%")
        
        # Record model usage
        manager.record_model_usage("qwen/qwen3-8b", latency=150.0, success=True, tokens_used=50)
        
        # Get recommendations
        context = {"task_type": "coding", "complexity": 0.8}
        recommendations = manager.get_model_recommendations(context)
        
    finally:
        await manager.stop_monitoring()
        await manager.cleanup()
```

## Core Components

### 1. ModelMemoryManager

The main class that handles all model memory operations:

```python
manager = ModelMemoryManager(
    max_memory_usage_percent=80.0,  # Trigger optimization at 80% memory
    model_timeout_hours=2.0,        # Unload models unused for 2+ hours
    min_free_memory_gb=4.0,         # Maintain at least 4GB free
    monitoring_interval=30.0        # Check memory every 30 seconds
)
```

**Key Methods:**

- `get_loaded_models()` - Get list of currently loaded models
- `load_model(model_name, priority)` - Load specific model
- `unload_model(model_name)` - Unload specific model  
- `ensure_models_available(models)` - Ensure multiple models are loaded
- `clear_all_models()` - Unload all models
- `optimize_memory_usage()` - Perform intelligent memory optimization
- `record_model_usage(model, latency, success)` - Record usage metrics
- `get_model_recommendations(context)` - Get intelligent model recommendations

### 2. CognitiveModelIntegrator

Integrates the memory manager with existing cognitive platform components:

```python
integrator = CognitiveModelIntegrator(memory_manager)

# Integrate with kernel to automatically track model usage
integrator.integrate_with_kernel(universal_provider_kernel)

# Integrate with strategist to preload required models
integrator.integrate_with_strategist(enhanced_learning_strategist)
```

### 3. ModelMemoryMiddleware

Provides automatic model management for request processing:

```python
middleware = ModelMemoryMiddleware(memory_manager)

# Use as middleware in request pipeline
async def process_request(context):
    return await middleware(context, your_request_handler)
```

### 4. ModelHealthMonitor

Continuously monitors model health and performance:

```python
monitor = ModelHealthMonitor(memory_manager)
await monitor.start_monitoring(interval=60.0)

# Get comprehensive health report
health_report = monitor.get_health_report()
```

## Advanced Usage

### Integration with Cognitive Platform

```python
from unified.mcp_integration import MCPIntegrationWrapper

async def cognitive_platform_integration():
    # Use MCP integration for LMStudio
    async with MCPIntegrationWrapper(use_mcp=True) as manager:
        
        # The manager now automatically discovers and manages LMStudio models
        models = await manager.get_loaded_models()
        
        # Integration with cognitive strategies
        strategy_context = {
            "task_type": "reasoning",
            "complexity": 0.9,
            "urgency": 0.2,
            "available_memory_gb": 12.0
        }
        
        recommendations = manager.get_model_recommendations(strategy_context)
        
        # Preload recommended models
        if recommendations["recommended_models"]:
            await manager.ensure_models_available(
                recommendations["recommended_models"], 
                priority=8
            )
```

### Memory Optimization Scenarios

```python
# Scenario 1: Multi-agent task with concurrent models
async def multi_agent_scenario():
    async with MCPIntegrationWrapper() as manager:
        # Ensure multiple models for different agents
        agent_models = [
            "qwen/qwen3-8b",           # Reasoning agent
            "qwen/qwen2.5-coder-14b",  # Coding agent
            "deepseek/deepseek-r1-0528-qwen3-8b"  # Analysis agent
        ]
        
        await manager.ensure_models_available(agent_models, priority=9)
        
        # Simulate concurrent usage
        tasks = []
        for model in agent_models:
            task = asyncio.create_task(simulate_agent_work(manager, model))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # System automatically optimizes memory after concurrent usage

# Scenario 2: Adaptive model selection based on performance
async def adaptive_selection():
    async with MCPIntegrationWrapper() as manager:
        # Record usage over time
        for task in tasks:
            model = select_best_model_for_task(task, manager.get_memory_stats())
            
            start_time = time.time()
            result = await process_with_model(task, model)
            latency = (time.time() - start_time) * 1000
            
            # Record performance
            manager.record_model_usage(model, latency, result.success, result.tokens)
            
            # System learns from this data for future recommendations
```

### Custom Memory Pressure Handling

```python
class CustomMemoryManager(ModelMemoryManager):
    async def _handle_critical_memory(self):
        """Custom critical memory handling"""
        # Custom logic for critical memory situations
        logger.warning("Custom critical memory handler activated")
        
        # First, try gentle optimization
        await self.optimize_memory_usage()
        
        # If still critical, unload non-essential models
        loaded = await self.get_loaded_models()
        essential_models = ["qwen/qwen3-8b"]  # Always keep core reasoning model
        
        for model in loaded:
            if model not in essential_models:
                await self.unload_model(model)
                
                # Check if pressure is relieved
                await self._check_memory_pressure()
                if self.memory_pressure != MemoryPressure.CRITICAL:
                    break

# Usage
manager = CustomMemoryManager(max_memory_usage_percent=75.0)
```

## Configuration Options

### Memory Manager Configuration

```python
ModelMemoryManager(
    max_memory_usage_percent=80.0,   # System memory threshold for optimization
    model_timeout_hours=2.0,         # Hours before unused models are unloaded
    min_free_memory_gb=4.0,          # Minimum free memory to maintain
    monitoring_interval=30.0         # Seconds between memory checks
)
```

### Event Hooks

```python
def on_model_loaded(model_name):
    print(f"Model loaded: {model_name}")

def on_memory_pressure(pressure_level):
    print(f"Memory pressure: {pressure_level}")

manager.on_model_loaded = on_model_loaded
manager.on_memory_pressure = on_memory_pressure
```

## Performance Metrics

The system tracks comprehensive metrics for optimization:

```python
stats = manager.get_memory_stats()

# System memory information
print(f"Memory usage: {stats['system_memory']['used_percent']:.1f}%")
print(f"Available memory: {stats['system_memory']['available_gb']:.1f}GB")

# Model statistics
print(f"Loaded models: {stats['model_counts']['loaded']}")
print(f"Models in error: {stats['model_counts']['error']}")

# Top performing models
for model_info in stats['top_models_by_usage']:
    print(f"{model_info['model']}: {model_info['frequency']:.2f} uses/hour")
```

## Error Handling

The system includes comprehensive error handling:

```python
try:
    success = await manager.load_model("nonexistent/model")
    if not success:
        # Handle loading failure
        alternatives = manager.get_model_recommendations(context)["recommended_models"]
        for alt in alternatives:
            if await manager.load_model(alt):
                break
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    # Fallback strategy
```

## Best Practices

### 1. Always Use Context Managers
```python
async with MCPIntegrationWrapper() as manager:
    # Your code here
    pass  # Automatic cleanup
```

### 2. Record Usage Patterns
```python
# Always record model usage for learning
manager.record_model_usage(model_name, latency, success, tokens_used)
```

### 3. Use Priorities Appropriately
```python
# High priority for critical models
await manager.ensure_models_available(critical_models, priority=9)

# Lower priority for optional models
await manager.ensure_models_available(optional_models, priority=3)
```

### 4. Monitor System Health
```python
# Regular health monitoring
monitor = ModelHealthMonitor(manager)
await monitor.start_monitoring()

# Check health reports
health = monitor.get_health_report()
if health["system_health"]["memory_pressure"] == "critical":
    # Take action
    await manager.clear_all_models()
```

### 5. Implement Custom Optimization
```python
# Custom optimization for specific workloads
class WorkloadSpecificManager(ModelMemoryManager):
    def get_model_recommendations(self, context):
        recommendations = super().get_model_recommendations(context)
        
        # Add workload-specific logic
        if context.get("workload_type") == "batch_processing":
            recommendations["models_to_preload"].extend(["large-batch-model"])
        
        return recommendations
```

## Troubleshooting

### Common Issues

**High Memory Usage:**
```python
# Check what's consuming memory
stats = manager.get_memory_stats()
print("Models by memory usage:")
for model, metrics in manager.model_metrics.items():
    print(f"{model}: {metrics.estimated_memory_mb}MB")

# Force optimization
await manager.optimize_memory_usage()
```

**Models Not Loading:**
```python
# Check model availability
loaded = await manager.get_loaded_models()
print(f"Available models: {loaded}")

# Check for errors
for model, status in manager.model_status.items():
    if status == ModelStatus.ERROR:
        print(f"Model in error state: {model}")
```

**Poor Performance:**
```python
# Analyze model performance
for model_name, metrics in manager.model_metrics.items():
    print(f"{model_name}:")
    print(f"  Success rate: {metrics.success_rate:.2f}")
    print(f"  Avg latency: {metrics.avg_latency:.2f}ms")
    print(f"  Usage frequency: {metrics.usage_frequency:.2f}/hour")
```

## Integration Examples

### With FastAPI
```python
from fastapi import FastAPI
from unified.mcp_integration import create_model_memory_manager

app = FastAPI()
manager = None

@app.on_event("startup")
async def startup():
    global manager
    manager = await create_model_memory_manager(use_mcp=True)

@app.on_event("shutdown")
async def shutdown():
    if manager:
        await manager.cleanup()

@app.post("/generate")
async def generate(request: GenerateRequest):
    # Ensure required model is available
    await manager.ensure_models_available([request.model])
    
    # Process request...
    
    # Record usage
    manager.record_model_usage(request.model, latency, success, tokens)
```

### With AsyncIO Workers
```python
async def worker_with_memory_management(worker_id: int):
    async with MCPIntegrationWrapper() as manager:
        while True:
            task = await get_next_task()
            
            # Get model recommendations for task
            recommendations = manager.get_model_recommendations(task.context)
            
            # Ensure recommended models are available
            await manager.ensure_models_available(
                recommendations["recommended_models"],
                priority=task.priority
            )
            
            # Process task...
            
            # Record results
            manager.record_model_usage(task.model, task.latency, task.success, task.tokens)

# Start multiple workers
workers = [worker_with_memory_management(i) for i in range(4)]
await asyncio.gather(*workers)
```

This intelligent model memory management system provides the foundation for building efficient, scalable cognitive platforms that can automatically optimize their resource usage while maintaining high performance and reliability.