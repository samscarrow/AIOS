"""
AIOS Orchestrator - Async thought stream management and strategic orchestration
"""

from .async_executor import AsyncThoughtExecutor, ThoughtNode
from .aios_strategic_orchestrator import (
    AIOSStrategicOrchestrator,
    CognitiveTask,
    ThoughtStream,
    AIOS_ORCHESTRATION_TOOLS
)

__all__ = [
    'AsyncThoughtExecutor', 
    'ThoughtNode',
    'AIOSStrategicOrchestrator',
    'CognitiveTask', 
    'ThoughtStream',
    'AIOS_ORCHESTRATION_TOOLS'
]

# Default orchestrator instance for AIOS integration
_default_orchestrator = None

def get_orchestrator(aios_kernel=None):
    """Get or create the default AIOS strategic orchestrator"""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = AIOSStrategicOrchestrator(aios_kernel)
    return _default_orchestrator

def initialize_orchestrator(aios_kernel):
    """Initialize orchestrator with AIOS kernel"""
    global _default_orchestrator
    _default_orchestrator = AIOSStrategicOrchestrator(aios_kernel)
    return _default_orchestrator