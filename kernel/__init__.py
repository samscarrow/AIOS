"""
GAIA Kernel - The cognitive core of the General AI Architecture
"""

from .core import GAIAKernel
from .attention import AttentionManager
from .fault_tolerance import FaultToleranceManager, CircuitBreaker, ErrorSeverity

__all__ = ['GAIAKernel', 'AttentionManager', 'FaultToleranceManager', 'CircuitBreaker', 'ErrorSeverity']