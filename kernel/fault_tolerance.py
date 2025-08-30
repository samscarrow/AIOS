"""
Fault Tolerance System - Provides error recovery and graceful degradation for GAIA
"""

import asyncio
import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


class ComponentState(Enum):
    """Health states for system components"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorRecord:
    """Records a system error"""
    component_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: float = field(default_factory=time.time)
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: Optional[bool] = None


@dataclass
class ComponentHealth:
    """Tracks health of a system component"""
    component_id: str
    state: ComponentState = ComponentState.HEALTHY
    error_count: int = 0
    last_error: Optional[ErrorRecord] = None
    last_health_check: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    
    def is_healthy(self) -> bool:
        return self.state == ComponentState.HEALTHY
        
    def is_operational(self) -> bool:
        return self.state in [ComponentState.HEALTHY, ComponentState.DEGRADED]


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for component isolation"""
    component_id: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    test_request_count: int = 3
    
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    successful_calls: int = 0
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed through"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.successful_calls = 0
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return self.successful_calls < self.test_request_count
        return False
        
    def record_success(self):
        """Record successful operation"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.successful_calls += 1
            if self.successful_calls >= self.test_request_count:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
            
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN


class FaultToleranceManager:
    """
    Central fault tolerance manager for GAIA system
    """
    
    def __init__(self):
        self.component_health: Dict[str, ComponentHealth] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_handlers: Dict[str, Callable] = {}
        self.degradation_handlers: Dict[str, Callable] = {}
        
        # Error patterns and thresholds
        self.error_patterns = defaultdict(int)
        self.critical_components = {"kernel", "attention_manager", "async_executor"}
        
        # System-wide limits
        self.max_total_errors = 100  # per hour
        self.max_component_errors = 10  # per component per hour
        self.shutdown_threshold = 5  # critical errors before shutdown
        
    def register_component(self, component_id: str, 
                          recovery_handler: Optional[Callable] = None,
                          degradation_handler: Optional[Callable] = None,
                          max_recovery_attempts: int = 3):
        """Register a component for health monitoring"""
        self.component_health[component_id] = ComponentHealth(
            component_id=component_id,
            max_recovery_attempts=max_recovery_attempts
        )
        self.circuit_breakers[component_id] = CircuitBreaker(component_id)
        
        if recovery_handler:
            self.recovery_handlers[component_id] = recovery_handler
        if degradation_handler:
            self.degradation_handlers[component_id] = degradation_handler
            
        logger.info(f"Registered component {component_id} for fault tolerance")
        
    async def execute_with_fault_tolerance(self, component_id: str, 
                                         operation: Callable, 
                                         *args, **kwargs) -> Any:
        """Execute operation with fault tolerance"""
        if component_id not in self.circuit_breakers:
            self.register_component(component_id)
            
        circuit_breaker = self.circuit_breakers[component_id]
        
        # Check circuit breaker
        if not circuit_breaker.should_allow_request():
            raise CircuitBreakerOpenError(f"Circuit breaker open for {component_id}")
            
        try:
            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
                
            # Record success
            circuit_breaker.record_success()
            self._record_success(component_id)
            
            return result
            
        except Exception as e:
            # Record failure
            circuit_breaker.record_failure()
            await self._handle_error(component_id, e, args, kwargs)
            raise
            
    async def _handle_error(self, component_id: str, error: Exception,
                          args: tuple, kwargs: dict):
        """Handle component error"""
        # Determine error severity
        severity = self._classify_error_severity(error, component_id)
        
        # Create error record
        error_record = ErrorRecord(
            component_id=component_id,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            stack_trace=traceback.format_exc(),
            context={"args": str(args), "kwargs": str(kwargs)}
        )
        
        # Store error
        self.error_history.append(error_record)
        health = self.component_health[component_id]
        health.error_count += 1
        health.last_error = error_record
        health.consecutive_failures += 1
        
        logger.error(f"Error in {component_id}: {error}")
        
        # Update component state based on severity
        await self._update_component_state(component_id, error_record)
        
        # Attempt recovery if appropriate
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            await self._attempt_recovery(component_id, error_record)
            
        # Check for system-wide issues
        await self._check_system_health()
        
    def _classify_error_severity(self, error: Exception, component_id: str) -> ErrorSeverity:
        """Classify error severity"""
        error_type = type(error).__name__
        
        # Critical errors
        if error_type in ["SystemExit", "KeyboardInterrupt", "MemoryError"]:
            return ErrorSeverity.CRITICAL
            
        # Component-specific critical errors
        if component_id in self.critical_components:
            if error_type in ["ConnectionError", "TimeoutError", "ResourceExhausted"]:
                return ErrorSeverity.CRITICAL
                
        # High severity errors
        if error_type in ["ValueError", "TypeError", "AttributeError"]:
            return ErrorSeverity.HIGH
            
        # Medium severity
        if error_type in ["RuntimeError", "IOError"]:
            return ErrorSeverity.MEDIUM
            
        # Default to low
        return ErrorSeverity.LOW
        
    async def _update_component_state(self, component_id: str, error_record: ErrorRecord):
        """Update component health state"""
        health = self.component_health[component_id]
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            health.state = ComponentState.FAILED
        elif error_record.severity == ErrorSeverity.HIGH:
            if health.consecutive_failures >= 3:
                health.state = ComponentState.FAILING
            else:
                health.state = ComponentState.DEGRADED
        elif error_record.severity == ErrorSeverity.MEDIUM:
            if health.consecutive_failures >= 5:
                health.state = ComponentState.DEGRADED
                
        logger.info(f"Component {component_id} state: {health.state.value}")
        
    async def _attempt_recovery(self, component_id: str, error_record: ErrorRecord):
        """Attempt to recover component"""
        health = self.component_health[component_id]
        
        if health.recovery_attempts >= health.max_recovery_attempts:
            logger.warning(f"Max recovery attempts reached for {component_id}")
            return
            
        if component_id in self.recovery_handlers:
            health.state = ComponentState.RECOVERING
            health.recovery_attempts += 1
            error_record.recovery_attempted = True
            
            try:
                recovery_handler = self.recovery_handlers[component_id]
                if asyncio.iscoroutinefunction(recovery_handler):
                    await recovery_handler()
                else:
                    recovery_handler()
                    
                # Reset health on successful recovery
                health.state = ComponentState.HEALTHY
                health.consecutive_failures = 0
                health.recovery_attempts = 0
                error_record.recovery_successful = True
                
                logger.info(f"Successfully recovered {component_id}")
                
            except Exception as e:
                logger.error(f"Recovery failed for {component_id}: {e}")
                health.state = ComponentState.FAILED
                error_record.recovery_successful = False
                
    def _record_success(self, component_id: str):
        """Record successful operation"""
        if component_id in self.component_health:
            health = self.component_health[component_id]
            health.consecutive_failures = 0
            health.last_health_check = time.time()
            
            # Improve state if degraded
            if health.state == ComponentState.DEGRADED:
                health.state = ComponentState.HEALTHY
                
    async def _check_system_health(self):
        """Check overall system health"""
        critical_errors = sum(
            1 for error in list(self.error_history)[-50:]  # Last 50 errors
            if error.severity == ErrorSeverity.CRITICAL
        )
        
        if critical_errors >= self.shutdown_threshold:
            logger.critical(f"System shutdown threshold reached: {critical_errors} critical errors")
            await self._initiate_graceful_shutdown()
            
    async def _initiate_graceful_shutdown(self):
        """Initiate graceful system shutdown"""
        logger.critical("Initiating graceful shutdown due to critical failures")
        
        # Notify all components
        for component_id in self.component_health:
            if component_id in self.degradation_handlers:
                try:
                    handler = self.degradation_handlers[component_id]
                    if asyncio.iscoroutinefunction(handler):
                        await handler("shutdown")
                    else:
                        handler("shutdown")
                except Exception as e:
                    logger.error(f"Error during shutdown of {component_id}: {e}")
                    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
        
        return {
            "overall_health": self._calculate_overall_health(),
            "components": {
                comp_id: {
                    "state": health.state.value,
                    "error_count": health.error_count,
                    "consecutive_failures": health.consecutive_failures,
                    "last_error": health.last_error.error_message if health.last_error else None
                }
                for comp_id, health in self.component_health.items()
            },
            "circuit_breakers": {
                comp_id: cb.state.value
                for comp_id, cb in self.circuit_breakers.items()
            },
            "recent_errors": len(recent_errors),
            "critical_errors": sum(1 for e in recent_errors if e.severity == ErrorSeverity.CRITICAL),
            "error_patterns": dict(self.error_patterns)
        }
        
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health"""
        if not self.component_health:
            return "unknown"
            
        states = [health.state for health in self.component_health.values()]
        
        if any(state == ComponentState.FAILED for state in states):
            if any(comp_id in self.critical_components 
                  for comp_id, health in self.component_health.items() 
                  if health.state == ComponentState.FAILED):
                return "critical"
            return "degraded"
            
        if any(state in [ComponentState.FAILING, ComponentState.DEGRADED] for state in states):
            return "degraded"
            
        return "healthy"


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class ResourceExhaustedError(Exception):
    """Raised when system resources are exhausted"""
    pass