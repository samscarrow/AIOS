"""
Unified Configuration System
Merges cognitive platform settings with multi-provider AI configurations
"""

from .unified_config import (
    UnifiedConfig,
    ProviderConfig, 
    CognitiveConfig,
    MCPConfig,
    CostConfig,
    ProviderType,
    StrategyType,
    UnifiedConfigManager,
    get_config,
    get_config_manager,
    reload_config
)

from .validation import (
    ConfigValidator,
    ValidationError,
    validate_config_file,
    print_validation_report
)

__all__ = [
    # Core configuration classes
    'UnifiedConfig',
    'ProviderConfig', 
    'CognitiveConfig',
    'MCPConfig',
    'CostConfig',
    'ProviderType',
    'StrategyType',
    
    # Configuration management
    'UnifiedConfigManager',
    'get_config',
    'get_config_manager', 
    'reload_config',
    
    # Validation
    'ConfigValidator',
    'ValidationError',
    'validate_config_file',
    'print_validation_report'
]