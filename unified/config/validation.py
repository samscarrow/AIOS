#!/usr/bin/env python3
"""
Configuration Validation
Validates unified configuration for correctness and best practices
"""

from typing import List, Dict, Any, Tuple
import re
import urllib.parse
from .unified_config import UnifiedConfig, ProviderConfig, ProviderType
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Configuration validation error"""
    pass


class ConfigValidator:
    """Validates unified configuration"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_config(self, config: UnifiedConfig) -> Tuple[bool, List[str], List[str]]:
        """
        Validate complete configuration
        Returns (is_valid, errors, warnings)
        """
        self.errors.clear()
        self.warnings.clear()
        
        self._validate_providers(config.providers)
        self._validate_cognitive_config(config.cognitive)
        self._validate_mcp_config(config.mcp)
        self._validate_cost_config(config.cost)
        self._validate_strategy_mappings(config.domain_strategies, config.complexity_routing)
        
        return len(self.errors) == 0, self.errors.copy(), self.warnings.copy()
    
    def _validate_providers(self, providers: List[ProviderConfig]) -> None:
        """Validate provider configurations"""
        if not providers:
            self.errors.append("At least one provider must be configured")
            return
        
        enabled_providers = [p for p in providers if p.enabled]
        if not enabled_providers:
            self.errors.append("At least one provider must be enabled")
        
        provider_names = set()
        for i, provider in enumerate(providers):
            # Check for duplicate names
            if provider.name in provider_names:
                self.errors.append(f"Duplicate provider name: {provider.name}")
            provider_names.add(provider.name)
            
            # Validate individual provider
            self._validate_provider(provider, i)
    
    def _validate_provider(self, provider: ProviderConfig, index: int) -> None:
        """Validate individual provider configuration"""
        prefix = f"Provider {index} ({provider.name})"
        
        # Validate name
        if not provider.name or not provider.name.strip():
            self.errors.append(f"{prefix}: Name cannot be empty")
        elif not re.match(r'^[a-zA-Z0-9_-]+$', provider.name):
            self.errors.append(f"{prefix}: Name must contain only alphanumeric characters, hyphens, and underscores")
        
        # Validate endpoint
        if not provider.endpoint:
            self.errors.append(f"{prefix}: Endpoint cannot be empty")
        else:
            try:
                parsed = urllib.parse.urlparse(provider.endpoint)
                if not parsed.scheme or not parsed.netloc:
                    self.errors.append(f"{prefix}: Invalid endpoint URL format")
                elif parsed.scheme not in ['http', 'https']:
                    self.warnings.append(f"{prefix}: Endpoint should use http or https")
            except Exception:
                self.errors.append(f"{prefix}: Invalid endpoint URL")
        
        # Validate API key requirements
        if provider.type in [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.GEMINI]:
            if not provider.api_key:
                if provider.enabled:
                    self.errors.append(f"{prefix}: API key required for {provider.type.value}")
                else:
                    self.warnings.append(f"{prefix}: API key missing (provider disabled)")
        
        # Validate models
        if not provider.models:
            self.warnings.append(f"{prefix}: No models specified")
        
        # Validate numeric values
        if provider.max_concurrent <= 0:
            self.errors.append(f"{prefix}: max_concurrent must be positive")
        elif provider.max_concurrent > 20:
            self.warnings.append(f"{prefix}: max_concurrent > 20 may cause resource issues")
        
        if provider.timeout <= 0:
            self.errors.append(f"{prefix}: timeout must be positive")
        elif provider.timeout < 5:
            self.warnings.append(f"{prefix}: timeout < 5s may cause request failures")
        elif provider.timeout > 300:
            self.warnings.append(f"{prefix}: timeout > 5min may cause long waits")
        
        if provider.cost_per_token < 0:
            self.errors.append(f"{prefix}: cost_per_token cannot be negative")
        
        if not (0 <= provider.priority <= 100):
            self.errors.append(f"{prefix}: priority must be between 0 and 100")
    
    def _validate_cognitive_config(self, cognitive) -> None:
        """Validate cognitive configuration"""
        if not (0 <= cognitive.learning_rate <= 1):
            self.errors.append("Cognitive learning_rate must be between 0 and 1")
        
        if not (0 <= cognitive.adaptation_threshold <= 1):
            self.errors.append("Cognitive adaptation_threshold must be between 0 and 1")
        
        if cognitive.memory_window_size <= 0:
            self.errors.append("Cognitive memory_window_size must be positive")
        elif cognitive.memory_window_size > 10000:
            self.warnings.append("Large memory_window_size may impact performance")
        
        if not (0 <= cognitive.confidence_threshold <= 1):
            self.errors.append("Cognitive confidence_threshold must be between 0 and 1")
        
        if not (0 <= cognitive.quality_gate_threshold <= 1):
            self.errors.append("Cognitive quality_gate_threshold must be between 0 and 1")
        
        if cognitive.max_retries < 0:
            self.errors.append("Cognitive max_retries cannot be negative")
        elif cognitive.max_retries > 10:
            self.warnings.append("High max_retries may cause long delays on failures")
        
        if cognitive.circuit_breaker_threshold <= 0:
            self.errors.append("Cognitive circuit_breaker_threshold must be positive")
    
    def _validate_mcp_config(self, mcp) -> None:
        """Validate MCP configuration"""
        # Validate port
        if not (1 <= mcp.server_port <= 65535):
            self.errors.append("MCP server_port must be between 1 and 65535")
        elif mcp.server_port < 1024:
            self.warnings.append("MCP server_port < 1024 may require root privileges")
        
        # Validate host
        if not mcp.server_host:
            self.errors.append("MCP server_host cannot be empty")
        
        # Validate connections
        if mcp.max_connections <= 0:
            self.errors.append("MCP max_connections must be positive")
        elif mcp.max_connections > 1000:
            self.warnings.append("High max_connections may impact performance")
        
        # Validate timeout
        if mcp.tool_timeout <= 0:
            self.errors.append("MCP tool_timeout must be positive")
        elif mcp.tool_timeout > 300:
            self.warnings.append("Long tool_timeout may cause client timeouts")
        
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if mcp.log_level.upper() not in valid_levels:
            self.errors.append(f"MCP log_level must be one of: {valid_levels}")
    
    def _validate_cost_config(self, cost) -> None:
        """Validate cost configuration"""
        if cost.daily_budget < 0:
            self.errors.append("Cost daily_budget cannot be negative")
        
        if cost.monthly_budget < 0:
            self.errors.append("Cost monthly_budget cannot be negative")
        
        if cost.daily_budget > cost.monthly_budget:
            self.warnings.append("Daily budget exceeds monthly budget")
        
        if cost.cost_tracking_enabled and cost.daily_budget == 0:
            self.warnings.append("Cost tracking enabled but daily budget is 0")
    
    def _validate_strategy_mappings(self, domain_strategies: Dict[str, Any], 
                                  complexity_routing: Dict[str, Any]) -> None:
        """Validate strategy mapping configurations"""
        # Check that we have mappings for common domains
        recommended_domains = {'code', 'math', 'creative', 'general', 'business'}
        missing_domains = recommended_domains - set(domain_strategies.keys())
        if missing_domains:
            self.warnings.append(f"Missing domain strategies for: {missing_domains}")
        
        # Check that we have mappings for all complexity levels
        required_complexity = {'low', 'medium', 'high'}
        missing_complexity = required_complexity - set(complexity_routing.keys())
        if missing_complexity:
            self.warnings.append(f"Missing complexity routing for: {missing_complexity}")


def validate_config_file(config_path: str) -> Tuple[bool, List[str], List[str]]:
    """
    Validate configuration file
    Returns (is_valid, errors, warnings)
    """
    try:
        from .unified_config import UnifiedConfigManager
        
        manager = UnifiedConfigManager(config_path)
        config = manager.load_config()
        
        validator = ConfigValidator()
        return validator.validate_config(config)
        
    except Exception as e:
        return False, [f"Failed to load config: {str(e)}"], []


def print_validation_report(is_valid: bool, errors: List[str], warnings: List[str]) -> None:
    """Print formatted validation report"""
    if is_valid:
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has errors!")
    
    if errors:
        print(f"\n🚨 Errors ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    if not errors and not warnings:
        print("\n✨ No issues found!")


if __name__ == "__main__":
    """CLI for configuration validation"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate unified configuration")
    parser.add_argument("config_file", nargs="?", help="Configuration file to validate")
    args = parser.parse_args()
    
    if args.config_file:
        is_valid, errors, warnings = validate_config_file(args.config_file)
        print_validation_report(is_valid, errors, warnings)
        sys.exit(0 if is_valid else 1)
    else:
        # Validate default configuration
        from .unified_config import get_config_manager
        
        manager = get_config_manager()
        config = manager.load_config()
        
        validator = ConfigValidator()
        is_valid, errors, warnings = validator.validate_config(config)
        
        print(f"Validating: {manager.config_path}")
        print_validation_report(is_valid, errors, warnings)
        sys.exit(0 if is_valid else 1)