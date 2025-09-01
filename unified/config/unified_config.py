#!/usr/bin/env python3
"""
Unified Configuration System
Merges cognitive platform settings with omnibus provider configs
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Available AI provider types"""
    LMSTUDIO = "lmstudio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    MOCK = "mock"  # For testing


class StrategyType(Enum):
    """Cognitive strategy types"""
    FAST_SMALL = "fast_small"
    SINGLE_LARGE = "single_large"
    MULTI_AGENT_REFINE = "multi_agent_refine"
    CONSENSUS = "consensus"
    ADAPTIVE = "adaptive"


@dataclass
class ProviderConfig:
    """Configuration for a single AI provider"""
    name: str
    type: ProviderType
    endpoint: str
    api_key: Optional[str] = None
    models: List[str] = field(default_factory=list)
    max_concurrent: int = 3
    timeout: float = 30.0
    cost_per_token: float = 0.0001
    enabled: bool = True
    priority: int = 50  # 0-100, higher = more preferred


@dataclass
class CognitiveConfig:
    """Cognitive platform configuration"""
    learning_rate: float = 0.1
    adaptation_threshold: float = 0.2
    memory_window_size: int = 1000
    confidence_threshold: float = 0.7
    quality_gate_threshold: float = 0.6
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    enable_introspection: bool = True
    golden_bus_enabled: bool = True


@dataclass
class MCPConfig:
    """MCP (Model Context Protocol) configuration"""
    server_host: str = "localhost"
    server_port: int = 3000
    max_connections: int = 100
    tool_timeout: float = 30.0
    enable_logging: bool = True
    log_level: str = "INFO"


@dataclass
class CostConfig:
    """Cost optimization configuration"""
    daily_budget: float = 10.0  # USD
    monthly_budget: float = 300.0  # USD
    cost_tracking_enabled: bool = True
    budget_alerts_enabled: bool = True
    auto_fallback_on_budget: bool = True
    cost_optimization_enabled: bool = True


@dataclass 
class UnifiedConfig:
    """Complete unified configuration"""
    providers: List[ProviderConfig] = field(default_factory=list)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    
    # Strategy preferences by domain
    domain_strategies: Dict[str, StrategyType] = field(default_factory=lambda: {
        "code": StrategyType.ADAPTIVE,
        "math": StrategyType.CONSENSUS,
        "creative": StrategyType.SINGLE_LARGE,
        "general": StrategyType.ADAPTIVE,
        "business": StrategyType.SINGLE_LARGE
    })
    
    # Complexity-based routing
    complexity_routing: Dict[str, StrategyType] = field(default_factory=lambda: {
        "low": StrategyType.FAST_SMALL,
        "medium": StrategyType.ADAPTIVE,
        "high": StrategyType.SINGLE_LARGE
    })


class UnifiedConfigManager:
    """Manages unified configuration across cognitive platform and providers"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self._config: Optional[UnifiedConfig] = None
        
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path"""
        # Check environment variable first
        if env_path := os.getenv("UNIFIED_CONFIG_PATH"):
            return Path(env_path)
        
        # Look in common locations
        possible_paths = [
            Path.cwd() / "unified_config.yaml",
            Path.home() / ".config" / "aios" / "config.yaml",
            Path(__file__).parent / "default_config.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        # Default to current directory
        return Path.cwd() / "unified_config.yaml"
    
    def load_config(self) -> UnifiedConfig:
        """Load configuration from file"""
        if self._config is not None:
            return self._config
            
        if not self.config_path.exists():
            logger.info(f"Config file not found at {self.config_path}, creating default")
            self._config = self._create_default_config()
            self.save_config()
            return self._config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self._config = self._dict_to_config(data)
            logger.info(f"Loaded configuration from {self.config_path}")
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
            self._config = self._create_default_config()
            return self._config
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        if self._config is None:
            raise ValueError("No configuration to save")
            
        # Create directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = self._config_to_dict(self._config)
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Saved configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
            raise
    
    def get_provider_by_name(self, name: str) -> Optional[ProviderConfig]:
        """Get provider configuration by name"""
        config = self.load_config()
        return next((p for p in config.providers if p.name == name), None)
    
    def get_enabled_providers(self) -> List[ProviderConfig]:
        """Get all enabled providers sorted by priority (descending)"""
        config = self.load_config()
        return sorted(
            [p for p in config.providers if p.enabled],
            key=lambda p: p.priority,
            reverse=True
        )
    
    def update_provider(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update provider configuration"""
        config = self.load_config()
        
        for provider in config.providers:
            if provider.name == name:
                for key, value in updates.items():
                    if hasattr(provider, key):
                        setattr(provider, key, value)
                self.save_config()
                return True
        return False
    
    def add_provider(self, provider: ProviderConfig) -> None:
        """Add new provider configuration"""
        config = self.load_config()
        
        # Remove existing provider with same name
        config.providers = [p for p in config.providers if p.name != provider.name]
        config.providers.append(provider)
        self.save_config()
    
    def remove_provider(self, name: str) -> bool:
        """Remove provider configuration"""
        config = self.load_config()
        original_count = len(config.providers)
        config.providers = [p for p in config.providers if p.name != name]
        
        if len(config.providers) < original_count:
            self.save_config()
            return True
        return False
    
    def _create_default_config(self) -> UnifiedConfig:
        """Create default configuration with common providers"""
        providers = [
            ProviderConfig(
                name="lmstudio-local",
                type=ProviderType.LMSTUDIO,
                endpoint="http://localhost:1234/v1",
                models=["llama-3.2-3b", "deepseek-coder", "mistral-7b"],
                priority=90,
                cost_per_token=0.0  # Local models are free
            ),
            ProviderConfig(
                name="openai-gpt4",
                type=ProviderType.OPENAI,
                endpoint="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
                models=["gpt-4o", "gpt-4o-mini"],
                priority=80,
                cost_per_token=0.03,  # Approximate
                enabled=bool(os.getenv("OPENAI_API_KEY"))
            ),
            ProviderConfig(
                name="anthropic-claude",
                type=ProviderType.ANTHROPIC,
                endpoint="https://api.anthropic.com",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                models=["claude-3-5-sonnet", "claude-3-haiku"],
                priority=85,
                cost_per_token=0.015,  # Approximate
                enabled=bool(os.getenv("ANTHROPIC_API_KEY"))
            )
        ]
        
        return UnifiedConfig(providers=providers)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> UnifiedConfig:
        """Convert dictionary to UnifiedConfig object"""
        # Convert providers
        providers = []
        for p_data in data.get("providers", []):
            provider = ProviderConfig(
                name=p_data["name"],
                type=ProviderType(p_data["type"]),
                endpoint=p_data["endpoint"],
                api_key=p_data.get("api_key"),
                models=p_data.get("models", []),
                max_concurrent=p_data.get("max_concurrent", 3),
                timeout=p_data.get("timeout", 30.0),
                cost_per_token=p_data.get("cost_per_token", 0.0001),
                enabled=p_data.get("enabled", True),
                priority=p_data.get("priority", 50)
            )
            providers.append(provider)
        
        # Convert cognitive config
        cognitive_data = data.get("cognitive", {})
        cognitive = CognitiveConfig(**{
            k: v for k, v in cognitive_data.items()
            if k in CognitiveConfig.__dataclass_fields__
        })
        
        # Convert MCP config
        mcp_data = data.get("mcp", {})
        mcp = MCPConfig(**{
            k: v for k, v in mcp_data.items()
            if k in MCPConfig.__dataclass_fields__
        })
        
        # Convert cost config
        cost_data = data.get("cost", {})
        cost = CostConfig(**{
            k: v for k, v in cost_data.items()
            if k in CostConfig.__dataclass_fields__
        })
        
        # Convert strategy mappings
        domain_strategies = {}
        for domain, strategy_str in data.get("domain_strategies", {}).items():
            domain_strategies[domain] = StrategyType(strategy_str)
            
        complexity_routing = {}
        for complexity, strategy_str in data.get("complexity_routing", {}).items():
            complexity_routing[complexity] = StrategyType(strategy_str)
        
        return UnifiedConfig(
            providers=providers,
            cognitive=cognitive,
            mcp=mcp,
            cost=cost,
            domain_strategies=domain_strategies,
            complexity_routing=complexity_routing
        )
    
    def _config_to_dict(self, config: UnifiedConfig) -> Dict[str, Any]:
        """Convert UnifiedConfig to dictionary for serialization"""
        # Convert providers
        providers_data = []
        for provider in config.providers:
            p_dict = asdict(provider)
            p_dict["type"] = provider.type.value
            providers_data.append(p_dict)
        
        # Convert strategy enums to strings
        domain_strategies = {k: v.value for k, v in config.domain_strategies.items()}
        complexity_routing = {k: v.value for k, v in config.complexity_routing.items()}
        
        return {
            "providers": providers_data,
            "cognitive": asdict(config.cognitive),
            "mcp": asdict(config.mcp),
            "cost": asdict(config.cost),
            "domain_strategies": domain_strategies,
            "complexity_routing": complexity_routing
        }


# Global configuration manager instance
_config_manager: Optional[UnifiedConfigManager] = None


def get_config() -> UnifiedConfig:
    """Get the global unified configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = UnifiedConfigManager()
    return _config_manager.load_config()


def get_config_manager() -> UnifiedConfigManager:
    """Get the global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = UnifiedConfigManager()
    return _config_manager


def reload_config() -> UnifiedConfig:
    """Force reload of configuration from file"""
    global _config_manager
    if _config_manager is not None:
        _config_manager._config = None
    return get_config()