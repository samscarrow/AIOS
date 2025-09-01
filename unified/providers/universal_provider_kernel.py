#!/usr/bin/env python3
"""
Universal Provider Kernel
Real multi-provider execution engine replacing mock implementations
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging

from ..config import (
    UnifiedConfig, ProviderConfig, ProviderType, get_config
)

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider connection status"""
    UNKNOWN = "unknown"
    CONNECTED = "connected" 
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class ModelResponse:
    """Response from a provider model"""
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ProviderError:
    """Error information from a provider"""
    provider: str
    error_type: str
    message: str
    recoverable: bool = True
    retry_after: Optional[float] = None


class ProviderKernel:
    """Base class for provider kernels"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.status = ProviderStatus.UNKNOWN
        self.last_error: Optional[str] = None
        self.requests_made = 0
        self.total_cost = 0.0
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
        
    async def connect(self) -> None:
        """Connect to the provider"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        
        try:
            await self.health_check()
            self.status = ProviderStatus.CONNECTED
            logger.info(f"Connected to {self.config.name}")
        except Exception as e:
            self.status = ProviderStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to connect to {self.config.name}: {e}")
            
    async def disconnect(self) -> None:
        """Disconnect from the provider"""
        if self.session:
            await self.session.close()
            self.session = None
        self.status = ProviderStatus.DISCONNECTED
        
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        try:
            # Basic connectivity test - implement in subclasses
            return await self._health_check_impl()
        except Exception as e:
            logger.error(f"Health check failed for {self.config.name}: {e}")
            return False
            
    async def _health_check_impl(self) -> bool:
        """Provider-specific health check implementation"""
        raise NotImplementedError("Subclasses must implement health check")
        
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> ModelResponse:
        """Generate response from the provider"""
        if self.status != ProviderStatus.CONNECTED:
            raise RuntimeError(f"Provider {self.config.name} not connected")
            
        start_time = time.perf_counter()
        
        try:
            response = await self._generate_impl(prompt, model, **kwargs)
            response.latency_ms = (time.perf_counter() - start_time) * 1000
            response.cost = response.tokens_used * self.config.cost_per_token
            
            self.requests_made += 1
            self.total_cost += response.cost
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed for {self.config.name}: {e}")
            raise ProviderError(
                provider=self.config.name,
                error_type=type(e).__name__,
                message=str(e)
            )
    
    async def _generate_impl(self, prompt: str, model: Optional[str], **kwargs) -> ModelResponse:
        """Provider-specific generation implementation"""
        raise NotImplementedError("Subclasses must implement generation")
    
    async def stream_generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> AsyncIterator[str]:
        """Stream response from the provider"""
        # Default implementation - can be overridden for true streaming
        response = await self.generate(prompt, model, **kwargs)
        yield response.content


class OpenAIKernel(ProviderKernel):
    """OpenAI API provider kernel"""
    
    async def _health_check_impl(self) -> bool:
        """Check OpenAI API availability"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.get(
                f"{self.config.endpoint}/models",
                headers=headers
            ) as response:
                return response.status == 200
                
        except Exception:
            return False
    
    async def _generate_impl(self, prompt: str, model: Optional[str], **kwargs) -> ModelResponse:
        """Generate using OpenAI API"""
        if not self.config.api_key:
            raise ValueError("OpenAI API key not configured")
            
        model = model or (self.config.models[0] if self.config.models else "gpt-3.5-turbo")
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        
        async with self.session.post(
            f"{self.config.endpoint}/chat/completions",
            headers=headers,
            json=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"OpenAI API error {response.status}: {error_text}")
            
            result = await response.json()
            
            return ModelResponse(
                content=result["choices"][0]["message"]["content"],
                model=model,
                provider=self.config.name,
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
                metadata={"finish_reason": result["choices"][0]["finish_reason"]}
            )


class AnthropicKernel(ProviderKernel):
    """Anthropic Claude API provider kernel"""
    
    async def _health_check_impl(self) -> bool:
        """Basic health check for Anthropic"""
        # Anthropic doesn't have a simple health endpoint
        # We'll just validate the configuration
        return self.config.api_key is not None
    
    async def _generate_impl(self, prompt: str, model: Optional[str], **kwargs) -> ModelResponse:
        """Generate using Anthropic API"""
        if not self.config.api_key:
            raise ValueError("Anthropic API key not configured")
            
        model = model or (self.config.models[0] if self.config.models else "claude-3-haiku-20240307")
        
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        async with self.session.post(
            f"{self.config.endpoint}/v1/messages",
            headers=headers,
            json=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Anthropic API error {response.status}: {error_text}")
            
            result = await response.json()
            
            return ModelResponse(
                content=result["content"][0]["text"],
                model=model,
                provider=self.config.name,
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
                metadata={"stop_reason": result.get("stop_reason")}
            )


class LMStudioKernel(ProviderKernel):
    """LM Studio local API provider kernel"""
    
    async def _health_check_impl(self) -> bool:
        """Check LM Studio server availability"""
        try:
            async with self.session.get(f"{self.config.endpoint}/models") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def _generate_impl(self, prompt: str, model: Optional[str], **kwargs) -> ModelResponse:
        """Generate using LM Studio API (OpenAI-compatible)"""
        model = model or (self.config.models[0] if self.config.models else "local-model")
        
        headers = {"Content-Type": "application/json"}
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "stream": False
        }
        
        async with self.session.post(
            f"{self.config.endpoint}/chat/completions",
            headers=headers,
            json=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"LM Studio error {response.status}: {error_text}")
            
            result = await response.json()
            
            return ModelResponse(
                content=result["choices"][0]["message"]["content"],
                model=model,
                provider=self.config.name,
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
                metadata={"finish_reason": result["choices"][0]["finish_reason"]}
            )


class OllamaKernel(ProviderKernel):
    """Ollama local API provider kernel"""
    
    async def _health_check_impl(self) -> bool:
        """Check Ollama server availability"""
        try:
            async with self.session.get(f"{self.config.endpoint}/api/tags") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def _generate_impl(self, prompt: str, model: Optional[str], **kwargs) -> ModelResponse:
        """Generate using Ollama API"""
        model = model or (self.config.models[0] if self.config.models else "llama2")
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1000)
            }
        }
        
        async with self.session.post(
            f"{self.config.endpoint}/api/generate",
            json=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Ollama error {response.status}: {error_text}")
            
            result = await response.json()
            
            return ModelResponse(
                content=result["response"],
                model=model,
                provider=self.config.name,
                tokens_used=0,  # Ollama doesn't provide token counts
                metadata={"done": result.get("done", True)}
            )


class MockKernel(ProviderKernel):
    """Mock provider for testing"""
    
    async def _health_check_impl(self) -> bool:
        """Mock health check always succeeds"""
        return True
    
    async def _generate_impl(self, prompt: str, model: Optional[str], **kwargs) -> ModelResponse:
        """Generate mock response"""
        await asyncio.sleep(0.1)  # Simulate API latency
        
        model = model or "mock-model"
        
        return ModelResponse(
            content=f"Mock response to: {prompt[:50]}...",
            model=model,
            provider=self.config.name,
            tokens_used=len(prompt.split()) + 10,  # Rough token estimate
            metadata={"mock": True}
        )


class UniversalProviderKernel:
    """Manages multiple provider kernels"""
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or get_config()
        self.kernels: Dict[str, ProviderKernel] = {}
        self.kernel_factories = {
            ProviderType.OPENAI: OpenAIKernel,
            ProviderType.ANTHROPIC: AnthropicKernel,
            ProviderType.LMSTUDIO: LMStudioKernel,
            ProviderType.OLLAMA: OllamaKernel,
            ProviderType.MOCK: MockKernel
        }
        
    async def initialize(self) -> None:
        """Initialize all enabled provider kernels"""
        for provider_config in self.config.providers:
            if not provider_config.enabled:
                continue
                
            kernel_class = self.kernel_factories.get(provider_config.type)
            if not kernel_class:
                logger.warning(f"No kernel implementation for provider type: {provider_config.type}")
                continue
                
            kernel = kernel_class(provider_config)
            self.kernels[provider_config.name] = kernel
            
            # Connect to provider
            try:
                await kernel.connect()
                logger.info(f"Initialized provider: {provider_config.name}")
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_config.name}: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown all provider kernels"""
        for kernel in self.kernels.values():
            try:
                await kernel.disconnect()
            except Exception as e:
                logger.error(f"Error shutting down kernel: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of connected provider names"""
        return [
            name for name, kernel in self.kernels.items()
            if kernel.status == ProviderStatus.CONNECTED
        ]
    
    def get_provider_models(self, provider_name: str) -> List[str]:
        """Get available models for a provider"""
        if provider_name in self.kernels:
            return self.kernels[provider_name].config.models
        return []
    
    async def generate(self, provider_name: str, prompt: str, 
                      model: Optional[str] = None, **kwargs) -> ModelResponse:
        """Generate response from specific provider"""
        if provider_name not in self.kernels:
            raise ValueError(f"Provider {provider_name} not found")
        
        kernel = self.kernels[provider_name]
        if kernel.status != ProviderStatus.CONNECTED:
            raise RuntimeError(f"Provider {provider_name} not connected")
        
        return await kernel.generate(prompt, model, **kwargs)
    
    async def generate_from_any(self, prompt: str, model: Optional[str] = None,
                               preferred_providers: Optional[List[str]] = None,
                               **kwargs) -> ModelResponse:
        """Generate response from any available provider"""
        # Get available providers, preferring specified ones
        available = self.get_available_providers()
        if not available:
            raise RuntimeError("No providers available")
        
        # Order by preference if specified
        if preferred_providers:
            ordered_providers = []
            for pref in preferred_providers:
                if pref in available:
                    ordered_providers.append(pref)
            # Add remaining providers
            for prov in available:
                if prov not in ordered_providers:
                    ordered_providers.append(prov)
            available = ordered_providers
        
        # Try providers in order
        last_error = None
        for provider_name in available:
            try:
                return await self.generate(provider_name, prompt, model, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
        # All providers failed
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers"""
        stats = {}
        for name, kernel in self.kernels.items():
            stats[name] = {
                "status": kernel.status.value,
                "requests_made": kernel.requests_made,
                "total_cost": kernel.total_cost,
                "last_error": kernel.last_error,
                "config": {
                    "type": kernel.config.type.value,
                    "endpoint": kernel.config.endpoint,
                    "models": kernel.config.models,
                    "priority": kernel.config.priority
                }
            }
        return stats
    
    @asynccontextmanager
    async def provider_context(self):
        """Async context manager for automatic initialization and cleanup"""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()


# Global kernel instance
_global_kernel: Optional[UniversalProviderKernel] = None


async def get_kernel() -> UniversalProviderKernel:
    """Get the global universal provider kernel"""
    global _global_kernel
    if _global_kernel is None:
        _global_kernel = UniversalProviderKernel()
        await _global_kernel.initialize()
    return _global_kernel


async def shutdown_kernel() -> None:
    """Shutdown the global kernel"""
    global _global_kernel
    if _global_kernel is not None:
        await _global_kernel.shutdown()
        _global_kernel = None