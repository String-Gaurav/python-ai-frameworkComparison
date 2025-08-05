"""
Additional LLM Model Providers
Extends the framework to support OpenAI, Anthropic, and other providers
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from config import config

logger = logging.getLogger(__name__)


class BaseModelProvider(ABC):
    """Base class for all model providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.timeout = config.get(f"models.{name}.timeout", 60)
        self.max_retries = config.get(f"models.{name}.max_retries", 3)
    
    @abstractmethod
    def generate_response(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """Generate a response from the model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass
    
    @abstractmethod
    def list_models(self) -> list:
        """List available models for this provider"""
        pass


class OpenAIProvider(BaseModelProvider):
    """OpenAI API provider"""
    
    def __init__(self):
        super().__init__("openai")
        self.api_key = config.get("models.openai.api_key")
        self.client = None
        
        if self.api_key and OPENAI_AVAILABLE:
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and self.client is not None and self.api_key is not None
    
    def list_models(self) -> list:
        return config.get("models.openai.models", ["gpt-3.5-turbo", "gpt-4"])
    
    def generate_response(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available")
        
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout
                )
                
                response_time = time.time() - start_time
                
                return {
                    "response": response.choices[0].message.content,
                    "response_time": response_time,
                    "metadata": {
                        "provider": "openai",
                        "model": model_name,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    }
                }
            
            except Exception as e:
                logger.warning(f"OpenAI API attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError("All OpenAI API attempts failed")


class AnthropicProvider(BaseModelProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self):
        super().__init__("anthropic")
        self.api_key = config.get("models.anthropic.api_key")
        self.client = None
        
        if self.api_key and ANTHROPIC_AVAILABLE:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
    
    def is_available(self) -> bool:
        return ANTHROPIC_AVAILABLE and self.client is not None and self.api_key is not None
    
    def list_models(self) -> list:
        return config.get("models.anthropic.models", ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"])
    
    def generate_response(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """Generate response using Anthropic API"""
        if not self.is_available():
            raise RuntimeError("Anthropic provider not available")
        
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=model_name,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                response_time = time.time() - start_time
                
                return {
                    "response": response.content[0].text,
                    "response_time": response_time,
                    "metadata": {
                        "provider": "anthropic",
                        "model": model_name,
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens
                        }
                    }
                }
            
            except Exception as e:
                logger.warning(f"Anthropic API attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError("All Anthropic API attempts failed")


class LocalProvider(BaseModelProvider):
    """Provider for local models (Ollama, etc.)"""
    
    def __init__(self):
        super().__init__("local")
        try:
            import ollama
            self.ollama = ollama
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("Ollama not available - local models disabled")
    
    def is_available(self) -> bool:
        return self.available
    
    def list_models(self) -> list:
        if not self.is_available():
            return []
        
        try:
            models = self.ollama.list()
            return [model['name'] for model in models.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return config.get("models.ollama.default_models", [])
    
    def generate_response(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """Generate response using local Ollama model"""
        if not self.is_available():
            raise RuntimeError("Local provider not available")
        
        start_time = time.time()
        
        try:
            response = self.ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_time = time.time() - start_time
            
            return {
                "response": response['message']['content'],
                "response_time": response_time,
                "metadata": {
                    "provider": "local",
                    "model": model_name
                }
            }
        
        except Exception as e:
            logger.error(f"Local model generation failed: {e}")
            raise


# Provider registry
PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "local": LocalProvider
}


def get_provider(provider_name: str) -> BaseModelProvider:
    """Get a provider instance by name"""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    return PROVIDERS[provider_name]()


def list_available_providers() -> Dict[str, bool]:
    """List all providers and their availability"""
    return {
        name: get_provider(name).is_available()
        for name in PROVIDERS.keys()
    }