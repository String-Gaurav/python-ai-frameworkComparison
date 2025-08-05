"""
Configuration management for LLM Test Framework
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any


class Config:
    """Centralized configuration management"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_default_config()
        if self.config_file.exists():
            self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Default configuration values"""
        return {
            "models": {
                "ollama": {
                    "default_models": ["llama3.2:latest", "mistral:7b"],
                    "timeout": 120,
                    "max_retries": 3
                },
                "openai": {
                    "models": ["gpt-3.5-turbo", "gpt-4"],
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "timeout": 60
                },
                "anthropic": {
                    "models": ["claude-3-sonnet", "claude-3-haiku"],
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    "timeout": 60
                }
            },
            "testing": {
                "categories": ["math", "programming", "creative", "knowledge", "reasoning"],
                "parallel_execution": False,
                "max_workers": 3,
                "default_difficulty": "medium"
            },
            "scoring": {
                "weights": {
                    "length_score": 0.2,
                    "code_quality": 0.3,
                    "confidence": 0.25,
                    "creativity": 0.25,
                    "math_content": 0.3
                }
            },
            "output": {
                "directory": "test_results",
                "formats": ["csv", "html", "json", "markdown"],
                "include_timestamps": True
            },
            "database": {
                "name": "llm_test_results.db",
                "backup_enabled": True,
                "retention_days": 30
            }
        }
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                user_config = json.load(f)
            self._merge_config(user_config)
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def _merge_config(self, user_config: Dict[str, Any]):
        """Merge user configuration with defaults"""
        def merge_dict(default: dict, user: dict):
            for key, value in user.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_dict(default[key], value)
                else:
                    default[key] = value
        
        merge_dict(self.config, user_config)
    
    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'models.ollama.timeout')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value


# Global configuration instance
config = Config()