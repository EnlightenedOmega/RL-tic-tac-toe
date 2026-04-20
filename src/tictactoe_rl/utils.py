"""Utility functions."""

import pickle
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_model(model_path: str):
    """Load trained model.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded agent (agent_x if multiple agents saved)
    """
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    # Handle both single agent and multi-agent saves
    if isinstance(data, dict) and "agent_x" in data:
        # Multi-agent save from trainer
        return data["agent_x"]
    else:
        # Single agent save
        return data


def save_model(agent, model_path: str):
    """Save trained agent.
    
    Args:
        agent: Agent to save
        model_path: Path to save model
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(agent, f)
