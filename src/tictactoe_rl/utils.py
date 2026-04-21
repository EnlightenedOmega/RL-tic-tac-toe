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
    """Load trained agent model.
    
    Supports loading from split files (agent_x.pkl and agent_o.pkl) or 
    a single legacy pickle file.
    
    Args:
        model_path: Path to model file. If split files exist, will load agent_x.
                    For split files with base path 'artifacts/models/agent.pkl',
                    will look for 'artifacts/models/agent_x.pkl'
        
    Returns:
        Loaded agent (agent_x for split files, or the saved agent for single file)
    """
    # Try to load from split files first
    base_path = str(model_path).rsplit(".", 1)[0] if "." in str(model_path) else str(model_path)
    path_x = f"{base_path}_x.pkl"
    
    # Check if split files exist
    if Path(path_x).exists():
        with open(path_x, "rb") as f:
            return pickle.load(f)
    
    # Fallback to legacy single file format
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    # Handle both single agent and multi-agent saves
    if isinstance(data, dict) and "agent_x" in data:
        # Multi-agent save from old trainer format
        return data["agent_x"]
    else:
        # Single agent save
        return data


def load_both_models(model_path: str):
    """Load both trained agents from split files.
    
    Args:
        model_path: Base path to model files. Will load from model_path_x.pkl 
                    and model_path_o.pkl. E.g., 'artifacts/models/agent.pkl' 
                    will load 'agent_x.pkl' and 'agent_o.pkl'
        
    Returns:
        Tuple of (agent_x, agent_o)
    """
    base_path = str(model_path).rsplit(".", 1)[0] if "." in str(model_path) else str(model_path)
    path_x = f"{base_path}_x.pkl"
    path_o = f"{base_path}_o.pkl"
    
    with open(path_x, "rb") as f:
        agent_x = pickle.load(f)
    
    with open(path_o, "rb") as f:
        agent_o = pickle.load(f)
    
    return agent_x, agent_o


def save_model(agent, model_path: str):
    """Save trained agent.
    
    Args:
        agent: Agent to save
        model_path: Path to save model
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(agent, f)
