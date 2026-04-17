"""Plotting utilities for visualization."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def plot_training_history(history: Dict[str, List[float]], output_path: str = None):
    """Plot training history.
    
    Args:
        history: Dictionary with training metrics
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, len(history), figsize=(15, 4))

    if len(history) == 1:
        axes = [axes]

    for ax, (key, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_title(key)
        ax.set_xlabel("Episode")
        ax.set_ylabel(key)
        ax.grid(True)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")

    plt.close()


def plot_evaluation_results(results: Dict[str, float], output_path: str = None):
    """Plot evaluation results.
    
    Args:
        results: Dictionary with evaluation metrics
        output_path: Path to save plot
    """
    labels = ["Wins", "Losses", "Draws"]
    values = [results["win_rate"], results["loss_rate"], results["draw_rate"]]
    colors = ["green", "red", "blue"]

    plt.figure(figsize=(8, 6))
    plt.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.title("Agent Performance")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")

    plt.close()


def plot_performance_comparison(
    performance_dict: Dict[str, Dict[str, float]], output_path: str = None
):
    """Plot performance comparison across agents.
    
    Args:
        performance_dict: Dictionary with agent performance
        output_path: Path to save plot
    """
    agents = list(performance_dict.keys())
    win_rates = [performance_dict[agent]["win_rate"] for agent in agents]
    loss_rates = [performance_dict[agent]["loss_rate"] for agent in agents]
    draw_rates = [performance_dict[agent]["draw_rate"] for agent in agents]

    x = np.arange(len(agents))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, win_rates, width, label="Wins")
    ax.bar(x, loss_rates, width, label="Losses")
    ax.bar(x + width, draw_rates, width, label="Draws")

    ax.set_xlabel("Agent")
    ax.set_ylabel("Rate")
    ax.set_title("Agent Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.legend()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")

    plt.close()
