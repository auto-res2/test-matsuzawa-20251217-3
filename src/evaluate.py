#!/usr/bin/env python
"""
Independent evaluation and visualization script.
Retrieves comprehensive experimental data from WandB and generates comparison figures.

Executes INDEPENDENTLY via:
  uv run python -m src.evaluate results_dir={path} run_ids='["run-1", "run-2"]'

NOT called from main.py - executes as separate workflow after training completes.
"""

import json
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from omegaconf import OmegaConf

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate AGVR experiments from WandB"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        required=True,
        help='JSON string list of run IDs (e.g., \'["run-1", "run-2"]\')'
    )
    return parser.parse_args()


def load_wandb_config() -> Dict[str, Any]:
    """Load WandB configuration from config/config.yaml."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        log.warning(f"Config file not found: {config_path}")
        return {
            "entity": "gengaru617-personal",
            "project": "2025-11-19",
        }
    
    cfg = OmegaConf.load(config_path)
    return {
        "entity": cfg.wandb.entity,
        "project": cfg.wandb.project,
    }


def fetch_wandb_data(run_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch comprehensive data from WandB for all runs."""
    if not HAS_WANDB:
        log.error("WandB not installed")
        return {}
    
    config = load_wandb_config()
    api = wandb.Api()
    
    runs_data = {}
    
    for run_id in run_ids:
        log.info(f"Fetching data for run: {run_id}")
        
        try:
            run_path = f"{config['entity']}/{config['project']}/{run_id}"
            run = api.run(run_path)
            
            # Get history (time-series metrics) as DataFrame
            history = run.history()
            
            # Get summary (final/best metrics)
            summary = run.summary._json_dict if hasattr(run.summary, "_json_dict") else {}
            
            # Get config
            run_config = dict(run.config)
            
            runs_data[run_id] = {
                "history": history,
                "summary": summary,
                "config": run_config,
            }
            
            log.info(f"  ✓ Fetched {len(history)} time-steps, {len(summary)} summary metrics")
        
        except Exception as e:
            log.error(f"  ✗ Failed to fetch run {run_id}: {e}")
    
    return runs_data


def process_run_metrics(run_id: str, run_data: Dict[str, Any], 
                       results_dir: Path) -> Dict[str, Any]:
    """Process metrics for a single run and save to JSON."""
    
    # Create run-specific directory
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    history = run_data["history"]
    summary = run_data["summary"]
    
    # Extract relevant metrics from history (handles pd.DataFrame properly)
    metrics_dict = {
        "run_id": run_id,
        "summary": summary,
        "history": {}
    }
    
    # Extract time-series data from DataFrame
    if not history.empty:
        metrics_dict["history"]["epochs"] = list(range(len(history)))
        
        # Extract columns with defensive checks
        for col in ["train/epoch_loss", "val/epoch_loss"]:
            if col in history.columns:
                metrics_dict["history"][col] = history[col].tolist()
        
        for col in ["val/accuracy"]:
            if col in history.columns:
                metrics_dict["history"][col] = history[col].tolist()
        
        for col in ["val/perplexity"]:
            if col in history.columns:
                metrics_dict["history"][col] = history[col].tolist()
        
        for col in ["optimizer/rho_adaptive", "optimizer/lambda_t"]:
            if col in history.columns:
                metrics_dict["history"][col] = history[col].tolist()
    
    # Save to JSON
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    
    log.info(f"Saved metrics to {metrics_file}")
    print(f"{metrics_file}")
    
    # Generate per-run figures
    generate_run_figures(run_id, metrics_dict, run_dir)
    
    return metrics_dict


def generate_run_figures(run_id: str, metrics: Dict[str, Any], 
                        output_dir: Path) -> None:
    """Generate per-run visualization figures."""
    
    history = metrics["history"]
    
    # Only generate if we have data
    if not history or all(len(v) == 0 if isinstance(v, list) else True for v in history.values()):
        log.warning(f"Skipping figure generation for {run_id} - no history data")
        return
    
    epochs = history.get("epochs", list(range(max(len(v) for v in history.values() if isinstance(v, list)))))
    
    # 1. Learning curves
    has_loss_data = "train/epoch_loss" in history or "val/epoch_loss" in history
    if has_loss_data and len(epochs) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        if "train/epoch_loss" in history and len(history["train/epoch_loss"]) > 0:
            axes[0].plot(epochs[:len(history["train/epoch_loss"])], 
                        history["train/epoch_loss"], label="Train Loss", 
                        marker="o", linewidth=2)
        if "val/epoch_loss" in history and len(history["val/epoch_loss"]) > 0:
            axes[0].plot(epochs[:len(history["val/epoch_loss"])], 
                        history["val/epoch_loss"], label="Val Loss", 
                        marker="s", linewidth=2)
        
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].set_title(f"{run_id} - Training Dynamics", fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Annotate final values
        if "val/epoch_loss" in history and len(history["val/epoch_loss"]) > 0:
            axes[0].text(len(history["val/epoch_loss"])-1, history["val/epoch_loss"][-1], 
                        f'{history["val/epoch_loss"][-1]:.4f}', ha='right', 
                        fontsize=9, fontweight='bold')
        
        # Accuracy or perplexity
        if "val/accuracy" in history and len(history["val/accuracy"]) > 0:
            axes[1].plot(epochs[:len(history["val/accuracy"])], 
                        history["val/accuracy"], label="Val Accuracy", 
                        marker="o", color="green", linewidth=2)
            axes[1].set_ylabel("Accuracy", fontsize=11)
            axes[1].text(len(history["val/accuracy"])-1, history["val/accuracy"][-1], 
                        f'{history["val/accuracy"][-1]:.4f}', ha='right', 
                        fontsize=9, fontweight='bold')
        elif "val/perplexity" in history and len(history["val/perplexity"]) > 0:
            axes[1].plot(epochs[:len(history["val/perplexity"])], 
                        history["val/perplexity"], label="Val Perplexity", 
                        marker="o", color="red", linewidth=2)
            axes[1].set_ylabel("Perplexity", fontsize=11)
            axes[1].text(len(history["val/perplexity"])-1, history["val/perplexity"][-1], 
                        f'{history["val/perplexity"][-1]:.4f}', ha='right', 
                        fontsize=9, fontweight='bold')
        
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_title(f"{run_id} - Primary Metric", fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = output_dir / f"{run_id}_learning_curve.pdf"
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        log.info(f"Saved figure: {fig_path}")
        print(f"{fig_path}")
        plt.close()
    
    # 2. Adaptive threshold trajectory (if AGVR)
    if "optimizer/rho_adaptive" in history and len(history["optimizer/rho_adaptive"]) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # ρ_adaptive trajectory
        axes[0].plot(epochs[:len(history["optimizer/rho_adaptive"])], 
                    history["optimizer/rho_adaptive"], 
                    label="ρ_adaptive_t", marker="o", color="blue", linewidth=2)
        axes[0].axhline(y=4.0, color="red", linestyle="--", linewidth=2, 
                       label="RAdam ρ=4")
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("ρ_adaptive_t", fontsize=11)
        axes[0].set_title(f"{run_id} - Adaptive Threshold Trajectory", 
                         fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        if len(history["optimizer/rho_adaptive"]) > 0:
            axes[0].text(len(history["optimizer/rho_adaptive"])-1, 
                        history["optimizer/rho_adaptive"][-1], 
                        f'{history["optimizer/rho_adaptive"][-1]:.4f}', 
                        ha='right', fontsize=9, fontweight='bold')
        
        # λ_t trajectory
        if "optimizer/lambda_t" in history and len(history["optimizer/lambda_t"]) > 0:
            axes[1].plot(epochs[:len(history["optimizer/lambda_t"])], 
                        history["optimizer/lambda_t"], 
                        label="λ_t (Gradient Variance Maturity)", marker="s", 
                        color="green", linewidth=2)
            axes[1].set_xlabel("Epoch", fontsize=11)
            axes[1].set_ylabel("λ_t", fontsize=11)
            axes[1].set_title(f"{run_id} - Gradient Variance Signal", 
                             fontsize=12, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            if len(history["optimizer/lambda_t"]) > 0:
                axes[1].text(len(history["optimizer/lambda_t"])-1, 
                            history["optimizer/lambda_t"][-1], 
                            f'{history["optimizer/lambda_t"][-1]:.4f}', 
                            ha='right', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        fig_path = output_dir / f"{run_id}_adaptive_threshold.pdf"
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        log.info(f"Saved figure: {fig_path}")
        print(f"{fig_path}")
        plt.close()


def aggregate_metrics(all_runs_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all runs."""
    
    aggregated = {
        "metrics": defaultdict(dict)
    }
    
    # Collect primary metrics
    primary_metric_best_proposed = None
    primary_metric_best_baseline = None
    proposed_run_id = None
    baseline_run_id = None
    
    for run_id, run_data in all_runs_data.items():
        summary = run_data["summary"]
        
        # Collect all metrics
        for metric_name, metric_value in summary.items():
            if isinstance(metric_value, (int, float)):
                aggregated["metrics"][metric_name][run_id] = metric_value
        
        # Identify primary metric (validation accuracy or perplexity)
        is_proposed = "agvr" in run_id.lower() or "proposed" in run_id.lower()
        
        # Check for test accuracy
        if "test/accuracy" in summary or "final_test_accuracy" in summary:
            metric_value = summary.get("test/accuracy") or summary.get("final_test_accuracy")
            if metric_value is not None:
                if is_proposed:
                    if primary_metric_best_proposed is None or metric_value > primary_metric_best_proposed:
                        primary_metric_best_proposed = metric_value
                        proposed_run_id = run_id
                else:
                    if primary_metric_best_baseline is None or metric_value > primary_metric_best_baseline:
                        primary_metric_best_baseline = metric_value
                        baseline_run_id = run_id
        
        # Check for test perplexity
        elif "test/perplexity" in summary or "final_test_perplexity" in summary:
            metric_value = summary.get("test/perplexity") or summary.get("final_test_perplexity")
            if metric_value is not None:
                if is_proposed:
                    if primary_metric_best_proposed is None or metric_value < primary_metric_best_proposed:
                        primary_metric_best_proposed = metric_value
                        proposed_run_id = run_id
                else:
                    if primary_metric_best_baseline is None or metric_value < primary_metric_best_baseline:
                        primary_metric_best_baseline = metric_value
                        baseline_run_id = run_id
    
    # Determine if we're dealing with accuracy or perplexity
    is_accuracy = any("test/accuracy" in rd.get("summary", {}) or "final_test_accuracy" in rd.get("summary", {})
                     for rd in all_runs_data.values())
    
    # Calculate gap
    gap = None
    if primary_metric_best_proposed is not None and primary_metric_best_baseline is not None:
        if is_accuracy:
            # For accuracy, higher is better
            gap = (primary_metric_best_proposed - primary_metric_best_baseline) / primary_metric_best_baseline * 100
        else:
            # For perplexity, lower is better
            gap = (primary_metric_best_baseline - primary_metric_best_proposed) / primary_metric_best_baseline * 100
    
    aggregated["primary_metric"] = (
        "validation_accuracy (for image classification tasks CIFAR-10, MNIST, CIFAR-100, "
        "reported as accuracy percentage with higher values indicating better performance; "
        "for language modeling Penn Treebank, validation_perplexity with lower values "
        "indicating better performance)"
    )
    
    if primary_metric_best_proposed is not None:
        aggregated["best_proposed"] = {
            "run_id": proposed_run_id,
            "value": float(primary_metric_best_proposed),
        }
    
    if primary_metric_best_baseline is not None:
        aggregated["best_baseline"] = {
            "run_id": baseline_run_id,
            "value": float(primary_metric_best_baseline),
        }
    
    if gap is not None:
        aggregated["gap"] = gap
    
    return aggregated


def generate_comparison_figures(all_runs_data: Dict[str, Dict[str, Any]], 
                               results_dir: Path) -> None:
    """Generate comparison figures across all runs."""
    
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize data by metric
    metrics_by_name = defaultdict(dict)
    
    for run_id, run_data in all_runs_data.items():
        summary = run_data["summary"]
        for metric_name, metric_value in summary.items():
            if isinstance(metric_value, (int, float)):
                metrics_by_name[metric_name][run_id] = metric_value
    
    # 1. Comparison bar chart for primary metric
    primary_metrics = [m for m in metrics_by_name.keys() 
                      if "test/accuracy" in m or "test/perplexity" in m or "final_test" in m]
    
    for primary_metric in primary_metrics:
        if primary_metric in metrics_by_name:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            runs = list(metrics_by_name[primary_metric].keys())
            values = list(metrics_by_name[primary_metric].values())
            
            is_proposed = ["agvr" in r.lower() or "proposed" in r.lower() for r in runs]
            colors = ["#2E86AB" if p else "#A23B72" for p in is_proposed]
            
            bars = ax.bar(range(len(runs)), values, color=colors, alpha=0.8, 
                         edgecolor='black', linewidth=1.5)
            ax.set_xticks(range(len(runs)))
            ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=10)
            ax.set_ylabel(primary_metric, fontsize=12, fontweight='bold')
            ax.set_title(f"Comparison: {primary_metric}", fontsize=14, fontweight='bold')
            
            # Annotate values on bars
            for i, (bar, v) in enumerate(zip(bars, values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{v:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            fig_path = comparison_dir / f"comparison_{primary_metric.replace('/', '_')}_bar.pdf"
            plt.savefig(fig_path, dpi=100, bbox_inches="tight")
            log.info(f"Saved figure: {fig_path}")
            print(f"{fig_path}")
            plt.close()
    
    # 2. Aggregated metrics table
    if metrics_by_name:
        df_data = []
        for metric_name in sorted(metrics_by_name.keys()):
            row = {"metric": metric_name}
            for run_id in sorted(metrics_by_name[metric_name].keys()):
                row[run_id] = metrics_by_name[metric_name][run_id]
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        
        table = ax.table(cellText=df.round(4).values, colLabels=df.columns,
                        cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        plt.title("Aggregated Metrics Across All Runs", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        fig_path = comparison_dir / "comparison_metrics_table.pdf"
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        log.info(f"Saved figure: {fig_path}")
        print(f"{fig_path}")
        plt.close()


def main():
    """Main evaluation entry point."""
    args = parse_arguments()
    
    # Parse run_ids
    try:
        run_ids = json.loads(args.run_ids)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse run_ids JSON: {e}")
        return
    
    log.info(f"Evaluating {len(run_ids)} runs: {run_ids}")
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch WandB data
    log.info("Fetching data from WandB...")
    all_runs_data = fetch_wandb_data(run_ids)
    
    if not all_runs_data:
        log.error("No runs fetched from WandB")
        return
    
    log.info(f"Successfully fetched {len(all_runs_data)} runs")
    
    # STEP 1: Per-run processing
    log.info("\nProcessing per-run metrics...")
    all_metrics = {}
    for run_id, run_data in all_runs_data.items():
        metrics = process_run_metrics(run_id, run_data, results_dir)
        all_metrics[run_id] = metrics
    
    # STEP 2: Aggregated analysis
    log.info("\nAggregating metrics across runs...")
    aggregated = aggregate_metrics(all_runs_data)
    
    # Save aggregated metrics
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated_file = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_file, "w") as f:
        # Convert defaultdict to regular dict for JSON serialization
        aggregated_dict = {
            "primary_metric": aggregated["primary_metric"],
            "metrics": dict(aggregated["metrics"]),
            "best_proposed": aggregated.get("best_proposed"),
            "best_baseline": aggregated.get("best_baseline"),
            "gap": aggregated.get("gap"),
        }
        json.dump(aggregated_dict, f, indent=2)
    
    log.info(f"Saved aggregated metrics to {aggregated_file}")
    print(f"{aggregated_file}")
    
    # Generate comparison figures
    log.info("\nGenerating comparison figures...")
    generate_comparison_figures(all_runs_data, results_dir)
    
    # Print summary
    log.info("\n" + "="*60)
    log.info("EVALUATION SUMMARY")
    log.info("="*60)
    
    if "best_proposed" in aggregated and aggregated["best_proposed"]:
        log.info(f"Best Proposed: {aggregated['best_proposed']['run_id']} = {aggregated['best_proposed']['value']:.4f}")
    
    if "best_baseline" in aggregated and aggregated["best_baseline"]:
        log.info(f"Best Baseline: {aggregated['best_baseline']['run_id']} = {aggregated['best_baseline']['value']:.4f}")
    
    if "gap" in aggregated and aggregated["gap"] is not None:
        log.info(f"Performance Gap: {aggregated['gap']:.2f}%")
    
    log.info(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
