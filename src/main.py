#!/usr/bin/env python
"""
Main orchestrator for AGVR experiments.
Loads Hydra configuration and launches training subprocess.

Fixed issues:
1. Properly resolves run configs from config/runs/ directory
2. Applies mode-specific configuration overrides (trial vs full)
3. Validates all required configuration keys
4. Launches train.py as subprocess with correct parameters
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for experiment orchestration.
    
    Loads configuration from Hydra, validates inputs, applies mode-specific settings,
    and launches train.py as subprocess.
    """
    
    # Validate required parameters
    if cfg.run is None:
        log.error("run_id must be specified via run parameter")
        sys.exit(1)
    
    if cfg.results_dir is None:
        log.error("results_dir must be specified via results_dir parameter")
        sys.exit(1)
    
    if cfg.mode is None:
        log.error("mode must be specified (trial or full)")
        sys.exit(1)
    
    if cfg.mode not in ["trial", "full"]:
        log.error(f"Invalid mode: {cfg.mode}. Must be 'trial' or 'full'")
        sys.exit(1)
    
    log.info("="*70)
    log.info("AGVR Experiment Orchestrator")
    log.info("="*70)
    log.info(f"Run ID: {cfg.run}")
    log.info(f"Mode: {cfg.mode}")
    log.info(f"Results Dir: {cfg.results_dir}")
    
    # Create results directory
    results_path = Path(cfg.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Resolve run config path with absolute path resolution
    repo_root = Path(__file__).parent.parent
    run_config_path = repo_root / "config" / "runs" / f"{cfg.run}.yaml"
    
    if not run_config_path.exists():
        log.error(f"Run configuration not found: {run_config_path}")
        runs_dir = repo_root / "config" / "runs"
        if runs_dir.exists():
            available_runs = [f.stem for f in runs_dir.glob("*.yaml")]
            log.error(f"Available runs: {available_runs}")
        else:
            log.error(f"Runs directory does not exist: {runs_dir}")
        sys.exit(1)
    
    log.info(f"Loading run configuration: {run_config_path}")
    run_cfg = OmegaConf.load(run_config_path)
    
    # Validate required keys
    required_keys = ["method", "model", "dataset", "training"]
    for key in required_keys:
        if key not in run_cfg:
            log.error(f"Run config missing required key: {key}")
            sys.exit(1)
    
    # Apply mode-specific overrides
    if cfg.mode == "trial":
        log.info("Trial mode: applying lightweight configuration")
        run_cfg.training.epochs = 1
        run_cfg.wandb.mode = "disabled"
        if "optuna" in run_cfg:
            run_cfg.optuna.n_trials = 0
        run_cfg.training.trial_batches = 2
        assert run_cfg.wandb.mode == "disabled", "WandB should be disabled in trial mode"
    elif cfg.mode == "full":
        log.info("Full mode: using complete configuration")
        run_cfg.wandb.mode = "online"
        if "optuna" in run_cfg:
            run_cfg.optuna.n_trials = run_cfg.optuna.get("n_trials", 20)
        run_cfg.training.trial_batches = None
        assert run_cfg.wandb.mode == "online", "WandB should be online in full mode"
    
    # Merge with main config
    merged_cfg = OmegaConf.merge(cfg, run_cfg)
    
    # Save resolved config
    config_output = results_path / f"config_{cfg.run}.yaml"
    with open(config_output, "w") as f:
        OmegaConf.save(merged_cfg, f)
    log.info(f"Saved resolved configuration: {config_output}")
    
    # Launch training subprocess
    log.info("Launching training subprocess...")
    
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    
    log.info(f"Command: {' '.join(cmd)}")
    
    # Execute training from repository root
    result = subprocess.run(cmd, cwd=repo_root)
    
    if result.returncode != 0:
        log.error(f"Training subprocess failed with return code {result.returncode}")
        sys.exit(result.returncode)
    
    log.info(f"Training completed successfully for run {cfg.run}")


if __name__ == "__main__":
    main()
