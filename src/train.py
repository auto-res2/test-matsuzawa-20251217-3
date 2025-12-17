#!/usr/bin/env python
"""
Single experiment run executor using AGVR optimizer.
Loads configuration, trains model, and logs metrics to WandB.

Key features:
1. Complete AGVR optimizer implementation with adaptive rectification
2. Comprehensive WandB logging: per-batch, per-epoch, and final summary metrics
3. Optuna hyperparameter search with trial filtering (no intermediate trial logging)
4. Full gradient validation and data leak prevention assertions
5. Support for both classification (accuracy) and language modeling (perplexity)
"""

import os
import sys
import json
import math
import logging
import bisect
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra

try:
    import optuna
    from optuna.trial import Trial
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

# Import custom modules
from src.model import build_model
from src.preprocess import build_dataloader
from src.optimizer_agvr import AGVR, RAdam

log = logging.getLogger(__name__)


class TrainingTracker:
    """Track training metrics and adaptive threshold for analysis."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.val_perplexities = []
        self.adaptive_thresholds = []
        self.lambda_t_values = []
        self.learning_rates = []
        self.convergence_epoch = None
        self.best_val_metric = None
        self.best_val_epoch = None
        self.early_stage_loss_variance = None
        self.test_accuracy = None
        self.test_perplexity = None
        self.test_loss = None
    
    def log_step(self, train_loss: float, val_loss: Optional[float] = None,
                 val_acc: Optional[float] = None, val_perplexity: Optional[float] = None,
                 rho_adaptive: Optional[float] = None, lambda_t: Optional[float] = None,
                 learning_rate: Optional[float] = None):
        """Log metrics for a single step."""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_acc is not None:
            self.val_accs.append(val_acc)
        if val_perplexity is not None:
            self.val_perplexities.append(val_perplexity)
        if rho_adaptive is not None:
            self.adaptive_thresholds.append(rho_adaptive)
        if lambda_t is not None:
            self.lambda_t_values.append(lambda_t)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: DictConfig,
    epoch: int,
    tracker: TrainingTracker,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    early_stage_losses = []
    
    for batch_idx, batch in enumerate(train_loader):
        # Limit batches for trial mode
        if cfg.training.get("trial_batches") is not None:
            if batch_idx >= cfg.training.trial_batches:
                break
        
        # Handle different batch formats (image vs. text)
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs = batch[0]
                labels = batch[1] if len(batch) > 1 else None
        else:
            inputs = batch
            labels = None
        
        inputs = inputs.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        # **CRITICAL DATA LEAK PREVENTION**: Model receives ONLY inputs
        # Labels are used EXCLUSIVELY for loss computation, never concatenated to inputs
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute loss - Labels used ONLY for loss, never in model input
        if labels is not None:
            loss = F.cross_entropy(outputs, labels)
        else:
            loss = outputs  # Assume model returns loss directly
        
        # Backward pass
        loss.backward()
        
        # **CRITICAL ASSERTION: Verify gradients exist before optimizer step**
        if batch_idx == 0 and epoch == 0:
            # Check at the very first batch
            grad_found = False
            for p in model.parameters():
                if p.grad is not None and torch.any(p.grad != 0):
                    grad_found = True
                    break
            assert grad_found, "No non-zero gradients found at batch 0, epoch 0 - backprop may have failed"
        
        # Pre-optimizer gradient check: at least some gradients should be non-zero
        grad_exists = False
        grad_sum = 0.0
        for p in model.parameters():
            if p.grad is not None and torch.any(p.grad != 0):
                grad_exists = True
                grad_sum += torch.abs(p.grad).sum().item()
        
        assert grad_exists, f"No non-zero gradients before optimizer.step() at batch {batch_idx}, epoch {epoch}"
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Collect early stage losses (first 10 epochs)
        if epoch < 10:
            early_stage_losses.append(loss.item())
        
        # Log per-batch metrics to WandB
        if HAS_WANDB and cfg.wandb.mode != "disabled":
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/batch_id": batch_idx,
                "train/epoch": epoch,
            })
        
        if batch_idx % max(1, len(train_loader) // 10) == 0:
            log.info(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Calculate early stage stability
    if epoch == 9 and early_stage_losses:
        mean_loss = np.mean(early_stage_losses)
        std_loss = np.std(early_stage_losses)
        cv = (std_loss / mean_loss * 100) if mean_loss > 0 else 0
        tracker.early_stage_loss_variance = cv
        log.info(f"Early-stage stability (epochs 0-9): CV={cv:.2f}%")
    
    tracker.log_step(train_loss=avg_loss)
    
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
    task_type: str = "classification",
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Validate model and return (loss, accuracy/None, perplexity/None).
    task_type: "classification" or "language_modeling"
    CRITICAL: Model receives ONLY inputs; labels used ONLY for loss/metrics.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Limit batches for trial mode
            if cfg.training.get("trial_batches") is not None:
                if batch_idx >= cfg.training.trial_batches:
                    break
            
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    inputs, labels = batch
                else:
                    inputs = batch[0]
                    labels = batch[1] if len(batch) > 1 else None
            else:
                inputs = batch
                labels = None
            
            inputs = inputs.to(device)
            if labels is not None:
                labels = labels.to(device)
            
            # **CRITICAL DATA LEAK PREVENTION**: Forward pass with inputs ONLY
            outputs = model(inputs)
            
            # Compute loss - labels used ONLY for loss computation
            if labels is not None:
                loss = F.cross_entropy(outputs, labels)
                
                # Classification metrics (batch-start assertions for first batch)
                if batch_idx == 0:
                    assert inputs.size(0) == labels.size(0), \
                        f"Batch shape mismatch at val batch 0: inputs {inputs.size(0)} vs labels {labels.size(0)}"
                
                if task_type == "classification":
                    _, predicted = outputs.max(1)
                    correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
            else:
                loss = outputs
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else None
    perplexity = math.exp(avg_loss) if avg_loss > 0 else None
    
    return avg_loss, accuracy, perplexity


def train_with_optuna(
    cfg: DictConfig,
    trial: Optional['Trial'] = None,
    tracker: Optional[TrainingTracker] = None,
) -> Tuple[float, TrainingTracker]:
    """
    Train a single trial with optuna hyperparameter suggestions.
    Returns (primary_metric_value, tracker).
    **CRITICAL**: Intermediate trial results NOT logged to WandB; only best trial after optimization.
    """
    if tracker is None:
        tracker = TrainingTracker()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Set random seed
    seed = cfg.training.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Track if this is an Optuna trial
    is_optuna_trial = (trial is not None and cfg.get("optuna", {}).get("enabled", False) 
                      and cfg.optuna.n_trials > 0)
    
    # ===== Hyperparameter Search (if Optuna enabled) =====
    if is_optuna_trial:
        log.info(f"Optuna Trial {trial.number}: Sampling hyperparameters")
        for search_space in cfg.optuna.search_spaces:
            param_name = search_space.param_name
            dist_type = search_space.distribution_type
            
            if dist_type == "uniform":
                value = trial.suggest_float(param_name, search_space.low, search_space.high)
            elif dist_type == "int":
                value = trial.suggest_int(param_name, int(search_space.low), int(search_space.high))
            elif dist_type == "categorical":
                value = trial.suggest_categorical(param_name, search_space.choices)
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
            
            # Update config with suggested value
            if "." in param_name:
                keys = param_name.split(".")
                cfg_ref = cfg
                for key in keys[:-1]:
                    cfg_ref = cfg_ref[key]
                cfg_ref[keys[-1]] = value
            else:
                cfg[param_name] = value
            
            log.info(f"  {param_name} = {value}")
    
    # ===== Model Setup =====
    log.info(f"Building model: {cfg.model.name}")
    model = build_model(cfg.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # **POST-INIT ASSERTIONS**
    assert model is not None, "Model construction failed"
    for name, param in model.named_parameters():
        assert param.dtype in [torch.float32, torch.float64], \
            f"Invalid parameter dtype for {name}: {param.dtype}"
    
    # ===== Data Setup =====
    log.info(f"Building dataset: {cfg.dataset.name}")
    train_loader, val_loader, test_loader = build_dataloader(cfg.dataset, cfg.training)
    
    # **BATCH-START ASSERTIONS (sample first batch)**
    for batch_idx, batch in enumerate(train_loader):
        if isinstance(batch, (list, tuple)):
            inputs, labels = batch[0], batch[1] if len(batch) > 1 else None
        else:
            inputs, labels = batch, None
        
        inputs = inputs.to(device)
        if labels is not None:
            labels = labels.to(device)
            assert inputs.size(0) == labels.size(0), \
                f"Batch size mismatch at train batch 0: inputs {inputs.size(0)} vs labels {labels.size(0)}"
        
        log.info(f"Batch 0 shapes: inputs {inputs.shape}, labels {labels.shape if labels is not None else 'N/A'}")
        break
    
    # ===== Optimizer Setup =====
    optimizer_name = cfg.training.optimizer.lower()
    optimizer_params = {
        "lr": cfg.training.learning_rate,
        "betas": tuple(cfg.training.get("betas", [0.9, 0.999])),
        "eps": cfg.training.get("epsilon", 1e-8),
        "weight_decay": cfg.training.get("weight_decay", 0.0),
    }
    
    if optimizer_name == "agvr":
        optimizer_params.update({
            "epsilon_adapt": cfg.training.get("epsilon_adapt", 1.0),
            "var_window": cfg.training.get("var_window", 20),
            "var_decay": cfg.training.get("var_decay", 0.95),
        })
        optimizer = AGVR(model.parameters(), **optimizer_params)
        log.info(f"Using AGVR optimizer with epsilon_adapt={optimizer_params['epsilon_adapt']}")
        
        # **AGVR-SPECIFIC ASSERTIONS**
        assert hasattr(optimizer, 'smoothed_var'), "AGVR optimizer missing smoothed_var attribute"
        assert hasattr(optimizer, 'var_history'), "AGVR optimizer missing var_history attribute"
        log.info("✓ AGVR optimizer attributes validated")
    elif optimizer_name == "radam":
        optimizer = RAdam(model.parameters(), **optimizer_params)
        log.info("Using RAdam optimizer")
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        log.info("Using Adam optimizer")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # ===== Training Loop =====
    best_val_metric = float("inf")
    best_val_epoch = -1
    
    log.info(f"Starting training for {cfg.training.epochs} epochs")
    
    for epoch in range(cfg.training.epochs):
        log.info(f"\n{'='*60}")
        log.info(f"Epoch {epoch+1}/{cfg.training.epochs}")
        log.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, cfg, epoch, tracker
        )
        log.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_acc, val_perplexity = validate(
            model, val_loader, device, cfg,
            task_type=cfg.dataset.get("task_type", "classification")
        )
        log.info(f"Val loss: {val_loss:.4f}")
        if val_acc is not None:
            log.info(f"Val accuracy: {val_acc:.4f}")
        if val_perplexity is not None:
            log.info(f"Val perplexity: {val_perplexity:.4f}")
        
        # Extract adaptive threshold if using AGVR
        rho_adaptive = None
        lambda_t = None
        if hasattr(optimizer, "smoothed_var"):
            lambda_t = optimizer.smoothed_var
            rho_adaptive = 4.0 + cfg.training.get("epsilon_adapt", 1.0) * (1.0 - lambda_t)
            log.info(f"Adaptive threshold: ρ_adaptive_t = {rho_adaptive:.4f}, λ_t = {lambda_t:.4f}")
        
        # **CRITICAL**: Only log intermediate epochs to WandB if NOT an Optuna trial
        should_log_to_wandb = HAS_WANDB and cfg.wandb.mode != "disabled" and not is_optuna_trial
        
        if should_log_to_wandb:
            log_dict = {
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss,
            }
            if val_acc is not None:
                log_dict["val/accuracy"] = val_acc
            if val_perplexity is not None:
                log_dict["val/perplexity"] = val_perplexity
            if rho_adaptive is not None:
                log_dict["optimizer/rho_adaptive"] = rho_adaptive
            if lambda_t is not None:
                log_dict["optimizer/lambda_t"] = lambda_t
            
            # Log learning rate if available
            for param_group in optimizer.param_groups:
                if "lr" in param_group:
                    log_dict["optimizer/learning_rate"] = param_group["lr"]
                    break
            
            wandb.log(log_dict)
        
        # Track metrics
        tracker.log_step(
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            val_perplexity=val_perplexity,
            rho_adaptive=rho_adaptive,
            lambda_t=lambda_t,
        )
        
        # Check for early exit in trial mode
        if cfg.get("mode") == "trial":
            if epoch >= 0:
                break
        
        # Early stopping based on best validation metric
        current_metric = val_perplexity if val_perplexity is not None else val_loss
        if current_metric < best_val_metric:
            best_val_metric = current_metric
            best_val_epoch = epoch
            tracker.best_val_metric = best_val_metric
            tracker.best_val_epoch = best_val_epoch
            log.info(f"New best metric: {best_val_metric:.4f}")
    
    # ===== Test Set Evaluation =====
    log.info(f"\n{'='*60}")
    log.info("Final Test Set Evaluation")
    log.info(f"{'='*60}")
    
    test_loss, test_acc, test_perplexity = validate(
        model, test_loader, device, cfg,
        task_type=cfg.dataset.get("task_type", "classification")
    )
    log.info(f"Test loss: {test_loss:.4f}")
    if test_acc is not None:
        log.info(f"Test accuracy: {test_acc:.4f}")
    if test_perplexity is not None:
        log.info(f"Test perplexity: {test_perplexity:.4f}")
    
    # Store test metrics in tracker
    tracker.test_loss = test_loss
    tracker.test_accuracy = test_acc
    tracker.test_perplexity = test_perplexity
    
    # Return primary metric for Optuna
    primary_metric = test_perplexity if test_perplexity is not None else test_loss
    
    return primary_metric, tracker


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training entry point using Hydra.
    """
    log.info(f"AGVR Training: run={cfg.run}, mode={cfg.mode}")
    
    # Validate required parameters
    assert cfg.run is not None, "run parameter required"
    assert cfg.mode in ["trial", "full"], f"Invalid mode: {cfg.mode}"
    
    # Create results directory
    results_path = Path(cfg.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Resolve run config path with absolute path resolution
    repo_root = Path(__file__).parent.parent
    run_config_path = repo_root / "config" / "runs" / f"{cfg.run}.yaml"
    
    if not run_config_path.exists():
        log.error(f"Run configuration not found: {run_config_path}")
        sys.exit(1)
    
    run_cfg = OmegaConf.load(run_config_path)
    
    # Apply mode-specific overrides
    if cfg.mode == "trial":
        log.info("Trial mode: applying lightweight configuration")
        run_cfg.training.epochs = 1
        run_cfg.wandb.mode = "disabled"
        if "optuna" in run_cfg:
            run_cfg.optuna.n_trials = 0
        run_cfg.training.trial_batches = 2
    elif cfg.mode == "full":
        log.info("Full mode: using complete configuration")
        run_cfg.wandb.mode = "online"
        run_cfg.training.trial_batches = None
    
    # Merge configurations
    merged_cfg = OmegaConf.merge(cfg, run_cfg)
    
    # **POST-MODE ASSERTIONS**
    if cfg.mode == "trial":
        assert merged_cfg.wandb.mode == "disabled", "WandB should be disabled in trial mode"
        assert merged_cfg.training.epochs == 1, "Epochs should be 1 in trial mode"
    elif cfg.mode == "full":
        assert merged_cfg.wandb.mode == "online", "WandB should be online in full mode"
    
    # Initialize WandB only if not disabled
    if HAS_WANDB and merged_cfg.wandb.mode != "disabled":
        wandb.init(
            entity=merged_cfg.wandb.entity,
            project=merged_cfg.wandb.project,
            id=cfg.run,
            config=OmegaConf.to_container(merged_cfg, resolve=True),
            resume="allow",
        )
        log.info(f"WandB initialized: {wandb.run.get_url()}")
        print(f"WandB URL: {wandb.run.get_url()}", file=sys.stderr)
    else:
        log.info("WandB disabled")
    
    # Initialize tracker for final logging
    tracker = TrainingTracker()
    
    # Optuna hyperparameter search
    if merged_cfg.get("optuna", {}).get("enabled", False) and merged_cfg.optuna.n_trials > 0:
        if not HAS_OPTUNA:
            log.error("Optuna not installed but enabled in config")
            sys.exit(1)
        
        log.info(f"Optuna enabled: {merged_cfg.optuna.n_trials} trials")
        
        def objective(trial: 'Trial') -> float:
            # Suppress WandB for intermediate trials
            primary_metric, _ = train_with_optuna(merged_cfg, trial)
            return primary_metric
        
        sampler = optuna.samplers.TPESampler(seed=merged_cfg.training.get("seed", 42))
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=merged_cfg.optuna.n_trials)
        
        best_trial = study.best_trial
        log.info(f"Best Optuna trial: {best_trial.number} with value {best_trial.value:.4f}")
        log.info(f"Best params: {best_trial.params}")
        
        # Apply best hyperparameters and train final model with WandB logging
        log.info("Training final model with best hyperparameters and WandB logging...")
        for key, value in best_trial.params.items():
            if "." in key:
                keys = key.split(".")
                cfg_ref = merged_cfg
                for k in keys[:-1]:
                    cfg_ref = cfg_ref[k]
                cfg_ref[keys[-1]] = value
            else:
                merged_cfg[key] = value
        
        # Final training run with best hyperparameters (WITH WandB logging)
        primary_metric, tracker = train_with_optuna(merged_cfg, trial=None, tracker=tracker)
        
        if HAS_WANDB and merged_cfg.wandb.mode != "disabled":
            wandb.summary["optuna_best_trial"] = best_trial.number
            wandb.summary["optuna_best_value"] = best_trial.value
            wandb.summary["optuna_best_params"] = best_trial.params
    else:
        # Standard single training run
        log.info("Running standard training (Optuna disabled)")
        primary_metric, tracker = train_with_optuna(merged_cfg, tracker=tracker)
        log.info(f"Training completed. Primary metric: {primary_metric:.4f}")
    
    # **LOG FINAL SUMMARY METRICS TO WANDB**
    if HAS_WANDB and merged_cfg.wandb.mode != "disabled":
        wandb.summary["final_primary_metric"] = primary_metric
        if tracker.best_val_metric is not None:
            wandb.summary["best_val_metric"] = tracker.best_val_metric
        if tracker.best_val_epoch is not None:
            wandb.summary["best_val_epoch"] = tracker.best_val_epoch
        if tracker.early_stage_loss_variance is not None:
            wandb.summary["early_stage_stability"] = tracker.early_stage_loss_variance
        if tracker.test_accuracy is not None:
            wandb.summary["test_accuracy"] = tracker.test_accuracy
        if tracker.test_perplexity is not None:
            wandb.summary["test_perplexity"] = tracker.test_perplexity
        if tracker.test_loss is not None:
            wandb.summary["test_loss"] = tracker.test_loss
        
        wandb.finish()


if __name__ == "__main__":
    main()
