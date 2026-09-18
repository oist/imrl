"""
Checkpoint utilities for saving and resuming training.

This module provides functions for:
- Saving training checkpoints (train_state, ex_state, update_step, rng, config)
- Loading/resuming from checkpoints
- Periodic checkpoint saving during training
- Finding previous wandb runs for resumption
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

logger = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[int]:
    """
    Find the latest checkpoint step in a directory.
    
    Returns:
        The latest checkpoint step, or None if no checkpoints found.
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for numbered directories (Orbax checkpoint format)
    steps = []
    for entry in os.listdir(checkpoint_dir):
        entry_path = os.path.join(checkpoint_dir, entry)
        if os.path.isdir(entry_path):
            try:
                step = int(entry)
                steps.append(step)
            except ValueError:
                continue
    
    if not steps:
        return None
    
    return max(steps)


def find_latest_wandb_run_with_checkpoints(wandb_dir: str = "wandb", project: str = None) -> Optional[Tuple[str, str, str]]:
    """
    Find the most recent wandb run directory that contains checkpoints.
    
    Args:
        wandb_dir: Path to the wandb directory (default: "wandb")
        project: Optional project name to filter by
        
    Returns:
        Tuple of (run_id, checkpoint_dir, run_dir) or None if not found
    """
    if not os.path.exists(wandb_dir):
        return None
    
    # Find all run directories, sorted by modification time (most recent first)
    run_dirs = []
    for entry in os.listdir(wandb_dir):
        if entry.startswith("run-"):
            run_path = os.path.join(wandb_dir, entry)
            if os.path.isdir(run_path):
                mtime = os.path.getmtime(run_path)
                run_dirs.append((mtime, entry, run_path))
    
    # Sort by modification time, most recent first
    run_dirs.sort(reverse=True)
    
    for mtime, entry, run_path in run_dirs:
        # Extract run ID from directory name (format: run-YYYYMMDD_HHMMSS-<run_id>)
        parts = entry.split("-")
        if len(parts) >= 3:
            run_id = parts[-1]
            
            # Check for checkpoints in this run
            files_dir = os.path.join(run_path, "files")
            checkpoint_dir = os.path.join(files_dir, "checkpoints")
            
            if os.path.exists(checkpoint_dir):
                # Verify there are actual checkpoint files
                checkpoint_steps = find_latest_checkpoint(checkpoint_dir)
                if checkpoint_steps is not None:
                    logger.info(f"Found checkpoints in wandb run {run_id} at {checkpoint_dir}")
                    return (run_id, checkpoint_dir, run_path)
    
    return None


def get_checkpoint_dir(config: Dict[str, Any]) -> str:
    """
    Determine the checkpoint directory based on config.
    
    Priority:
    1. CHECKPOINT_DIR if explicitly set
    2. wandb run dir if USE_WANDB is True
    3. output_dir/checkpoints if output_dir is set
    4. Default to logdir/checkpoints
    """
    if config.get("CHECKPOINT_DIR"):
        path = config["CHECKPOINT_DIR"]
        # Ensure absolute path for Orbax and other tooling
        abs_path = os.path.abspath(path)
        if abs_path != path:
            logger.info(f"Converting CHECKPOINT_DIR to absolute path: {path} -> {abs_path}")
        # Ensure directory exists
        os.makedirs(abs_path, exist_ok=True)
        return abs_path
    
    if config.get("USE_WANDB"):
        try:
            import wandb
            if wandb.run is not None and hasattr(wandb.run, 'dir'):
                return os.path.join(wandb.run.dir, "checkpoints")
        except Exception:
            pass
    
    record_dir = config.get("RECORD_DIR")
    if record_dir:
        return os.path.join(record_dir, "checkpoints")
    
    # Default fallback - ensure absolute path so Orbax is happy
    default_dir = os.path.join("logdir", "checkpoints")
    abs_default = os.path.abspath(default_dir)
    if abs_default != default_dir:
        logger.info(f"Using absolute default checkpoint dir: {abs_default}")
    os.makedirs(abs_default, exist_ok=True)
    return abs_default


def create_checkpoint_manager(checkpoint_dir: str, max_to_keep: int = 3) -> CheckpointManager:
    """Create an Orbax checkpoint manager."""
    # Normalize to absolute path to satisfy Orbax requirements
    abs_dir = os.path.abspath(checkpoint_dir)
    if abs_dir != checkpoint_dir:
        logger.info(f"Normalizing checkpoint_dir to absolute path: {checkpoint_dir} -> {abs_dir}")
    os.makedirs(abs_dir, exist_ok=True)
    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    checkpoint_manager = CheckpointManager(abs_dir, orbax_checkpointer, options)
    return checkpoint_manager


def get_serializable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only JSON-serializable parts of config.
    Excludes callable functions, JAX arrays, and other non-serializable objects.
    """
    serializable_config = {}
    for key, value in config.items():
        if key in ["VAE_PARAMS", "VAE_APPLY_FN", "VAE_APPLY_FN_JIT", "INTRINSIC_REWARD_NORMALIZER",
                   "SCIRE_MODEL", "SCIRE_PARAMS", "SCIRE_ACTION_SEQUENCES"]:
            # Skip non-serializable intrinsic reward components
            continue
        if callable(value):
            continue
        try:
            # Test if JSON serializable
            json.dumps(value)
            serializable_config[key] = value
        except (TypeError, ValueError):
            # Skip non-serializable values
            if isinstance(value, jnp.ndarray):
                # Convert JAX arrays to lists
                try:
                    serializable_config[key] = value.tolist()
                except Exception:
                    pass
            continue
    return serializable_config


def save_checkpoint(
    checkpoint_dir: str,
    update_step: int,
    train_state: TrainState,
    ex_state: Dict[str, Any],
    rng: jax.random.PRNGKey,
    config: Dict[str, Any],
    best_metric: float = -float('inf'),
    best_step: int = 0,
    env_state: Any = None,
    obs: Any = None,
) -> str:
    """
    Save a complete training checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        update_step: Current update step
        train_state: Flax TrainState with network parameters and optimizer state
        ex_state: Exploration/intrinsic reward state (VAE params, ICM params, normalizer, etc.)
        rng: Current JAX random key
        config: Training configuration
        best_metric: Best metric achieved so far
        best_step: Update step where best metric was achieved
        env_state: Environment state (optional, for exact resumption)
        obs: Current observations (optional, for exact resumption)
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint pytree - only include JAX-compatible data
    checkpoint_data = {
        "train_state": train_state,
        "update_step": jnp.int32(update_step),
        "rng": rng,
        "best_metric": jnp.float32(best_metric) if best_metric != -float('inf') else jnp.float32(-1e10),
        "best_step": jnp.int32(best_step),
    }
    
    # Add ex_state components that are serializable JAX pytrees
    serializable_ex_state = {}
    for key, value in ex_state.items():
        if value is None:
            continue
        # TrainState objects (ICM networks) are directly serializable
        if isinstance(value, TrainState):
            serializable_ex_state[key] = value
        # Dict of arrays (intrinsic components)
        elif isinstance(value, dict):
            try:
                # Check if it's a JAX-compatible dict
                jax.tree.map(lambda x: x, value)
                serializable_ex_state[key] = value
            except Exception:
                pass
        # JAX arrays or other pytrees
        elif hasattr(value, 'dtype') or isinstance(value, (tuple, list)):
            try:
                jax.tree.map(lambda x: x, value)
                serializable_ex_state[key] = value
            except Exception:
                pass
    
    checkpoint_data["ex_state"] = serializable_ex_state
    
    # Optionally save env_state and obs for exact resumption
    if env_state is not None:
        try:
            jax.tree.map(lambda x: x, env_state)
            checkpoint_data["env_state"] = env_state
        except Exception:
            logger.warning("Could not serialize env_state, skipping")
    
    if obs is not None:
        checkpoint_data["obs"] = obs
    
    # Save the checkpoint using Orbax
    checkpoint_manager = create_checkpoint_manager(checkpoint_dir, max_to_keep=3)
    save_args = orbax_utils.save_args_from_target(checkpoint_data)
    
    checkpoint_manager.save(
        update_step,
        checkpoint_data,
        save_kwargs={"save_args": save_args},
    )
    
    # Save config as JSON alongside checkpoint
    config_path = os.path.join(checkpoint_dir, f"config_{update_step}.json")
    try:
        serializable_config = get_serializable_config(config)
        serializable_config["_checkpoint_update_step"] = update_step
        serializable_config["_checkpoint_best_metric"] = float(best_metric) if best_metric != -float('inf') else None
        serializable_config["_checkpoint_best_step"] = int(best_step)
        
        # Save wandb run ID for resume capability
        try:
            import wandb
            if wandb.run is not None:
                serializable_config["_wandb_run_id"] = wandb.run.id
                serializable_config["_wandb_run_name"] = wandb.run.name
                serializable_config["_wandb_project"] = wandb.run.project
                serializable_config["_wandb_entity"] = wandb.run.entity
        except Exception:
            pass
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save config JSON: {e}")

    # Ensure there's a metrics.jsonl entry for this checkpoint step so resumes
    # have aligned x/y pairs (best-effort). This prevents situations where a
    # checkpoint exists but no metrics were written at the same step.
    try:
        record_dir = serializable_config.get('RECORD_DIR') or config.get('RECORD_DIR')
        if record_dir:
            stats_path = os.path.join(record_dir, 'metrics.jsonl')
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            # Compute total_timesteps if we have NUM_STEPS/NUM_ENVS
            num_steps = int(serializable_config.get('NUM_STEPS', config.get('NUM_STEPS', 1)) or 1)
            num_envs = int(serializable_config.get('NUM_ENVS', config.get('NUM_ENVS', 1)) or 1)
            total_timesteps = int(update_step) * num_steps * num_envs
            entry = {'update_step': int(update_step), 'total_timesteps': total_timesteps}
            if serializable_config.get('_checkpoint_best_metric') is not None:
                entry['checkpoint_best_metric'] = float(serializable_config.get('_checkpoint_best_metric'))
                entry['best_metric'] = float(serializable_config.get('_checkpoint_best_metric'))

            # Append only if the file doesn't already contain this step
            exists = False
            if os.path.exists(stats_path):
                try:
                    with open(stats_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line)
                                if int(obj.get('update_step', -1)) == int(update_step):
                                    exists = True
                                    break
                            except Exception:
                                continue
                except Exception:
                    exists = False

            if not exists and len(entry) > 2:
                with open(stats_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry) + '\n')
                logger.info(f"Wrote synthetic metrics.jsonl entry for checkpoint {update_step} to {stats_path}")
    except Exception as e:
        logger.debug(f"Failed to write checkpoint metrics entry: {e}")
    
    logger.info(f"Checkpoint saved at update_step {update_step} to {checkpoint_dir}")
    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: str,
    step: Optional[int] = None,
    train_state_template: Optional[TrainState] = None,
    ex_state_template: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Specific step to load (default: latest)
        train_state_template: Template TrainState for structure
        ex_state_template: Template ex_state dict for structure
        
    Returns:
        Tuple of (checkpoint_data, config) where checkpoint_data contains:
        - train_state: Restored TrainState
        - ex_state: Restored exploration state
        - update_step: The update step of the checkpoint
        - rng: Restored random key
        - best_metric: Best metric value
        - best_step: Step where best metric was achieved
        - env_state: Restored environment state (if available)
        - obs: Restored observations (if available)
    """
    if step is None:
        step = find_latest_checkpoint(checkpoint_dir)
    
    if step is None:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    logger.info(f"Loading checkpoint from step {step} in {checkpoint_dir}")
    
    # Create checkpoint manager and restore
    checkpoint_manager = create_checkpoint_manager(checkpoint_dir, max_to_keep=3)
    
    # Build template for restoration
    template = {}
    if train_state_template is not None:
        template["train_state"] = train_state_template
    if ex_state_template is not None:
        template["ex_state"] = ex_state_template
    template["update_step"] = jnp.int32(0)
    template["rng"] = jax.random.PRNGKey(0)
    template["best_metric"] = jnp.float32(0.0)
    template["best_step"] = jnp.int32(0)
    
    # Try to restore with template
    try:
        checkpoint_data = checkpoint_manager.restore(step, items=template)
    except Exception as e:
        logger.warning(f"Could not restore with template: {e}, trying without template")
        checkpoint_data = checkpoint_manager.restore(step)
    
    # Load config from JSON
    config = {}
    config_path = os.path.join(checkpoint_dir, f"config_{step}.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config JSON: {e}")
    
    return checkpoint_data, config


def should_save_checkpoint(update_step: int, checkpoint_freq: int) -> bool:
    """Check if we should save a checkpoint at this update step."""
    if checkpoint_freq <= 0:
        return False
    return update_step > 0 and update_step % checkpoint_freq == 0


class CheckpointCallback:
    """
    Callback class for periodic checkpoint saving during training.
    
    This is designed to be called from within the training callback in ppo.py
    to save checkpoints at regular intervals.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_freq: int,
        config: Dict[str, Any],
    ):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.config = config
        self.last_saved_step = -1
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpoint callback initialized. Saving every {checkpoint_freq} steps to {checkpoint_dir}")
    
    def maybe_save(
        self,
        update_step: int,
        train_state: TrainState,
        ex_state: Dict[str, Any],
        rng: jax.random.PRNGKey,
        best_metric: float = -float('inf'),
        best_step: int = 0,
        env_state: Any = None,
        obs: Any = None,
    ) -> bool:
        """
        Save checkpoint if it's time to do so.
        
        Returns:
            True if checkpoint was saved, False otherwise.
        """
        if not should_save_checkpoint(update_step, self.checkpoint_freq):
            return False
        
        if update_step == self.last_saved_step:
            return False
        
        save_checkpoint(
            checkpoint_dir=self.checkpoint_dir,
            update_step=update_step,
            train_state=train_state,
            ex_state=ex_state,
            rng=rng,
            config=self.config,
            best_metric=best_metric,
            best_step=best_step,
            env_state=env_state,
            obs=obs,
        )
        
        self.last_saved_step = update_step
        return True
