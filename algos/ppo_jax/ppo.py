# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


import datetime

# --- DEVICE SELECTION: must be before any JAX/Flax imports ---
import argparse
import sys
import os

# Ensure legacy `import gym` resolves to Gymnasium when available to avoid
# deprecation warnings coming from third-party packages that still import
# the unmaintained `gym` package. This is safe because Gymnasium aims to be
# a drop-in replacement for the vast majority of use cases.
try:
    import gymnasium as _gymnasium
    if sys.modules.get('gym') is not _gymnasium:
        sys.modules['gym'] = _gymnasium
except Exception:
    # If gymnasium isn't present, do nothing. Downstream imports will behave
    # as they did before and will emit their own warnings if necessary.
    pass

def generate_recording_dir(config):
    """Generate recording directory path based on config parameters.
    
    Returns an absolute path to ensure files are saved in the current working
    directory (which should be $SLURM_TMPDIR in SLURM jobs).
    """
    # Get the weights for intrinsic rewards
    novelty_weight = config.get("NOVELTY_WEIGHT", 0)
    surprise_weight = config.get("SURPRISE_WEIGHT", 0)
    empowerment_weight = config.get("EMPOWERMENT_WEIGHT", 0)
    seed = config.get("SEED", 0)
    
    # Build directory name based on active intrinsic rewards
    dir_parts = []
    
    if novelty_weight != 0:
        dir_parts.append(f"novelty_{novelty_weight:g}")
    if surprise_weight != 0:
        dir_parts.append(f"surprise_{surprise_weight:g}")
    if empowerment_weight != 0:
        dir_parts.append(f"empowerment_{empowerment_weight:g}")
    
    # If no intrinsic rewards are active, it's vanilla PPO
    if not dir_parts:
        dir_name = "ppo"
    else:
        dir_name = "_".join(dir_parts)
    
    # Return absolute path based on current working directory
    # This ensures files are saved to $SLURM_TMPDIR when running in SLURM
    return os.path.abspath(f"logdir/{dir_name}/{seed}")


def restore_last_metrics_from_jsonl(metrics_path):
    """Read the last logged metrics from metrics.jsonl for smooth resume.
    
    This ensures WandB plots continue from the exact last values instead of
    resetting, which would cause discontinuities/jumps in the curves.
    
    Args:
        metrics_path: Path to metrics.jsonl file
        
    Returns:
        dict: Last metrics data with total_timesteps, or None if file doesn't exist
    """
    import json
    
    if not os.path.exists(metrics_path):
        return None
    
    try:
        # Read last line of JSONL file
        last_line = None
        with open(metrics_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    last_line = line
        
        if last_line:
            last_metrics = json.loads(last_line)
            logger.info(f"📊 Restored last metrics from checkpoint: update_step={last_metrics.get('update_step')}, total_timesteps={last_metrics.get('total_timesteps')}")
            return last_metrics
        else:
            logger.warning("metrics.jsonl exists but is empty")
            return None
            
    except Exception as e:
        logger.warning(f"Failed to restore last metrics from {metrics_path}: {e}")
        return None


def parse_device_arg():
    for i, arg in enumerate(sys.argv):
        if arg == '--device' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith('--device='):
            return arg.split('=', 1)[1]
    return None

device = parse_device_arg()
if device is not None:
    if device == 'cuda':
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    elif device == 'cpu':
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import time
import logging
import jax
import jax.numpy as jnp
import numpy as np
import optax

import wandb
from typing import NamedTuple

# Module logger
logger = logging.getLogger(__name__)

from tracking.episode_monitor import create_episode_monitor
from tracking.episode_video_integration import create_episode_recorder_if_enabled
from tracking.checkpoint_utils import (
    get_checkpoint_dir,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    find_latest_wandb_run_with_checkpoints,
    should_save_checkpoint,
    CheckpointCallback,
)

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from logging_utils.batch_logging import batch_log, create_log_dict
from models.actor_critic import (
    ActorCritic,
    ActorCriticConv,
)
from models.icm import ICMEncoder, ICMForward, ICMInverse
from models.rnd import RNDNetwork
from intrinsic_rewards import vime_jax
from intrinsic_rewards.count_based_rewards import compute_batch_bonus_jax
import scire
from envs.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)


def _coerce_normalizer(normalizer):
    """Ensure `normalizer` is a `scire.NormalizerState`.

    Orbax checkpoint restores without an explicit target structure can come
    back as plain (possibly nested) dicts instead of NamedTuples, so this
    reconstructs the expected nested NamedTuple shape, falling back to a
    fresh normalizer if the structure doesn't match.
    """
    if isinstance(normalizer, scire.NormalizerState):
        return normalizer

    def to_stats(value):
        if isinstance(value, scire.RunningStats):
            return value
        if isinstance(value, dict) and set(scire.RunningStats._fields) <= set(value.keys()):
            return scire.RunningStats(**{k: value[k] for k in scire.RunningStats._fields})
        return scire.RunningStats.init()

    if isinstance(normalizer, dict) and set(scire.NormalizerState._fields) <= set(normalizer.keys()):
        return scire.NormalizerState(**{k: to_stats(normalizer[k]) for k in scire.NormalizerState._fields})

    return scire.NormalizerState.init()


def has_intrinsic_rewards(config):
    """Check if intrinsic rewards are enabled in the config."""
    return (
        config.get("USE_JAX_INTRINSIC_REWARDS", False) or
        config.get("TRAIN_ICM", False) or
        config.get("TRAIN_RND", False) or
        config.get("TRAIN_VIME", False) or
        config.get("USE_COUNT_BASED_NOVELTY", False)
    )


def should_use_dual_value_heads(config):
    """Check if dual value heads should be used (only when intrinsic rewards are enabled)."""
    return config.get("USE_DUAL_VALUE_HEADS", False) and has_intrinsic_rewards(config)


def get_algorithm_name(config):
    """Generate algorithm name based on intrinsic reward weights."""
    # Preference order: RND, ICM, VIME, count-based, then SCIRE weights
    if config.get("TRAIN_RND", False):
        return "PPO_RND"

    if config.get("TRAIN_ICM", False):
        return "ICM"

    if config.get("TRAIN_VIME", False):
        return "VIME"

    if config.get("USE_COUNT_BASED_NOVELTY", False):
        return "CountBased"

    novelty_weight = config.get("NOVELTY_WEIGHT", 0.0)
    surprise_weight = config.get("SURPRISE_WEIGHT", 0.0)
    empowerment_weight = config.get("EMPOWERMENT_WEIGHT", 0.0)
    
    # Count non-zero weights
    active_weights = []
    if novelty_weight > 0:
        active_weights.append("Novelty")
    if surprise_weight > 0:
        active_weights.append("Surprise") 
    if empowerment_weight > 0:
        active_weights.append("Empowerment")
    
    # Determine algorithm name
    if len(active_weights) == 0:
        return "PPO"
    elif len(active_weights) == 1:
        return active_weights[0]
    else:
        # Multiple intrinsic rewards - use combined name
        if len(active_weights) == 2:
            return f"{active_weights[0]}+{active_weights[1]}"
        elif len(active_weights) == 3:
            return "NSE-Combined"  # Novelty+Surprise+Empowerment
        else:
            return "Combined"


def compute_run_name(config, algorithm_name=None):
    """Return a run name string. If config contains RUN_NAME, return that, otherwise
    compute based on ENV_NAME, algorithm_name, and TOTAL_TIMESTEPS.
    """
    if config.get("RUN_NAME"):
        return config.get("RUN_NAME")
    algorithm_name = algorithm_name or get_algorithm_name(config)
    return config["ENV_NAME"] + "-" + algorithm_name + "-" + str(int(config["TOTAL_TIMESTEPS"] // 1e6)) + "M"


def make_craftax_env_from_name(env_name, auto_reset=True, base_seed=None):
    """Create environment from name - now routes through common factory"""
    from envs.env_factory import make_env_from_name
    result = make_env_from_name(env_name, auto_reset, base_seed)
    if isinstance(result, tuple):
        env, env_params = result
        return env  # Return just the environment for backward compatibility
    else:
        return result


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray  # Combined value (for backward compatibility)
    value_ext: jnp.ndarray  # Extrinsic value head
    value_int: jnp.ndarray  # Intrinsic value head
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config, best_policy_tracker_container=None):
    # Ensure numeric config entries are integers to avoid float division and orbax naming issues
    config["TOTAL_TIMESTEPS"] = int(float(config.get("TOTAL_TIMESTEPS", 0)))
    config["NUM_STEPS"] = int(config.get("NUM_STEPS", config.get("NUM_STEPS", 0)))
    config["NUM_ENVS"] = int(config.get("NUM_ENVS", config.get("NUM_ENVS", 1)))
    config["NUM_MINIBATCHES"] = int(config.get("NUM_MINIBATCHES", config.get("NUM_MINIBATCHES", 1)))

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Extract the actual dict from the container (if provided)
    # The container is a list with one element: [dict]
    # This allows the dict to be mutated by reference through JAX JIT
    if best_policy_tracker_container is not None:
        best_policy_tracker = best_policy_tracker_container[0]
    else:
        # If no tracker provided, create a local one (for vmap case)
        best_policy_tracker = {
            'best_return': -float('inf'),
            'best_metric': -float('inf'),
            'best_step': 0,
            'best_train_state': None,
            'save_best': config.get("SAVE_BEST_POLICY", False)
        }

    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"], config.get("SEED")
    )
    env_params = env.default_params

    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    # Calculate observation dimension for intrinsic reward modules
    obs_shape = env.observation_space(env_params).shape
    config["OBS_DIM"] = int(jnp.prod(jnp.array(obs_shape)))
    config["ACTION_DIM"] = int(env.action_space(env_params).n)
    
    # Note: JAX intrinsic rewards (SCIRE_MODEL, SCIRE_PARAMS, VAE_PARAMS for EMI) are
    # set up by train_agent.py when called via imrl.py. They don't need to be initialized here.

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        # Only use dual value heads if intrinsic rewards are enabled
        use_dual_value = should_use_dual_value_heads(config)
        
        if not has_intrinsic_rewards(config) and config.get("USE_DUAL_VALUE_HEADS", False):
            logger.info("Dual value heads requested but no intrinsic rewards enabled - using single value head")
        
        if "Symbolic" in config["ENV_NAME"] or "MiniGrid" in config["ENV_NAME"]:
            if use_dual_value:
                from models.actor_critic import ActorCriticDualValue
                network = ActorCriticDualValue(env.action_space(env_params).n, config["LAYER_SIZE"])
                logger.info("Using ActorCriticDualValue (separate value heads for intrinsic/extrinsic)")
            else:
                network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
        else:
            if use_dual_value:
                from models.actor_critic import ActorCriticConvDualValue
                network = ActorCriticConvDualValue(
                    env.action_space(env_params).n, config["LAYER_SIZE"]
                )
                logger.info("Using ActorCriticConvDualValue (separate value heads for intrinsic/extrinsic)")
            else:
                network = ActorCriticConv(
                    env.action_space(env_params).n, config["LAYER_SIZE"]
                )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Initialize episode monitor for MiniGrid environments
        episode_monitor = create_episode_monitor(config)

        # Exploration state
        ex_state = {
            "icm_encoder": None,
            "icm_forward": None,
            "icm_inverse": None,
            "rnd_target": None,
            "rnd_predictor": None,
            "vime_params": None,
            "vime_kl_buffer": None,
            "vime_buffer_idx": None,
            "intrinsic_reward_normalizer": config.get("INTRINSIC_REWARD_NORMALIZER", None),
            # For SCIRE: Store only the 'params' pytree, not the whole dict with model
            "scire_params": config.get("SCIRE_PARAMS", {}).get("params", None) if config.get("SCIRE_PARAMS") else None,
            # For EMI: Store the full emi_params dict (includes params, opt_state, embedding_pool, pool_idx)
            "emi_params": config.get("VAE_PARAMS") if config.get("USE_EMI", False) else None,
            # For count-based exploration: Hash table for visit counts
            "count_table": jnp.zeros(config.get("COUNT_TABLE_SIZE", 100000), dtype=jnp.float32) if config.get("USE_COUNT_BASED_NOVELTY", False) else None,
            # Accumulate weighted intrinsic components per environment for episode-level logging
            "intrinsic_components_accum": {
                'novelty_weighted': jnp.zeros((config['NUM_ENVS'],), dtype=jnp.float32),
                'surprise_weighted': jnp.zeros((config['NUM_ENVS'],), dtype=jnp.float32),
                'empowerment_weighted': jnp.zeros((config['NUM_ENVS'],), dtype=jnp.float32),
            },
            # Warmup scale for intrinsic rewards (starts at 0, scales up to 1)
            "intrinsic_warmup_scale": jnp.array(0.0, dtype=jnp.float32),
        }
        ex_state["intrinsic_reward_normalizer"] = _coerce_normalizer(ex_state["intrinsic_reward_normalizer"])

        if config["TRAIN_ICM"]:
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_shape = obs_shape[0]

            # Encoder
            icm_encoder_network = ICMEncoder(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_encoder_network_params = icm_encoder_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_encoder"] = TrainState.create(
                apply_fn=icm_encoder_network.apply,
                params=icm_encoder_network_params,
                tx=tx,
            )

            # Forward
            icm_forward_network = ICMForward(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
                num_actions=env.num_actions,
            )
            rng, _rng = jax.random.split(rng)
            icm_forward_network_params = icm_forward_network.init(
                _rng, jnp.zeros((1, config["ICM_LATENT_SIZE"])), jnp.zeros((1,))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_forward"] = TrainState.create(
                apply_fn=icm_forward_network.apply,
                params=icm_forward_network_params,
                tx=tx,
            )

            # Inverse
            icm_inverse_network = ICMInverse(
                num_layers=3,
                output_dim=env.num_actions,
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_inverse_network_params = icm_inverse_network.init(
                _rng,
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_inverse"] = TrainState.create(
                apply_fn=icm_inverse_network.apply,
                params=icm_inverse_network_params,
                tx=tx,
            )

        if config.get("TRAIN_RND", False):
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_shape = obs_shape[0]

            # Fixed random target network - never trained
            rnd_target_network = RNDNetwork(
                num_layers=3,
                output_dim=config["RND_OUTPUT_SIZE"],
                layer_size=config["RND_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            rnd_target_params = rnd_target_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )
            ex_state["rnd_target"] = TrainState.create(
                apply_fn=rnd_target_network.apply,
                params=rnd_target_params,
                tx=optax.set_to_zero(),
            )

            # Trainable predictor network - learns to mimic the fixed target
            rnd_predictor_network = RNDNetwork(
                num_layers=3,
                output_dim=config["RND_OUTPUT_SIZE"],
                layer_size=config["RND_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            rnd_predictor_params = rnd_predictor_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["RND_LR"], eps=1e-5),
            )
            ex_state["rnd_predictor"] = TrainState.create(
                apply_fn=rnd_predictor_network.apply,
                params=rnd_predictor_params,
                tx=tx,
            )

        if config.get("TRAIN_VIME", False):
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_dim = obs_shape[0]
            action_dim = config["ACTION_DIM"]

            rng, _rng = jax.random.split(rng)
            vime_layer_sizes = [
                obs_dim + action_dim,
                config["VIME_HIDDEN_DIM"],
                config["VIME_HIDDEN_DIM"],
                obs_dim,
            ]
            vime_init_params = vime_jax.init_bnn_params(_rng, vime_layer_sizes)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["VIME_LR"], eps=1e-5),
            )
            ex_state["vime_params"] = TrainState.create(
                apply_fn=lambda *a, **k: None,
                params=vime_init_params,
                tx=tx,
            )
            ex_state["vime_kl_buffer"] = jnp.ones(
                (config["VIME_KL_BUFFER_SIZE"],), dtype=jnp.float32
            )
            ex_state["vime_buffer_idx"] = jnp.array(0, dtype=jnp.int32)

        # Handle resume from checkpoint - restore train_state and ex_state
        resume_data = config.get("_RESUME_DATA")
        resume_update_step = 0
        resume_best_metric = jnp.float32(-jnp.inf)
        resume_best_step = jnp.int32(0)
        
        if resume_data is not None:
            logger.info("🔄 Restoring training state from checkpoint...")
            logger.info("  📦 Checkpoint contains keys: %s", list(resume_data.keys()))
            
            # Restore train_state (network params and optimizer state)
            if "train_state" in resume_data:
                try:
                    train_state = resume_data["train_state"]
                    logger.info("  ✅ Restored train_state (network parameters and optimizer state)")
                    # Verify optimizer state is present
                    if hasattr(train_state, 'opt_state') and train_state.opt_state is not None:
                        num_opt_params = len(jax.tree.leaves(train_state.opt_state))
                        logger.info("  ✅ Optimizer state verified: %d parameter leaves", num_opt_params)
                        # Check optimizer step count if available
                        if hasattr(train_state, 'step'):
                            logger.info("  ✅ Optimizer step count: %d", int(train_state.step))
                    else:
                        logger.warning("  ⚠️  Optimizer state may be missing or incomplete!")
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to restore train_state: {e}")
            
            # Restore ex_state (intrinsic reward modules, ICM, normalizer, etc.)
            if "ex_state" in resume_data:
                try:
                    restored_ex_state = resume_data["ex_state"]
                    # Merge restored ex_state with initialized ex_state
                    # This handles cases where new keys were added since the checkpoint
                    for key, value in restored_ex_state.items():
                        if value is not None:
                            ex_state[key] = value
                    # Orbax may restore the normalizer as a plain (nested) dict
                    # without an explicit target structure; re-coerce it.
                    ex_state["intrinsic_reward_normalizer"] = _coerce_normalizer(
                        ex_state["intrinsic_reward_normalizer"]
                    )
                    logger.info("  ✅ Restored ex_state (intrinsic reward state)")
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to restore ex_state: {e}")
            
            # Restore update step
            if "update_step" in resume_data:
                try:
                    resume_update_step = int(resume_data["update_step"])
                    logger.info(f"  ✅ Resuming from update_step {resume_update_step}")
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to restore update_step: {e}")
            
            # Restore best policy tracking
            if "best_metric" in resume_data:
                try:
                    resume_best_metric = jnp.float32(resume_data["best_metric"])
                except Exception:
                    pass
            
            if "best_step" in resume_data:
                try:
                    resume_best_step = jnp.int32(resume_data["best_step"])
                except Exception:
                    pass
            
            # Restore RNG if available
            if "rng" in resume_data:
                try:
                    rng = resume_data["rng"]
                    logger.info("  ✅ Restored random key")
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to restore rng: {e}")

        # INIT ENV
        # When resuming, restore env_state and obs from checkpoint to continue from exact state
        # Otherwise, reset to fresh environment state
        if resume_data is not None and "env_state" in resume_data and "obs" in resume_data:
            try:
                env_state = resume_data["env_state"]
                obsv = resume_data["obs"]
                logger.info("  ✅ Restored environment state and observations from checkpoint")
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to restore env_state/obs, resetting environment: {e}")
                rng, _rng = jax.random.split(rng)
                obsv, env_state = env.reset(_rng, env_params)
        else:
            rng, _rng = jax.random.split(rng)
            obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # Unpack runner_state including best policy tracking
            (train_state, env_state, last_obs, ex_state, rng, update_step,
             best_metric, best_step, best_train_state) = runner_state
            
            # COLLECT TRAJECTORIES
            def _env_step(runner_state_inner, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                ) = runner_state_inner

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                # Fix observation shape if it's a tuple (happens in some configurations)
                if isinstance(last_obs, tuple):
                    # Take the second element which should be the actual observation
                    actual_obs = last_obs[1] if len(last_obs) > 1 else last_obs[0]
                else:
                    actual_obs = last_obs
                
                # Handle dual value heads (only if intrinsic rewards are enabled)
                if should_use_dual_value_heads(config):
                    pi, value_ext, value_int = network.apply(train_state.params, actual_obs)
                    # Combined value for action selection (not used for training)
                    value = value_ext + value_int
                else:
                    pi, value = network.apply(train_state.params, actual_obs)
                    # Single value head - split conceptually for compatibility
                    value_ext = value
                    value_int = jnp.zeros_like(value)

                
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                reward_i = jnp.zeros(config["NUM_ENVS"])

                def _warmup_scale():
                    warmup_ratio = config.get("INTRINSIC_WARMUP_RATIO", 0.1)  # Default: 10% of training
                    total_updates = config["NUM_UPDATES"]
                    # Minimum 5 updates of warmup regardless of ratio (to allow model to train)
                    min_warmup_updates = 5
                    warmup_updates = max(min_warmup_updates, int(total_updates * warmup_ratio))
                    return jnp.minimum(1.0, (update_step + 1.0) / warmup_updates)

                # Add JAX-native intrinsic rewards if enabled
                if config.get("USE_JAX_INTRINSIC_REWARDS", False):
                    if config.get("USE_EMI", False):
                        # EMI is a fully independent intrinsic reward system (own coefficients,
                        # own reward/component names); it doesn't use the SCIRE normalizer.
                        obs_flat = jnp.reshape(last_obs, (config["NUM_ENVS"], config["OBS_DIM"]))
                        next_obs_flat = jnp.reshape(obsv, (config["NUM_ENVS"], config["OBS_DIM"]))
                        if config["ACTION_DIM"] > 1:  # Discrete action space
                            action_flat = jax.nn.one_hot(action, config["ACTION_DIM"])
                        else:
                            action_flat = jnp.reshape(action, (config["NUM_ENVS"], -1))

                        emi_params = ex_state.get("emi_params")
                        if emi_params is None:
                            emi_params = config["VAE_PARAMS"]

                        # Use the raw apply function for inner JAX-compiled loops
                        vae_apply = VAE_APPLY_FN_RAW_GLOBAL or VAE_APPLY_PJIT_GLOBAL
                        intrinsic_rewards, emi_info = vae_apply(
                            emi_params,
                            obs_flat,
                            action_flat,
                            next_obs_flat,
                            config["NOVELTY_WEIGHT"],
                            config["SURPRISE_WEIGHT"],
                            empowerment_weight=config["EMPOWERMENT_WEIGHT"],
                            return_components=True,
                        )

                        warmup_scale = _warmup_scale()
                        reward_i = reward_i + intrinsic_rewards * warmup_scale
                        ex_state = {**ex_state, "intrinsic_warmup_scale": warmup_scale}

                        # Update the EMI embedding pool; diversity rewards depend on it being populated
                        new_pool = emi_info.get('new_pool')
                        new_pool_idx = emi_info.get('new_pool_idx')
                        if new_pool is not None and new_pool_idx is not None:
                            ex_state = {
                                **ex_state,
                                "emi_params": {
                                    **emi_params,
                                    'embedding_pool': new_pool,
                                    'pool_idx': new_pool_idx,
                                },
                            }
                    else:
                        # SCIRE: VODM (novelty) + VFDM (surprise) + VEDM (empowerment)
                        obs_flat = jnp.reshape(last_obs, (config["NUM_ENVS"], config["OBS_DIM"]))
                        next_obs_flat = jnp.reshape(obsv, (config["NUM_ENVS"], config["OBS_DIM"]))

                        rng, scire_key = jax.random.split(rng)
                        scire_params = {"params": ex_state["scire_params"]}

                        intrinsic_rewards, scire_info = scire.compute_intrinsic_reward(
                            config["SCIRE_MODEL"],
                            scire_params,
                            obs_flat,
                            action,
                            next_obs_flat,
                            config["SCIRE_ACTION_SEQUENCES"],
                            scire_key,
                            novelty_weight=config["NOVELTY_WEIGHT"],
                            surprise_weight=config["SURPRISE_WEIGHT"],
                            empowerment_weight=config["EMPOWERMENT_WEIGHT"],
                            normalizer_state=ex_state["intrinsic_reward_normalizer"],
                            update_normalizer=True,
                        )

                        warmup_scale = _warmup_scale()
                        intrinsic_rewards = intrinsic_rewards * warmup_scale
                        reward_i = reward_i + intrinsic_rewards

                        accum = ex_state["intrinsic_components_accum"]
                        ex_state = {
                            **ex_state,
                            "intrinsic_reward_normalizer": scire_info["normalizer_state"],
                            "intrinsic_components_accum": {
                                "novelty_weighted": accum["novelty_weighted"] + config["NOVELTY_WEIGHT"] * scire_info["novelty_normalized"],
                                "surprise_weighted": accum["surprise_weighted"] + config["SURPRISE_WEIGHT"] * scire_info["surprise_normalized"],
                                "empowerment_weighted": accum["empowerment_weighted"] + config["EMPOWERMENT_WEIGHT"] * scire_info["empowerment_normalized"],
                            },
                            "intrinsic_warmup_scale": warmup_scale,
                        }

                if config["TRAIN_ICM"]:
                    latent_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, last_obs
                    )
                    latent_next_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, obsv
                    )

                    latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                        ex_state["icm_forward"].params, latent_obs, action
                    )
                    error = (latent_next_obs - latent_next_obs_pred) * (
                        1 - done[:, None]
                    )
                    mse = jnp.square(error).mean(axis=-1)

                    reward_i = mse * config["ICM_REWARD_COEFF"]

                if config.get("TRAIN_RND", False):
                    target_out = ex_state["rnd_target"].apply_fn(
                        ex_state["rnd_target"].params, obsv
                    )
                    predictor_out = ex_state["rnd_predictor"].apply_fn(
                        ex_state["rnd_predictor"].params, obsv
                    )
                    error = (target_out - predictor_out) * (1 - done[:, None])
                    mse = jnp.square(error).mean(axis=-1)

                    reward_i = reward_i + mse * config["RND_REWARD_COEFF"]

                if config.get("TRAIN_VIME", False):
                    obs_flat = jnp.reshape(last_obs, (config["NUM_ENVS"], config["OBS_DIM"]))
                    next_obs_flat = jnp.reshape(obsv, (config["NUM_ENVS"], config["OBS_DIM"]))
                    if config["ACTION_DIM"] > 1:
                        action_onehot = jax.nn.one_hot(action, config["ACTION_DIM"])
                    else:
                        action_onehot = jnp.reshape(action, (config["NUM_ENVS"], -1))
                    obs_action = jnp.concatenate([obs_flat, action_onehot], axis=-1)
                    target_delta = next_obs_flat - obs_flat

                    rng, _rng = jax.random.split(rng)
                    probe_keys = jax.random.split(_rng, config["NUM_ENVS"])
                    raw_kl = jax.vmap(
                        vime_jax.probe_kl_reward, in_axes=(None, 0, 0, 0, None, None)
                    )(
                        ex_state["vime_params"].params,
                        probe_keys,
                        obs_action,
                        target_delta,
                        config["VIME_STEP_SIZE"],
                        config["VIME_OUT_SIGMA"],
                    )
                    raw_kl = jnp.nan_to_num(raw_kl, nan=0.0, posinf=0.0, neginf=0.0)

                    kl_median = jnp.median(ex_state["vime_kl_buffer"])
                    normalized_kl = raw_kl / jnp.maximum(kl_median, 1e-8)
                    reward_i = reward_i + normalized_kl * config["VIME_REWARD_COEFF"]

                    buf = ex_state["vime_kl_buffer"]
                    buf_size = buf.shape[0]
                    write_positions = (ex_state["vime_buffer_idx"] + jnp.arange(config["NUM_ENVS"])) % buf_size
                    buf = buf.at[write_positions].set(raw_kl)
                    new_buffer_idx = (ex_state["vime_buffer_idx"] + config["NUM_ENVS"]) % buf_size
                    ex_state = {
                        **ex_state,
                        "vime_kl_buffer": buf,
                        "vime_buffer_idx": new_buffer_idx,
                    }

                if config.get("USE_COUNT_BASED_NOVELTY", False):
                    # Independent of SCIRE: count-based exploration replaces novelty only,
                    # scaled directly by COUNT_BASED_BONUS_COEF (no normalizer needed).
                    obs_flat = jnp.reshape(last_obs, (config["NUM_ENVS"], config["OBS_DIM"]))
                    count_bonus, updated_count_table = compute_batch_bonus_jax(
                        obs_flat,
                        ex_state["count_table"],
                        config.get("COUNT_BASED_BONUS_COEF", 1.0),
                        config.get("COUNT_TABLE_SIZE", 100000),
                    )
                    reward_i = reward_i + count_bonus
                    ex_state = {**ex_state, "count_table": updated_count_table}

                reward = reward_e + reward_i

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    value_ext=value_ext,
                    value_int=value_int,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                )
                runner_state_inner = (
                    train_state,
                    env_state,
                    obsv,
                    ex_state,
                    rng,
                    update_step,
                )
                # Debug: optionally log shapes/types of transition fields using logging at DEBUG level
                try:
                    if config.get("DEBUG", False):
                        def _dbg_print_transition_shapes(t):
                            try:
                                for field in getattr(t, '_fields', []):
                                    arr = getattr(t, field)
                                    if hasattr(arr, 'shape'):
                                        logger.debug("TRANS_FIELD_SHAPE: %s -> %s", field, getattr(arr, 'shape', None))
                                    else:
                                        logger.debug("TRANS_FIELD_TYPE: %s -> %s", field, type(arr))
                            except Exception as e:
                                logger.debug("DEBUG TRANS PRINT FAILED: %s", e)
                        jax.debug.callback(_dbg_print_transition_shapes, transition)
                except Exception:
                    pass
                return runner_state_inner, transition

            # Create inner runner_state without best policy tracking for _env_step scan
            runner_state_inner = (train_state, env_state, last_obs, ex_state, rng, update_step)
            runner_state_inner, traj_batch = jax.lax.scan(
                _env_step, runner_state_inner, None, config["NUM_STEPS"]
            )
            # Debug: optionally log shapes/types of the trajectory batch fields using logging at DEBUG level
            try:
                if config.get("DEBUG", False):
                    def _dbg_print_traj_shapes(tb):
                        try:
                            for field in getattr(tb, '_fields', []):
                                arr = getattr(tb, field)
                                if hasattr(arr, 'shape'):
                                    logger.debug("TRJ_FIELD_SHAPE: %s -> %s", field, getattr(arr, 'shape', None))
                                else:
                                    logger.debug("TRJ_FIELD_TYPE: %s -> %s", field, type(arr))
                        except Exception as e:
                            logger.debug("DEBUG TRJ SHAPE PRINT FAILED: %s", e)
                    jax.debug.callback(_dbg_print_traj_shapes, traj_batch)
            except Exception:
                pass

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step,
            ) = runner_state_inner
            
            # Note: SCIRE's normalizer is updated inline every step in _env_step
            # (update_normalizer=True), so no periodic batch update is needed here.

            # Fix observation shape if it's a tuple (happens in some configurations)
            if isinstance(last_obs, tuple):
                # Take the second element which should be the actual observation
                actual_obs = last_obs[1] if len(last_obs) > 1 else last_obs[0]
            else:
                actual_obs = last_obs
            
            # Get last value(s) based on network type
            if should_use_dual_value_heads(config):
                _, last_val_ext, last_val_int = network.apply(train_state.params, actual_obs)
                last_val = last_val_ext + last_val_int
            else:
                _, last_val = network.apply(train_state.params, actual_obs)
                last_val_ext = last_val
                last_val_int = jnp.zeros_like(last_val)

            def _calculate_gae_dual(traj_batch, last_val_ext, last_val_int):
                """Calculate separate GAE for extrinsic and intrinsic rewards"""
                def _get_advantages_ext(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value_ext,
                        transition.reward_e,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                def _get_advantages_int(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value_int,
                        transition.reward_i,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages_ext = jax.lax.scan(
                    _get_advantages_ext,
                    (jnp.zeros_like(last_val_ext), last_val_ext),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                _, advantages_int = jax.lax.scan(
                    _get_advantages_int,
                    (jnp.zeros_like(last_val_int), last_val_int),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                
                # Combined advantages for policy gradient
                advantages = advantages_ext + advantages_int
                # Separate targets for each value head
                targets_ext = advantages_ext + traj_batch.value_ext
                targets_int = advantages_int + traj_batch.value_int
                
                return advantages, targets_ext, targets_int

            def _calculate_gae(traj_batch, last_val):
                """Legacy single-value GAE calculation"""
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            # Use dual GAE if dual value heads enabled
            if should_use_dual_value_heads(config):
                advantages, targets_ext, targets_int = _calculate_gae_dual(traj_batch, last_val_ext, last_val_int)
                targets = targets_ext + targets_int  # Combined for logging
            else:
                advantages, targets = _calculate_gae(traj_batch, last_val)
                targets_ext = targets
                targets_int = jnp.zeros_like(targets)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    if should_use_dual_value_heads(config):
                        traj_batch, advantages, targets_ext_batch, targets_int_batch = batch_info
                    else:
                        traj_batch, advantages, targets = batch_info
                        targets_ext_batch = targets
                        targets_int_batch = jnp.zeros_like(targets)

                    # Policy/value network
                    def _loss_fn_dual(params, traj_batch, gae, targets_ext, targets_int):
                        """Loss function with separate value heads for intrinsic/extrinsic"""
                        # RERUN NETWORK
                        pi, value_ext, value_int = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE EXTRINSIC VALUE LOSS
                        value_ext_pred_clipped = traj_batch.value_ext + (
                            value_ext - traj_batch.value_ext
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_ext = jnp.square(value_ext - targets_ext)
                        value_losses_ext_clipped = jnp.square(value_ext_pred_clipped - targets_ext)
                        value_loss_ext = (
                            0.5 * jnp.maximum(value_losses_ext, value_losses_ext_clipped).mean()
                        )

                        # CALCULATE INTRINSIC VALUE LOSS
                        value_int_pred_clipped = traj_batch.value_int + (
                            value_int - traj_batch.value_int
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_int = jnp.square(value_int - targets_int)
                        value_losses_int_clipped = jnp.square(value_int_pred_clipped - targets_int)
                        value_loss_int = (
                            0.5 * jnp.maximum(value_losses_int, value_losses_int_clipped).mean()
                        )

                        # Combined value loss
                        value_loss = value_loss_ext + value_loss_int

                        # CALCULATE ACTOR LOSS (uses combined advantages)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, value_loss_ext, value_loss_int)

                    def _loss_fn(params, traj_batch, gae, targets):
                        """Legacy single-value loss function"""
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    # Use appropriate loss function
                    if should_use_dual_value_heads(config):
                        grad_fn = jax.value_and_grad(_loss_fn_dual, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets_ext_batch, targets_int_batch
                        )
                    else:
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets_ext_batch
                        )
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, 0)
                    return train_state, losses

                # Prepare update state based on value head configuration
                if should_use_dual_value_heads(config):
                    (
                        train_state,
                        traj_batch,
                        advantages,
                        targets_ext,
                        targets_int,
                        rng,
                    ) = update_state
                    targets = targets_ext + targets_int  # For logging
                else:
                    (
                        train_state,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    ) = update_state
                    targets_ext = targets
                    targets_int = jnp.zeros_like(targets)
                
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                
                if should_use_dual_value_heads(config):
                    batch = (traj_batch, advantages, targets_ext, targets_int)
                else:
                    batch = (traj_batch, advantages, targets)
                
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                
                if should_use_dual_value_heads(config):
                    update_state = (
                        train_state,
                        traj_batch,
                        advantages,
                        targets_ext,
                        targets_int,
                        rng,
                    )
                else:
                    update_state = (
                        train_state,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    )
                return update_state, losses

            if should_use_dual_value_heads(config):
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets_ext,
                    targets_int,
                    rng,
                )
            else:
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
            
            # Save old train_state for episode monitoring (before policy update)
            # Episode achievements come from rollouts with the OLD policy, not the updated one
            old_train_state_for_episodes = train_state
            
            # Also save traj_batch for action sequence extraction
            traj_batch_for_episodes = traj_batch
            
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            # Filter out achievements from metrics to avoid broadcasting issues, except for MiniGrid
            if "MiniGrid" in config["ENV_NAME"]:
                # Keep MiniGrid achievement metrics
                # Filter out episode action buffers and achievement arrays
                filtered_info = {k: v for k, v in traj_batch.info.items() 
                               if k not in ["returned_episode_achievements", "episode_actions", "episode_action_count"]}
            else:
                # Filter out all achievements and action buffers for other environments
                filtered_info = {k: v for k, v in traj_batch.info.items() 
                               if k not in ["returned_episode_achievements", "episode_actions", "episode_action_count"]}
            # Ensure 'returned_episode' is present in info; fallback to zeros and print debug keys if missing
            if not hasattr(traj_batch, 'info'):
                if config.get('DEBUG', False):
                    logger.debug("DEBUG: traj_batch has no 'info' attribute; skipping per-episode metrics")
                # Create safe zero mask to avoid division by zero
                returned_episode_mask = jnp.zeros((config['NUM_STEPS'], config['NUM_ENVS']), dtype=jnp.int32)
            else:
                # If missing 'returned_episode', log keys and create zero mask
                if 'returned_episode' not in traj_batch.info:
                    try:
                        # Print keys for debugging
                        if config.get('DEBUG', False):
                            logger.debug("DEBUG: traj_batch.info keys: %s", list(traj_batch.info.keys()))
                    except Exception:
                        if config.get('DEBUG', False):
                            logger.debug("DEBUG: Could not print traj_batch.info keys")
                    # Create safe zero mask matching expected info shape if possible
                    sample_shape = None
                    try:
                        # Pick any key's shape to determine shape
                        for v in traj_batch.info.values():
                            if hasattr(v, 'shape'):
                                sample_shape = v.shape
                                break
                    except Exception:
                        sample_shape = None
                    if sample_shape is not None and len(sample_shape) >= 2:
                        returned_episode_mask = jnp.zeros(sample_shape[:2], dtype=jnp.int32)
                    else:
                        returned_episode_mask = jnp.zeros((config['NUM_STEPS'], config['NUM_ENVS']), dtype=jnp.int32)
                else:
                    returned_episode_mask = traj_batch.info['returned_episode']

            metric = jax.tree.map(
                lambda x: (x * returned_episode_mask).sum() / jnp.maximum(1, returned_episode_mask.sum()),
                filtered_info,
            )

            rng = update_state[-1]

            # UPDATE EXPLORATION STATE
            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(ex_state, traj_batch):
                    def _inverse_loss_fn(
                        icm_encoder_params, icm_inverse_params, traj_batch
                    ):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.next_obs
                        )

                        action_pred_logits = ex_state["icm_inverse"].apply_fn(
                            icm_inverse_params, latent_obs, latent_next_obs
                        )
                        true_action = jax.nn.one_hot(
                            traj_batch.action, num_classes=action_pred_logits.shape[-1]
                        )

                        bce = -jnp.mean(
                            jnp.sum(
                                action_pred_logits
                                * true_action
                                * (1 - traj_batch.done[:, None]),
                                axis=1,
                            )
                        )

                        return bce * config["ICM_INVERSE_LOSS_COEF"]

                    inverse_grad_fn = jax.value_and_grad(
                        _inverse_loss_fn,
                        has_aux=False,
                        argnums=(
                            0,
                            1,
                        ),
                    )
                    inverse_loss, grads = inverse_grad_fn(
                        ex_state["icm_encoder"].params,
                        ex_state["icm_inverse"].params,
                        traj_batch,
                    )
                    icm_encoder_grad, icm_inverse_grad = grads
                    updated_icm_encoder = ex_state["icm_encoder"].apply_gradients(
                        grads=icm_encoder_grad
                    )
                    updated_icm_inverse = ex_state["icm_inverse"].apply_gradients(
                        grads=icm_inverse_grad
                    )
                    ex_state = {
                        **ex_state,
                        "icm_encoder": updated_icm_encoder,
                        "icm_inverse": updated_icm_inverse,
                    }

                    def _forward_loss_fn(icm_forward_params, traj_batch):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.next_obs
                        )

                        latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                            icm_forward_params, latent_obs, traj_batch.action
                        )

                        error = (latent_next_obs - latent_next_obs_pred) * (
                            1 - traj_batch.done[:, None]
                        )
                        return (
                            jnp.square(error).mean() * config["ICM_FORWARD_LOSS_COEF"]
                        )

                    forward_grad_fn = jax.value_and_grad(
                        _forward_loss_fn, has_aux=False
                    )
                    forward_loss, icm_forward_grad = forward_grad_fn(
                        ex_state["icm_forward"].params, traj_batch
                    )
                    updated_icm_forward = ex_state["icm_forward"].apply_gradients(
                        grads=icm_forward_grad
                    )
                    ex_state = {
                        **ex_state,
                        "icm_forward": updated_icm_forward,
                    }

                    losses = (inverse_loss, forward_loss)
                    return ex_state, losses

                (ex_state, traj_batch, rng) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                )
                # Debug: optionally log shapes/types of batch fields before shuffling using logging at DEBUG level
                try:
                    if config.get("DEBUG", False):
                        def _dbg_print_batch_shapes(b):
                            try:
                                for field in getattr(b, '_fields', []):
                                    arr = getattr(b, field)
                                    if hasattr(arr, 'shape'):
                                        logger.debug("BATCH_FIELD_SHAPE: %s -> %s", field, getattr(arr, 'shape', None))
                                    else:
                                        logger.debug("BATCH_FIELD_TYPE: %s -> %s", field, type(arr))
                            except Exception as e:
                                logger.debug("DEBUG BATCH SHAPE PRINT FAILED: %s", e)
                        jax.debug.callback(_dbg_print_batch_shapes, batch)
                except Exception:
                    pass
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                ex_state, losses = jax.lax.scan(
                    _update_ex_minbatch, ex_state, minibatches
                )
                update_state = (ex_state, traj_batch, rng)
                return update_state, losses

            if config["TRAIN_ICM"]:
                ex_update_state = (ex_state, traj_batch, rng)
                ex_update_state, ex_loss = jax.lax.scan(
                    _update_ex_epoch,
                    ex_update_state,
                    None,
                    config["EXPLORATION_UPDATE_EPOCHS"],
                )
                metric["icm_inverse_loss"] = ex_loss[0].mean()
                metric["icm_forward_loss"] = ex_loss[1].mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

                ex_state = ex_update_state[0]
                rng = ex_update_state[-1]

            # TRAIN RND PREDICTOR NETWORK
            if config.get("TRAIN_RND", False):
                def _update_rnd_epoch(update_state, unused):
                    def _update_rnd_minibatch(rnd_predictor, traj_batch):
                        def _rnd_loss_fn(rnd_predictor_params, traj_batch):
                            target_out = ex_state["rnd_target"].apply_fn(
                                ex_state["rnd_target"].params, traj_batch.next_obs
                            )
                            predictor_out = rnd_predictor.apply_fn(
                                rnd_predictor_params, traj_batch.next_obs
                            )
                            error = (target_out - predictor_out) * (
                                1 - traj_batch.done[:, None]
                            )
                            return jnp.square(error).mean() * config["RND_LOSS_COEFF"]

                        rnd_grad_fn = jax.value_and_grad(_rnd_loss_fn, has_aux=False)
                        rnd_loss, rnd_grad = rnd_grad_fn(
                            rnd_predictor.params, traj_batch
                        )
                        rnd_predictor = rnd_predictor.apply_gradients(grads=rnd_grad)
                        return rnd_predictor, rnd_loss

                    (rnd_predictor, traj_batch, rng) = update_state
                    rng, _rng = jax.random.split(rng)
                    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                    permutation = jax.random.permutation(_rng, batch_size)
                    batch = jax.tree.map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                    )
                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    minibatches = jax.tree.map(
                        lambda x: jnp.reshape(
                            x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                        ),
                        shuffled_batch,
                    )
                    rnd_predictor, losses = jax.lax.scan(
                        _update_rnd_minibatch, rnd_predictor, minibatches
                    )
                    update_state = (rnd_predictor, traj_batch, rng)
                    return update_state, losses

                rnd_update_state = (ex_state["rnd_predictor"], traj_batch, rng)
                rnd_update_state, rnd_loss = jax.lax.scan(
                    _update_rnd_epoch,
                    rnd_update_state,
                    None,
                    config.get("EXPLORATION_UPDATE_EPOCHS", 16),
                )
                metric["rnd_loss"] = rnd_loss.mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

                ex_state = {**ex_state, "rnd_predictor": rnd_update_state[0]}
                rng = rnd_update_state[-1]

            # TRAIN VIME BAYESIAN DYNAMICS MODEL
            if config.get("TRAIN_VIME", False):
                def _update_vime_epoch(update_state, unused):
                    def _update_vime_minibatch(vime_train_state, batch_and_key):
                        batch, key = batch_and_key
                        obs_flat = batch.obs.reshape(-1, config["OBS_DIM"])
                        next_obs_flat = batch.next_obs.reshape(-1, config["OBS_DIM"])
                        action_raw = batch.action.reshape(-1)
                        if config["ACTION_DIM"] > 1:
                            action_onehot = jax.nn.one_hot(action_raw, config["ACTION_DIM"])
                        else:
                            action_onehot = jnp.reshape(action_raw, (-1, 1))
                        obs_action_batch = jnp.concatenate([obs_flat, action_onehot], axis=-1)
                        target_delta_batch = next_obs_flat - obs_flat
                        n_total = config["NUM_ENVS"] * config["NUM_STEPS"]

                        def _loss_fn(params):
                            return vime_jax.elbo_loss(
                                params, key, obs_action_batch, target_delta_batch,
                                config["VIME_OUT_SIGMA"], config["VIME_PRIOR_STD"], n_total,
                            )

                        loss, grads = jax.value_and_grad(_loss_fn)(vime_train_state.params)
                        vime_train_state = vime_train_state.apply_gradients(grads=grads)
                        return vime_train_state, loss

                    (vime_train_state, traj_batch, rng) = update_state
                    rng, _rng = jax.random.split(rng)
                    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                    permutation = jax.random.permutation(_rng, batch_size)
                    batch = jax.tree.map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                    )
                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    minibatches = jax.tree.map(
                        lambda x: jnp.reshape(
                            x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                        ),
                        shuffled_batch,
                    )
                    rng, _rng = jax.random.split(rng)
                    minibatch_keys = jax.random.split(_rng, config["NUM_MINIBATCHES"])
                    vime_train_state, losses = jax.lax.scan(
                        _update_vime_minibatch, vime_train_state, (minibatches, minibatch_keys)
                    )
                    update_state = (vime_train_state, traj_batch, rng)
                    return update_state, losses

                vime_update_state = (ex_state["vime_params"], traj_batch, rng)
                vime_update_state, vime_loss = jax.lax.scan(
                    _update_vime_epoch,
                    vime_update_state,
                    None,
                    config.get("EXPLORATION_UPDATE_EPOCHS", 16),
                )
                metric["vime_loss"] = vime_loss.mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

                ex_state = {**ex_state, "vime_params": vime_update_state[0]}
                rng = vime_update_state[-1]

            # TRAIN SCIRE (VODM+VFDM) ON COLLECTED TRAJECTORIES
            # Skip this for EMI - EMI has its own training mechanism
            if config.get("USE_JAX_INTRINSIC_REWARDS", False) and not config.get("USE_EMI", False):
                def _update_scire_epoch(update_state, unused):
                    def _update_scire_minibatch(carry, batch_and_key):
                        scire_params = carry
                        batch, key = batch_and_key

                        # Flatten batch data for training
                        obs_flat = batch.obs.reshape(-1, config["OBS_DIM"])
                        action_raw = batch.action.reshape(-1)
                        next_obs_flat = batch.next_obs.reshape(-1, config["OBS_DIM"])

                        scire_model = config["SCIRE_MODEL"]

                        # Define loss function for gradient computation
                        def _loss_fn(nn_params):
                            flax_params = {'params': nn_params}
                            beta_kl = config.get("BETA_KL", 0.1)

                            loss, loss_info = scire_model.apply(
                                flax_params, obs_flat, action_raw, next_obs_flat, key, beta_kl,
                                method=scire_model.compute_training_loss
                            )
                            return loss, loss_info

                        # Compute gradients
                        (loss, loss_info), grads = jax.value_and_grad(_loss_fn, has_aux=True)(scire_params)

                        # Clip gradients for stability
                        max_grad_norm = config.get("MAX_GRAD_NORM", 1.0)
                        grads_flat = jax.tree_util.tree_leaves(grads)
                        total_grad_norm = jnp.sqrt(sum([jnp.sum(g**2) for g in grads_flat]))
                        clip_coef = jnp.minimum(1.0, max_grad_norm / (total_grad_norm + 1e-6))
                        clipped_grads = jax.tree.map(lambda g: g * clip_coef, grads)

                        # Apply gradients with configurable learning rate
                        # Use lower learning rate for deep networks (3e-4 vs 1e-3)
                        learning_rate = config.get("VAE_LEARNING_RATE", 3e-4)
                        updated_scire_params = jax.tree.map(
                            lambda p, g: p - learning_rate * g,
                            scire_params,
                            clipped_grads
                        )

                        return updated_scire_params, (loss, loss_info)

                    scire_params, batch, rng_scire = update_state
                    rng_scire, _rng = jax.random.split(rng_scire)

                    # Prepare minibatches
                    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                    permutation = jax.random.permutation(_rng, batch_size)
                    batch_flat = jax.tree.map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=0), batch_flat
                    )
                    minibatches = jax.tree.map(
                        lambda x: jnp.reshape(
                            x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                        ),
                        shuffled_batch,
                    )

                    # Generate random keys for each minibatch
                    rng_scire, _rng = jax.random.split(rng_scire)
                    minibatch_keys = jax.random.split(_rng, config["NUM_MINIBATCHES"])

                    # Update over minibatches (pass keys alongside batches)
                    scire_params, losses = jax.lax.scan(
                        _update_scire_minibatch, scire_params, (minibatches, minibatch_keys)
                    )

                    return (scire_params, batch, rng_scire), losses

                # Get current SCIRE neural network params from ex_state
                rng, rng_scire = jax.random.split(rng)
                current_scire_params = ex_state["scire_params"]

                # Train for multiple epochs
                scire_update_state = (current_scire_params, traj_batch, rng_scire)
                scire_update_state, scire_losses = jax.lax.scan(
                    _update_scire_epoch,
                    scire_update_state,
                    None,
                    config.get("EXPLORATION_UPDATE_EPOCHS", 16),
                )

                # Update ex_state with trained parameters
                ex_state = {
                    **ex_state,
                    "scire_params": scire_update_state[0],
                }

                # Log training metrics
                loss_values, loss_info_values = scire_losses

                # Average over all epochs and minibatches
                metric["scire/total_loss"] = jnp.mean(loss_values)
                metric["scire/recon_loss"] = jnp.mean(loss_info_values['recon_loss'])
                metric["scire/kl_loss"] = jnp.mean(loss_info_values['kl_loss'])
                metric["scire/pred_loss"] = jnp.mean(loss_info_values['pred_loss'])
                metric["scire/latent_pred_loss"] = jnp.mean(loss_info_values['latent_pred_loss'])

                # Log intrinsic/extrinsic rewards for JAX intrinsic rewards
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

            # TRAIN EMI (Dynamical State-Action Embedding) if enabled
            if config.get("USE_EMI", False):
                # Get the EMI training function from config (closure that captures JIT function)
                emi_train_fn = config.get("EMI_TRAIN_FN")
                
                if emi_train_fn is not None:
                    # EMI training: optimize forward dynamics and mutual information objectives
                    def _update_emi_epoch(update_state, unused):
                        def _update_emi_minibatch(emi_params, batch_and_key):
                            batch, key = batch_and_key
                            # Flatten batch data for training
                            obs_flat = batch.obs.reshape(-1, config["OBS_DIM"])
                            action_raw = batch.action.reshape(-1)
                            next_obs_flat = batch.next_obs.reshape(-1, config["OBS_DIM"])
                            
                            # Convert actions to one-hot if discrete
                            if config["ACTION_DIM"] > 1:  # Discrete action space
                                action_flat = jax.nn.one_hot(action_raw, config["ACTION_DIM"])
                            else:
                                action_flat = jnp.reshape(action_raw, (-1, 1))
                            
                            # Use the EMI training function from config (closure with JIT)
                            updated_emi_params, metrics = emi_train_fn(
                                emi_params, obs_flat, action_flat, next_obs_flat, key
                            )
                            
                            return updated_emi_params, metrics
                        
                        emi_params, batch, rng_emi = update_state
                        rng_emi, _rng = jax.random.split(rng_emi)
                        
                        # Prepare minibatches with corresponding keys
                        batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                        permutation = jax.random.permutation(_rng, batch_size)
                        batch_flat = jax.tree.map(
                            lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                        )
                        shuffled_batch = jax.tree.map(
                            lambda x: jnp.take(x, permutation, axis=0), batch_flat
                        )
                        minibatches = jax.tree.map(
                            lambda x: jnp.reshape(
                                x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                            ),
                            shuffled_batch,
                        )
                        
                        # Generate keys for each minibatch
                        minibatch_keys = jax.random.split(_rng, config["NUM_MINIBATCHES"])
                        
                        # Update over minibatches
                        emi_params, losses = jax.lax.scan(
                            _update_emi_minibatch, emi_params, (minibatches, minibatch_keys)
                        )
                        
                        return (emi_params, batch, rng_emi), losses
                    
                    # Get current EMI params from ex_state
                    rng, rng_emi = jax.random.split(rng)
                    current_emi_params = ex_state.get("emi_params")
                    if current_emi_params is None:
                        # First time - get from config
                        current_emi_params = config["VAE_PARAMS"]
                    
                    # Train for multiple epochs
                    emi_update_state = (current_emi_params, traj_batch, rng_emi)
                    emi_update_state, emi_losses = jax.lax.scan(
                        _update_emi_epoch,
                        emi_update_state,
                        None,
                        config.get("EXPLORATION_UPDATE_EPOCHS", 16),
                    )
                    
                    # Update ex_state with trained parameters
                    ex_state = {
                        **ex_state,
                        "emi_params": emi_update_state[0],
                    }
                    
                    # Log EMI training metrics
                    metric["emi/total_loss"] = jnp.mean(emi_losses['total_loss'])
                    metric["emi/dynamics_loss"] = jnp.mean(emi_losses['dynamics_loss'])
                    metric["emi/mi_action_loss"] = jnp.mean(emi_losses['mi_action_loss'])
                    metric["emi/mi_obs_loss"] = jnp.mean(emi_losses['mi_obs_loss'])
                    
                    # Log intrinsic/extrinsic rewards for EMI
                    metric["reward_i"] = traj_batch.reward_i.mean()
                    metric["reward_e"] = traj_batch.reward_e.mean()

            # Best policy tracking is handled in the callback below
            # The callback has access to processed metrics in to_log dict
            # We'll keep placeholder values in runner_state for structure compatibility
            if config.get("DEBUG", False) and best_policy_tracker['save_best']:
                # Extract current metric from the logged metrics
                # Try goal achievement metrics first (more stable than episode return)
                # We'll try to extract metrics in order of preference
                current_metric_jax = jnp.float32(-jnp.inf)
                
                # Helper function to safely extract metric value
                def safe_extract(val):
                    """Extract scalar from potentially array value"""
                    val_array = jnp.asarray(val, dtype=jnp.float32)
                    # If it has a non-zero shape, take mean; otherwise use as-is
                    return jnp.where(val_array.size > 1, jnp.mean(val_array), val_array.reshape(()))
                
                # Debug: print available metrics (only once, using logging at DEBUG level)
                def debug_print_metrics(metric_dict, step):
                    try:
                        if step == 0:  # Only print on first update
                            logger.debug("DEBUG: Available metrics in dict: %s", list(metric_dict.keys()))
                    except Exception as e:
                        logger.debug("DEBUG PRINT METRICS FAILED: %s", e)

                try:
                    jax.debug.callback(debug_print_metrics, metric, update_step)
                except Exception:
                    pass
                
                # Try each metric in order
                # Based on debug output, 'returned_episode_returns' is available
                # This contains the actual episode returns
                if 'returned_episode_returns' in metric:
                    val = metric['returned_episode_returns']
                    # This is an array of returns, take the mean of non-zero values
                    val_array = jnp.asarray(val, dtype=jnp.float32)
                    # Flatten and filter out zeros, then take mean
                    val_flat = val_array.flatten()
                    # Count non-zero elements
                    non_zero_mask = val_flat != 0.0
                    non_zero_count = jnp.sum(non_zero_mask)
                    # Take mean of non-zero values, or 0 if all are zero
                    current_metric_jax = jnp.where(
                        non_zero_count > 0,
                        jnp.sum(val_flat * non_zero_mask) / non_zero_count,
                        jnp.float32(0.0)
                    )
                elif 'any_goal_achievement_rate' in metric:
                    current_metric_jax = safe_extract(metric['any_goal_achievement_rate'])
                elif 'small_reward_goal_rate' in metric:
                    current_metric_jax = safe_extract(metric['small_reward_goal_rate'])
                elif 'large_reward_goal_rate' in metric:
                    current_metric_jax = safe_extract(metric['large_reward_goal_rate'])
                elif 'episode_return' in metric:
                    current_metric_jax = safe_extract(metric['episode_return'])
                elif 'episode_returns' in metric:
                    current_metric_jax = safe_extract(metric['episode_returns'])
                elif 'mean_episode_return' in metric:
                    current_metric_jax = safe_extract(metric['mean_episode_return'])
                
                # Update best policy tracking using JAX conditionals
                is_new_best = current_metric_jax > best_metric
                
                # Debug: print when we update (first few updates only)
                def debug_print_update(curr, prev, step, is_best, new_best_metric, new_best_step):
                    try:
                        if step < 3:  # Only print first few updates
                            logger.debug("DEBUG UPDATE: step=%s, current_metric=%s, prev_best=%s, is_new_best=%s", step, curr, prev, is_best)
                            logger.debug("  -> After update: best_metric=%s, best_step=%s", new_best_metric, new_best_step)
                    except Exception as e:
                        logger.debug("DEBUG PRINT UPDATE FAILED: %s", e)
                # Update the values
                new_best_metric = jnp.where(is_new_best, current_metric_jax, best_metric)
                new_best_step = jnp.where(is_new_best, update_step, best_step)
                new_best_train_state = jax.lax.cond(
                    is_new_best,
                    lambda: train_state,
                    lambda: best_train_state
                )
                
                try:
                    jax.debug.callback(debug_print_update, current_metric_jax, best_metric, update_step, is_new_best, new_best_metric, new_best_step)
                except Exception:
                    pass
                
                # IMPORTANT: Assign the new values back to the variables
                best_metric = new_best_metric
                best_step = new_best_step
                best_train_state = new_best_train_state
            
            # Logging callback - always enabled for metrics tracking (needed for HPO/evolutionary optimization)
            # The callback writes metrics to JSONL file which is essential for fitness computation
            def callback(metric, update_step, train_state, old_train_state_for_episodes, traj_batch_for_episodes, ex_state_for_logging, best_metric_val, best_step_val, env_state_for_checkpoint, obs_for_checkpoint):
                # Debug: confirm callback is being called
                if int(update_step) % (config.get('CHECKPOINT_FREQ', 1)) == 0:  # Log every CHECKPOINT_FREQ steps to avoid spam
                    print(f"[DEBUG] Callback called at update_step={update_step}, CHECKPOINT_FREQ={config.get('CHECKPOINT_FREQ', 0)}, NUM_UPDATES={config.get('NUM_UPDATES', 0)}")
                
                # Calculate total timesteps AFTER this update
                # update_step is 0-indexed, so we add 1 to get timesteps collected so far
                # This represents the total environment steps taken up to and including this update
                total_timesteps = (update_step + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]
                
                to_log = create_log_dict(metric, config)
                
                # Only log training loss metrics (model training progress)
                # Skip per-step intrinsic component losses as they're logged via episode aggregates
                for key in metric.keys():
                    if any(loss_type in key for loss_type in ['scire', 'icm', 'emi', 'rnd', 'vime']):
                        to_log[key] = metric[key]
                
                # Add intrinsic/extrinsic rewards if JAX intrinsic rewards are enabled
                if config.get("USE_JAX_INTRINSIC_REWARDS", False):
                    novelty_weight = config.get("NOVELTY_WEIGHT", 0)
                    surprise_weight = config.get("SURPRISE_WEIGHT", 0)
                    empowerment_weight = config.get("EMPOWERMENT_WEIGHT", 0)
                    
                    # Check if any intrinsic component is actually enabled
                    intrinsics_enabled = (novelty_weight > 0 or surprise_weight > 0 or empowerment_weight > 0)
                    
                    # Get mean intrinsic and extrinsic rewards from trajectory
                    if hasattr(traj_batch_for_episodes, 'reward_i'):
                        intrinsic_mean = jnp.mean(traj_batch_for_episodes.reward_i)
                        extrinsic_mean = jnp.mean(traj_batch_for_episodes.reward_e)
                        
                        # Only log intrinsic metrics if intrinsics are enabled
                        if intrinsics_enabled:
                            to_log["intrinsic_reward_mean"] = intrinsic_mean
                            to_log["extrinsic_reward_mean"] = extrinsic_mean
                            
                            # Log the weights for reference
                            to_log["novelty_weight"] = novelty_weight
                            to_log["surprise_weight"] = surprise_weight
                            to_log["empowerment_weight"] = empowerment_weight
                            
                            # Log warmup scale if available
                            if "intrinsic_warmup_scale" in ex_state_for_logging:
                                to_log["intrinsic_warmup_scale"] = ex_state_for_logging["intrinsic_warmup_scale"]
                            
                            # Only log ratio if both intrinsic and extrinsic are meaningful
                            if intrinsic_mean > 0 and extrinsic_mean > 0:
                                ratio = intrinsic_mean / extrinsic_mean
                                to_log["intrinsic_extrinsic_ratio"] = ratio
                        
                        # Log accumulated intrinsic components (per-episode averages - most informative)
                        if (
                            intrinsics_enabled
                            and "intrinsic_components_accum" in ex_state_for_logging
                            and ex_state_for_logging["intrinsic_components_accum"]
                            and hasattr(traj_batch_for_episodes, 'info')
                            and 'returned_episode' in traj_batch_for_episodes.info
                        ):
                            accum = ex_state_for_logging["intrinsic_components_accum"]
                            returned_episodes = traj_batch_for_episodes.info['returned_episode']
                            returned_episodes_int = returned_episodes.astype(jnp.int32)
                            total_episodes = jnp.sum(returned_episodes_int)
                            
                            if total_episodes > 0:
                                # Log per-episode averages for enabled components only
                                if novelty_weight > 0 and 'novelty_weighted' in accum:
                                    to_log["intrinsic/novelty_episode_mean"] = (
                                        jnp.sum(accum['novelty_weighted']) / total_episodes
                                    )
                                if surprise_weight > 0 and 'surprise_weighted' in accum:
                                    to_log["intrinsic/surprise_episode_mean"] = (
                                        jnp.sum(accum['surprise_weighted']) / total_episodes
                                    )
                                if empowerment_weight > 0 and 'empowerment_weighted' in accum:
                                    to_log["intrinsic/empowerment_episode_mean"] = (
                                        jnp.sum(accum['empowerment_weighted']) / total_episodes
                                    )
                
                # Also save all raw metrics to JSONL file before aggregation
                import json
                import math
                stats_path = os.path.join(config.get("RECORD_DIR", generate_recording_dir(config)), 'metrics.jsonl')
                os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                
                # Convert to_log to a serializable format and add timestep info
                metrics_data = {
                    'update_step': int(update_step),
                    'total_timesteps': int(total_timesteps),
                }
                
                # Add all metrics from to_log
                for k, v in to_log.items():
                    try:
                        # Convert JAX arrays and numpy arrays to Python types
                        if hasattr(v, 'item'):
                            # 0-dimensional array - use item()
                            val = float(v.item())
                        elif isinstance(v, (int, float)):
                            # Already a Python scalar
                            val = float(v)
                        elif hasattr(v, '__len__'):
                            # Array with length - take mean
                            if len(v) > 0:
                                val = float(jnp.mean(v))
                            else:
                                continue  # Skip empty arrays
                        else:
                            # Try direct conversion
                            val = float(v)
                        
                        # Skip NaN and inf values
                        if math.isnan(val) or math.isinf(val):
                            continue
                        
                        metrics_data[k] = val
                    except (ValueError, TypeError, AttributeError):
                        # Skip non-numeric values or conversion errors
                        pass
                
                # Write to JSONL if we have meaningful metrics
                # Always write at least timestep info for evolutionary/HPO tracking
                try:
                    with open(stats_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(metrics_data) + '\n')
                except Exception as e:
                    # Log write failure but don't crash training
                    pass
                
                # Now call batch_log which will aggregate and log to wandb
                batch_log(update_step, to_log, config, total_timesteps)
                
                # Check for episode achievements and save policies if found
                # IMPORTANT: Use old_train_state_for_episodes (before update) because
                # the episode achievements in metrics come from rollouts with the OLD policy
                # Also pass traj_batch to extract action sequences
                if episode_monitor is not None:
                    episode_monitor.check_for_achievements(update_step, to_log, old_train_state_for_episodes, traj_batch_for_episodes)
                
                # Track best policy if enabled
                if best_policy_tracker['save_best']:
                    # Use goal achievement rate as the primary metric for best policy tracking
                    # This provides more stable policy selection than episode return which can be noisy
                    current_metric = None
                    # Try goal achievement metrics first (more stable than episode return)
                    for metric_name in ['any_goal_achievement_rate', 'small_reward_goal_rate', 'large_reward_goal_rate']:
                        if metric_name in to_log and to_log[metric_name] is not None:
                            try:
                                # Handle different metric formats (scalar, array, etc.)
                                val = to_log[metric_name]
                                if hasattr(val, 'item'):
                                    current_metric = float(val.item())
                                elif hasattr(val, '__len__') and len(val) > 0:
                                    # If it's an array, take the mean
                                    current_metric = float(jnp.mean(val))
                                else:
                                    current_metric = float(val)
                                break
                            except (ValueError, TypeError, AttributeError):
                                continue
                    
                    # Fallback to episode return if no goal achievement metrics available
                    if current_metric is None:
                        for metric_name in ['episode_return', 'episode_returns', 'mean_episode_return', 'returned_episode_returns']:
                            if metric_name in to_log and to_log[metric_name] is not None:
                                try:
                                    # Handle different metric formats (scalar, array, etc.)
                                    val = to_log[metric_name]
                                    if hasattr(val, 'item'):
                                        current_metric = float(val.item())
                                    elif hasattr(val, '__len__') and len(val) > 0:
                                        # If it's an array, take the mean
                                        current_metric = float(jnp.mean(val))
                                    else:
                                        current_metric = float(val)
                                    break
                                except (ValueError, TypeError, AttributeError):
                                    continue
                    
                    # Print and save only when we have a new best
                    # IMPORTANT: Compare against the Python-side tracker, NOT the JAX-side best_metric_val
                    # The JAX-side tracks returned_episode_returns but the Python callback extracts
                    # episode_return from to_log which may have different values due to averaging logic
                    # Using the Python tracker ensures consistent comparison logic
                    python_best_metric = best_policy_tracker['best_metric']
                    
                    # Debug logging to help diagnose best policy tracking
                    if int(update_step) % 10 == 0:  # Log every 10 steps
                        jax_best_float = float(best_metric_val) if hasattr(best_metric_val, 'item') else float(best_metric_val)
                        logger.debug("Best policy tracking at step %s: current_metric=%s, python_best=%s, jax_best=%s",
                                    update_step, current_metric, python_best_metric, jax_best_float)
                    
                    if current_metric is not None and current_metric > python_best_metric:
                        logger.info("🏆 New best policy found at update_step %s with metric %s (prev best=%s)", 
                                    update_step, current_metric, python_best_metric)
                        
                        # *** CRITICAL: Update the best_policy_tracker dict ***
                        # This is a side effect that mutates the Python dict
                        best_policy_tracker['best_metric'] = current_metric
                        best_policy_tracker['best_step'] = int(update_step)
                        best_policy_tracker['best_train_state'] = train_state
                        
                        # Save the best policy immediately
                        if config["USE_WANDB"]:
                            try:
                                orbax_checkpointer = PyTreeCheckpointer()
                                options = CheckpointManagerOptions(max_to_keep=1, create=True)
                                path = os.path.join(wandb.run.dir, "best_policies")
                                checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
                                save_args = orbax_utils.save_args_from_target(train_state)
                                checkpoint_manager.save(
                                    int(update_step),
                                    train_state,
                                    save_kwargs={"save_args": save_args},
                                )
                                logger.info("Best policy saved to %s at update_step %s", path, update_step)
                                
                                # Note: Video recording moved to end of training
                                # See record_final_best_episode() call after training completes
                                # This ensures only ONE video is recorded for the best policy
                                
                                # After saving best policy, upload the top-1 episode video (best episode)
                                try:
                                    record_dir = config.get("RECORD_DIR", None)
                                    if not record_dir:
                                        record_dir = getattr(wandb.run, "dir", None)

                                    def _find_best_episode_video(directory):
                                        """Find the .mp4 file with the largest achievement count in filename.

                                        Filenames created by EpisodeName look like:
                                            20251024T035203-ach3-len42.mp4
                                        We parse '-achN-' and pick the highest N. If no ach found,
                                        pick the most recent file by mtime.
                                        """
                                        if not directory:
                                            return None
                                        best_file = None
                                        best_ach = -1
                                        try:
                                            for root, _, files in os.walk(directory):
                                                for fname in files:
                                                    if not fname.lower().endswith('.mp4'):
                                                        continue
                                                    m = re.search(r'-ach(\d+)-', fname)
                                                    path = os.path.join(root, fname)
                                                    if m:
                                                        ach = int(m.group(1))
                                                        if ach > best_ach:
                                                            best_ach = ach
                                                            best_file = path
                                                    else:
                                                        # If no achievements in names, consider mtime as fallback
                                                        if best_file is None:
                                                            best_file = path
                                            return best_file
                                        except Exception as e:
                                            logger.error("Error while searching for best episode video: %s", e)
                                            return None

                                    # Use regex for filename parsing
                                    import re
                                    best_video = _find_best_episode_video(record_dir)
                                    if best_video:
                                        try:
                                            # Calculate total_timesteps for proper step alignment
                                            video_total_timesteps = (int(update_step) + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]
                                            
                                            # BACKWARD COMPATIBILITY: Use appropriate logging method
                                            use_legacy = config.get("_LEGACY_WANDB_LOGGING", False)
                                            if use_legacy:
                                                # Old approach: step parameter
                                                wandb.log({"video/best_episode": wandb.Video(best_video, caption=os.path.basename(best_video))},
                                                         step=video_total_timesteps, commit=False)
                                            else:
                                                # New approach: total_timesteps in dict
                                                wandb.log({"video/best_episode": wandb.Video(best_video, caption=os.path.basename(best_video)),
                                                          "total_timesteps": video_total_timesteps}, commit=False)
                                            logger.info("Uploaded best episode video %s to wandb/video/best_episode", best_video)
                                        except Exception as e:
                                            logger.warning("Failed to upload best episode video: %s", e)
                                        else:
                                            try:
                                                wandb.log({}, commit=True)
                                            except Exception:
                                                pass
                                except Exception as e:
                                    logger.error("Error in best-episode upload: %s", e)
                            except Exception as e:
                                logger.error("Failed to save best policy: %s", e)
                
                # Periodic checkpoint saving for resume capability
                checkpoint_freq = config.get("CHECKPOINT_FREQ", 0)
                num_updates = config.get("NUM_UPDATES", 0)
                is_final_update = (int(update_step) >= num_updates - 1) if num_updates > 0 else False
                
                # Count how many episodes completed in the last step of the rollout
                # If done[last_step, env] = True, that env will start a fresh episode next
                last_step_dones = traj_batch_for_episodes.done[-1, :]  # Shape: (NUM_ENVS,)
                num_fresh_starts = jnp.sum(last_step_dones).item()
                
                # Prefer checkpoints when many envs will start fresh, but ensure regular saves
                step_int = int(update_step)
                at_checkpoint_freq = checkpoint_freq > 0 and should_save_checkpoint(step_int, checkpoint_freq)
                
                # Save if: regular checkpoint time OR final update
                should_save = at_checkpoint_freq or is_final_update
                
                if should_save:
                    logger.info("💾 Saving checkpoint at step %d (is_final=%s, fresh_starts=%d/%d, checkpoint_dir=%s)", 
                                int(update_step), is_final_update, num_fresh_starts, config["NUM_ENVS"], get_checkpoint_dir(config))
                    try:
                        checkpoint_dir = get_checkpoint_dir(config)
                        # Convert best_metric_val and best_step_val to Python types
                        best_metric_float = float(best_metric_val) if hasattr(best_metric_val, 'item') else float(best_metric_val)
                        best_step_int = int(best_step_val) if hasattr(best_step_val, 'item') else int(best_step_val)
                        
                        save_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            update_step=int(update_step),
                            train_state=train_state,
                            ex_state=ex_state_for_logging,
                            rng=jax.random.PRNGKey(int(update_step)),  # Save a deterministic key based on step
                            config=config,
                            best_metric=best_metric_float,
                            best_step=best_step_int,
                            env_state=env_state_for_checkpoint,
                            obs=obs_for_checkpoint,
                        )
                        if is_final_update:
                            logger.info("Final checkpoint saved at update_step %s", update_step)
                        else:
                            logger.info("Periodic checkpoint saved at update_step %s", update_step)
                    except Exception as e:
                        logger.warning("Failed to save checkpoint at step %s: %s", update_step, e)

            jax.debug.callback(
                callback,
                metric,
                update_step,
                train_state,
                old_train_state_for_episodes,
                traj_batch_for_episodes,
                ex_state,
                best_metric,
                best_step,
                env_state,
                last_obs,
            )

            # Reset per-update intrinsic accumulators so they don't persist across updates
            try:
                accum = ex_state.get("intrinsic_components_accum", None)
                if isinstance(accum, dict):
                    zero_accum = {k: jnp.zeros_like(v) for k, v in accum.items()}
                    ex_state = dict(ex_state)
                    ex_state["intrinsic_components_accum"] = zero_accum
            except Exception:
                # Safety: If resetting accum fails, continue without mutating ex_state
                pass

            # Return runner_state with best policy tracking fields
            runner_state = (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step + 1,
                best_metric,
                best_step,
                best_train_state,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        # Include best policy tracking in runner_state for proper JAX tracing
        # Format: (train_state, env_state, obs, ex_state, rng, update_step, 
        #          best_metric, best_step, best_train_state)
        
        # Use resume values if available, otherwise start from scratch
        initial_update_step = resume_update_step if resume_data is not None else 0
        initial_best_metric = resume_best_metric if resume_data is not None else jnp.float32(-jnp.inf)
        initial_best_step = resume_best_step if resume_data is not None else jnp.int32(0)
        
        # Calculate remaining updates if resuming
        remaining_updates = max(0, config["NUM_UPDATES"] - initial_update_step)
        if resume_data is not None and initial_update_step > 0:
            logger.info(f"🔄 Resuming training: {initial_update_step} steps completed, {remaining_updates} steps remaining")
        
        runner_state = (
            train_state,
            env_state,
            obsv,
            ex_state,
            _rng,
            initial_update_step,
            initial_best_metric,  # best_metric
            initial_best_step,  # best_step
            train_state,  # best_train_state (initialize with current train_state)
        )
        
        # Only run remaining updates if resuming
        num_updates_to_run = remaining_updates if resume_data is not None else config["NUM_UPDATES"]
        
        if num_updates_to_run > 0:
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, num_updates_to_run
            )
        else:
            logger.info("Training already complete (no remaining updates)")
            # Create dummy metric for compatibility
            metric = {}
        
        # Extract best policy tracking from final runner_state
        (final_train_state, final_env_state, final_obs, final_ex_state, final_rng, 
         final_update_step, best_metric_unused, best_step_unused, best_train_state_unused) = runner_state
        
        # Reconstruct runner_state without best policy tracking for backward compatibility
        runner_state_compat = (final_train_state, final_env_state, final_obs, 
                               final_ex_state, final_rng, final_update_step)
        
        # Use the Python dict values that were updated by the callback
        # The callback successfully tracks best policy and updates best_policy_tracker
        best_policy_tracker_out = {
            'best_metric': best_policy_tracker['best_metric'],  # From callback
            'best_step': best_policy_tracker['best_step'],  # From callback
            'best_train_state': best_policy_tracker['best_train_state'],  # From callback
            'best_return': best_policy_tracker['best_return'],
            'save_best': best_policy_tracker['save_best']
        }
        
        return {"runner_state": runner_state_compat, "best_policy_tracker": best_policy_tracker_out}
    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    # Handle resume from checkpoint BEFORE wandb.init
    # This allows us to resume the correct wandb run
    resume_data = None
    resume_wandb_run_id = None
    resume_checkpoint_path = None
    
    if config.get("RESUME", False):
        checkpoint_path = config.get("CHECKPOINT_PATH")
        
        if checkpoint_path is None:
            # Try to auto-detect from wandb directory
            # Look for the most recent run with checkpoints
            wandb_result = find_latest_wandb_run_with_checkpoints(
                wandb_dir="wandb",
                project=config.get("WANDB_PROJECT")
            )
            if wandb_result:
                resume_wandb_run_id, checkpoint_path, run_dir = wandb_result
                logger.info(f"🔍 Auto-detected wandb run to resume: {resume_wandb_run_id}")
                logger.info(f"   Checkpoint directory: {checkpoint_path}")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            latest_step = find_latest_checkpoint(checkpoint_path)
            if latest_step is not None:
                logger.info(f"🔄 Resuming from checkpoint at step {latest_step} in {checkpoint_path}")
                try:
                    resume_data, saved_config = load_checkpoint(checkpoint_path, latest_step)
                    resume_checkpoint_path = checkpoint_path
                    
                    # Log what was restored
                    logger.info(f"  - Restored update_step: {resume_data.get('update_step', 0)}")
                    logger.info(f"  - Restored best_metric: {resume_data.get('best_metric', -float('inf'))}")
                    logger.info(f"  - Restored best_step: {resume_data.get('best_step', 0)}")
                    
                    # Get wandb run ID from saved config if not already found
                    if resume_wandb_run_id is None and saved_config:
                        resume_wandb_run_id = saved_config.get("_wandb_run_id")
                        if resume_wandb_run_id:
                            logger.info(f"  - Found wandb run ID in checkpoint: {resume_wandb_run_id}")
                    
                    # Validate config compatibility
                    if saved_config:
                        for key in ['ENV_NAME', 'NUM_ENVS', 'LAYER_SIZE', 'NUM_STEPS']:
                            if key in saved_config and key in config:
                                if saved_config[key] != config[key]:
                                    logger.warning(f"  ⚠️  Config mismatch for {key}: saved={saved_config[key]}, current={config[key]}")
                except Exception as e:
                    logger.error(f"Failed to load checkpoint: {e}")
                    logger.warning("Starting fresh training instead of resuming")
                    resume_data = None
            else:
                logger.warning(f"No checkpoints found in {checkpoint_path}, starting fresh training")
        else:
            if checkpoint_path:
                logger.warning(f"Checkpoint path not found: {checkpoint_path}")
            logger.warning("No checkpoints found for resume. Starting fresh training.")

    if config["USE_WANDB"]:
        algorithm_name = get_algorithm_name(config)
        # Allow user override of run name via RUN_NAME
        run_name = compute_run_name(config, algorithm_name)
        # Create a serializable config for wandb by excluding non-serializable objects
        wandb_config = {
            k: v for k, v in config.items()
            if k not in ["VAE_PARAMS", "VAE_APPLY_FN", "SCIRE_MODEL", "SCIRE_PARAMS", "SCIRE_ACTION_SEQUENCES", "INTRINSIC_REWARD_NORMALIZER"]
            and not callable(v)
        }
        
        # If resuming, use the existing wandb run ID
        if resume_wandb_run_id and resume_data:
            logger.info("🔗 Resuming wandb run: %s", resume_wandb_run_id)
            
            # Use id + resume="must" to resume the specific run
            # The explicit step parameter in wandb.log() ensures continuity
            wandb.init(
                project=config["WANDB_PROJECT"],
                entity=config["WANDB_ENTITY"],
                config=wandb_config,
                id=resume_wandb_run_id,
                resume="must",
            )
            logger.info("✅ Resuming WandB run - metrics will continue from checkpoint step")
        else:
            # Fresh run
            wandb.init(
                project=config["WANDB_PROJECT"],
                entity=config["WANDB_ENTITY"],
                config=wandb_config,
                name=run_name,
            )
            logger.info("✅ Fresh run started")
        
        # Define custom x-axis metric for WandB plots
        # This allows setting x-axis to "total_timesteps" in the UI
        wandb.define_metric("total_timesteps")
        wandb.define_metric("*", step_metric="total_timesteps")
        
        # Ensure recorder writes into the wandb run directory so videos and metrics
        # created by Recorders end up inside the run folder
        try:
            run_dir = getattr(wandb.run, "dir", None)
            if run_dir:
                config["RECORD_DIR"] = run_dir
        except Exception as e:
            logger.warning("Warning: unable to set RECORD_DIR to wandb.run.dir: %s", e)

        # Helper: find any .mp4 files under the run dir and upload them to wandb
        def _upload_videos_to_wandb(run_dir_path):
            if not run_dir_path:
                return
            try:
                for root, _, files in os.walk(run_dir_path):
                    for fname in files:
                        if fname.lower().endswith('.mp4'):
                            path = os.path.join(root, fname)
                            key = f"video/{os.path.splitext(fname)[0]}"
                            try:
                                # commit=False so we can batch uploads with other logs if desired
                                wandb.log({key: wandb.Video(path, caption=fname)}, commit=False)
                                logger.info("Uploaded video %s to wandb as %s", path, key)
                            except Exception as e:
                                logger.error("Failed to upload video %s to wandb: %s", path, e)
            except Exception as e:
                logger.error("Error while scanning run dir for videos: %s", e)
    
    # Ensure RECORD_DIR is set even when USE_WANDB=False (for HPO, testing, etc.)
    if not config.get("RECORD_DIR"):
        config["RECORD_DIR"] = generate_recording_dir(config)
    
    # Always create RECORD_DIR to ensure metrics can be written
    # This is required for HPO and evolutionary optimization even when DEBUG=False
    record_dir = config.get("RECORD_DIR")
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        if config.get("DEBUG"):
            logger.debug("DEBUG mode: Metrics will be written to %s/metrics.jsonl", record_dir)

    # Store resume_data in config so make_train can access it
    config["_RESUME_DATA"] = resume_data

    # Log checkpoint configuration
    checkpoint_freq = config.get("CHECKPOINT_FREQ", 0)
    checkpoint_dir = get_checkpoint_dir(config) if checkpoint_freq > 0 or config.get("SAVE_BEST_POLICY") else None
    
    # Calculate NUM_UPDATES here (same formula as in make_train) for logging purposes
    # This is needed because make_train hasn't been called yet
    total_timesteps = int(float(config.get("TOTAL_TIMESTEPS", 0)))
    num_steps = int(config.get("NUM_STEPS", 64))
    num_envs = int(config.get("NUM_ENVS", 1024))
    num_updates = total_timesteps // num_steps // num_envs if num_steps > 0 and num_envs > 0 else 0
    
    if checkpoint_freq > 0:
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info("💾 Checkpointing enabled:")
        logger.info("   - Checkpoint directory: %s", checkpoint_dir)
        logger.info("   - Checkpoint frequency: every %d update steps", checkpoint_freq)
        logger.info("   - Total updates: %d", num_updates)
        expected_checkpoints = (num_updates // checkpoint_freq) + 1  # +1 for final checkpoint
        logger.info("   - Expected checkpoints: ~%d", expected_checkpoints)
    else:
        logger.info("💾 Periodic checkpointing disabled (checkpoint_freq=0). Final checkpoint will still be saved.")

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    # Expose a module-global VAE_APPLY callable (prefer jitted variant) so compiled
    # inner loops can use the function without JAX attempting to treat it as
    # an abstract array (pjit functions are not interp-compatible as pytrees).
    # This is populated from config keys and remains static for the run.
    # Make a global reference to the RAW, non-pjit apply function so the
    # compiled inner loops can call it safely. Also keep the jitted variant
    # available under VAE_APPLY_PJIT_GLOBAL for non-traced python paths.
    global VAE_APPLY_FN_RAW_GLOBAL, VAE_APPLY_PJIT_GLOBAL
    VAE_APPLY_FN_RAW_GLOBAL = config.get("VAE_APPLY_FN")
    VAE_APPLY_PJIT_GLOBAL = config.get("VAE_APPLY_FN_JIT")

    # For NUM_REPEATS=1 with best policy tracking, we need to avoid JIT
    # so that callback side effects (dict mutations) are visible after training
    if config["NUM_REPEATS"] == 1 and config.get("SAVE_BEST_POLICY", False):
        logger.info("📝 Note: Running without outer JIT to enable best policy tracking via callbacks")
        # Create tracker dict that callbacks will mutate
        best_policy_tracker_container = [{
            'best_return': -float('inf'),
            'best_metric': -float('inf'),
            'best_step': 0,
            'best_train_state': None,
            'save_best': True
        }]
        # Create train function WITHOUT jax.jit on the outer function
        # (inner functions are still JIT'd for performance)
        train_fn = make_train(config, best_policy_tracker_container)
    else:
        # Multiple repeats or no tracking: use JIT as normal
        best_policy_tracker_container = None
        train_fn = make_train(config, None)
        train_fn = jax.jit(train_fn)
    
    t0 = time.time()
    try:
        if config["NUM_REPEATS"] > 1:
            train_vmap = jax.vmap(train_fn)
            out = train_vmap(rngs)
        else:
            out = train_fn(rngs[0])
    except Exception as e:
        # Provide clearer guidance for out-of-memory errors commonly seen with
        # JAX/XLA (RESOURCE_EXHAUSTED). Re-raise after logging additional hints.
        msg = str(e)
        if 'RESOURCE_EXHAUSTED' in msg or 'Out of memory' in msg or 'out of memory' in msg:
            logger.error("JAX ran out of GPU memory during compilation/execution: %s", msg)
            logger.error("Suggestion: reduce `NUM_ENVS` (e.g., from %s to 32 or 16), lower `XLA_PYTHON_CLIENT_MEM_FRACTION`, or run fewer concurrent workers.", config.get('NUM_ENVS'))
        raise
    t1 = time.time()
    logger.info("Time to run experiment: %s", t1 - t0)
    logger.info("SPS: %s", config["TOTAL_TIMESTEPS"] / (t1 - t0))
    
    # Run deterministic evaluation after training to get a reproducible final mean reward
    # This matches the evaluation methodology used in ppo_pytorch_hpo_minigrid.py
    # NOTE: For RandKey environments, post-training evaluation generates NEW random layouts
    # that the agent never trained on, so evaluation will fail. Use rollout metrics instead.
    eval_mean_reward = None
    try:
        n_eval_episodes = config.get("NUM_EVAL_EPISODES", 10)
        # Skip evaluation for RandKey environments - they generate new layouts on each reset
        skip_eval = "RandKey" in config.get("ENV_NAME", "")
        if skip_eval:
            logger.info("Skipping post-training evaluation for RandKey environment (generates unseen layouts)")
            logger.info("Using rollout/ep_rew_mean from training instead")
            n_eval_episodes = 0
        
        if n_eval_episodes > 0:
            logger.info("Running %d evaluation episodes for final metric...", n_eval_episodes)
            
            # Extract final train_state from output
            if config["NUM_REPEATS"] > 1:
                final_train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])
            else:
                final_train_state = out["runner_state"][0]
            
            # Create evaluation environment (same as training env)
            # IMPORTANT: Use the same seed as training so the environment layout matches
            # The agent's policy is trained on a specific environment configuration
            from envs.env_factory import make_env_from_name
            env_result = make_env_from_name(config["ENV_NAME"], auto_reset=True, base_seed=config.get("SEED"))
            if isinstance(env_result, tuple):
                # Craftax-style: (env, env_params)
                eval_env, eval_env_params = env_result
            else:
                # MiniGrid-style: just env
                eval_env = env_result
                eval_env_params = eval_env.default_params
            
            # Create network (same architecture as training)
            use_dual_value = should_use_dual_value_heads(config)
            if "Symbolic" in config["ENV_NAME"] or "MiniGrid" in config["ENV_NAME"]:
                if use_dual_value:
                    from models.actor_critic import ActorCriticDualValue
                    eval_network = ActorCriticDualValue(eval_env.action_space(eval_env_params).n, config["LAYER_SIZE"])
                else:
                    eval_network = ActorCritic(eval_env.action_space(eval_env_params).n, config["LAYER_SIZE"])
            else:
                if use_dual_value:
                    from models.actor_critic import ActorCriticConvDualValue
                    eval_network = ActorCriticConvDualValue(
                        eval_env.action_space(eval_env_params).n, config["LAYER_SIZE"]
                    )
                else:
                    eval_network = ActorCriticConv(
                        eval_env.action_space(eval_env_params).n, config["LAYER_SIZE"]
                    )
            
            # Run evaluation episodes with deterministic RNG
            # IMPORTANT: For environments with random layouts (like MiniGrid-*-RandKey-*),
            # we must use the SAME RNG seed as training so the agent is evaluated on
            # the same environment configurations it was trained on.
            # Using a different seed would generate completely new layouts the agent has never seen.
            eval_seed = config.get("SEED", 0)  # Use same seed as training
            eval_rng = jax.random.PRNGKey(eval_seed)
            
            eval_returns = []
            eval_episode_lengths = []
            eval_done_action_counts = []
            
            for ep_idx in range(n_eval_episodes):
                eval_rng, ep_rng = jax.random.split(eval_rng)
                
                # Reset environment
                obs, eval_state = eval_env.reset(ep_rng, eval_env_params)
                done = False
                episode_return = 0.0
                episode_length = 0
                done_action_count = 0
                
                max_steps = 1000  # Safety limit
                for step in range(max_steps):
                    # Get action from policy (use sampling to match training behavior)
                    # Deterministic argmax doesn't work well with DONE action in MiniGrid
                    # Add batch dimension for network: [obs_dim] -> [1, obs_dim]
                    obs_batched = jnp.expand_dims(obs, axis=0)
                    network_output = eval_network.apply(final_train_state.params, obs_batched)
                    
                    # Networks always return tuples: (pi, value) or (pi, value_ext, value_int)
                    # Extract pi (policy distribution) - always the first element
                    pi = network_output[0]
                    
                    # Sample action from policy distribution (same as training)
                    # This ensures the agent can take DONE action when at goals
                    eval_rng, action_rng = jax.random.split(eval_rng)
                    action_array = pi.sample(seed=action_rng)  # Returns [batch_size] shape
                    action = int(action_array[0])  # Extract first element and convert to scalar
                    
                    # Count DONE actions (action 6 in MiniGrid)
                    if action == 6:
                        done_action_count += 1
                    
                    # Step environment
                    eval_rng, step_rng = jax.random.split(eval_rng)
                    obs, eval_state, reward, done, info = eval_env.step(
                        step_rng, eval_state, action, eval_env_params
                    )
                    episode_return += float(reward)  # Convert JAX array to Python float
                    episode_length += 1
                    
                    if bool(done):  # Convert JAX bool to Python bool
                        break
                
                eval_returns.append(float(episode_return))
                eval_episode_lengths.append(episode_length)
                eval_done_action_counts.append(done_action_count)
                
                # Log detailed info for first few episodes (use WARNING to ensure visibility)
                if ep_idx < 3:
                    logger.warning("  Eval episode %d: return=%.3f, length=%d, done_actions=%d", 
                                  ep_idx, episode_return, episode_length, done_action_count)
            
            eval_mean_reward = float(np.mean(eval_returns))
            eval_std_reward = float(np.std(eval_returns))
            avg_length = float(np.mean(eval_episode_lengths))
            avg_done_actions = float(np.mean(eval_done_action_counts))
            logger.warning("Evaluation complete: mean_reward=%.6f ± %.6f over %d episodes",
                          eval_mean_reward, eval_std_reward, n_eval_episodes)
            logger.warning("  Average episode length: %.1f, Average DONE actions: %.1f", avg_length, avg_done_actions)
    except Exception as e:
        logger.warning("Failed to run evaluation episodes: %s", e)
        import traceback
        traceback.print_exc()
    
    # Record final best episode video if enabled
    if config.get("RECORD_EPISODES", False):
        try:
            from tracking.episode_video_integration import record_final_best_episode
            from envs.env_factory import make_env_from_name
            
            # Extract final runner_state from output
            if config["NUM_REPEATS"] > 1:
                runner_state = jax.tree.map(lambda x: x[0], out["runner_state"])
            else:
                runner_state = out["runner_state"]
            
            # Use the best_policy_tracker dict that was mutated by callbacks during training
            # This only works when NUM_REPEATS=1 and best_policy_tracker_container was created above
            if config["NUM_REPEATS"] == 1 and best_policy_tracker_container is not None:
                tracker = best_policy_tracker_container[0]
                best_metric_float = tracker.get('best_metric', -float('inf'))
                best_step_int = tracker.get('best_step', 0)
            else:
                # Fallback: no tracker available
                tracker = None
                best_metric_float = -float('inf')
                best_step_int = 0
            
            # Get the ACTUAL best train state from the tracker
            # Don't use the final train state - use the one that had the best metric!
            # Check if we found a valid best policy (metric > -inf)
            if (tracker is not None and 
                'best_train_state' in tracker and 
                tracker['best_train_state'] is not None and
                best_metric_float > -float('inf')):
                train_state = tracker['best_train_state']
                logger.info("✅ Using best policy from update_step %s with metric %s", best_step_int, best_metric_float)
            else:
                # Fallback to final train state if best wasn't tracked
                logger.warning("⚠️  Warning: No best policy found in tracker, using final policy")
                train_state = runner_state[0]  # First element of runner_state tuple
                best_step_int = config["NUM_UPDATES"]
                best_metric_float = 0.0
            
            # Re-create environment for recording (without wrappers, auto-reset enabled)
            result = make_env_from_name(config["ENV_NAME"], auto_reset=True, base_seed=config.get("SEED"))
            if isinstance(result, tuple):
                env, env_params = result
            else:
                env = result
                env_params = env.default_params
            
            # Re-create network (check environment type)
            # IMPORTANT: Use the same network type that was used during training
            use_dual_value = should_use_dual_value_heads(config)
            
            if "Symbolic" in config["ENV_NAME"] or "MiniGrid" in config["ENV_NAME"]:
                if use_dual_value:
                    from models.actor_critic import ActorCriticDualValue
                    network = ActorCriticDualValue(
                        action_dim=env.action_space(env_params).n,
                        layer_width=config.get("LAYER_SIZE", 512),
                        activation=config.get("ACTIVATION", "tanh")
                    )
                else:
                    network = ActorCritic(
                        action_dim=env.action_space(env_params).n, 
                        layer_width=config.get("LAYER_SIZE", 512),
                        activation=config.get("ACTIVATION", "tanh")
                    )
            else:
                if use_dual_value:
                    from models.actor_critic import ActorCriticConvDualValue
                    network = ActorCriticConvDualValue(
                        action_dim=env.action_space(env_params).n, 
                        layer_width=config.get("LAYER_SIZE", 512)
                    )
                else:
                    network = ActorCriticConv(
                        action_dim=env.action_space(env_params).n, 
                        layer_width=config.get("LAYER_SIZE", 512)
                    )
            
            # Record the final best episode
            record_final_best_episode(
                config=config,
                env=env,
                env_params=env_params,
                train_state=train_state,
                network=network,
                best_policy_tracker=tracker if tracker is not None else {},
                recorder=None
            )
        except Exception as e:
            logger.error("⚠️  Failed to record final best episode: %s", e)
            import traceback
            traceback.print_exc()

    # Report metrics saved during training
    # Always compute final metrics (regardless of DEBUG or WANDB settings)
    # This is needed for HPO and evolutionary optimization
    wandb_metrics_path = os.path.join(config.get("RECORD_DIR", generate_recording_dir(config)), 'metrics.jsonl')
    
    if config.get("DEBUG") and config.get("USE_WANDB"):
        import json
        import csv
        
        # Count the number of metric entries written
        metric_entries = 0
        if os.path.exists(wandb_metrics_path):
            with open(wandb_metrics_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        metric_entries += 1
        
        logger.info("Wrote %s metric entries to %s", metric_entries, wandb_metrics_path)
        
        # Also create a CSV version in the logdir directory with all metrics
        logdir_path = generate_recording_dir(config)
        csv_path = os.path.join(logdir_path, 'progress.csv')
        
        if os.path.exists(wandb_metrics_path) and metric_entries > 0:
            try:
                os.makedirs(logdir_path, exist_ok=True)
                
                # Read all metrics from wandb jsonl file
                all_metrics = []
                with open(wandb_metrics_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_metrics.append(json.loads(line.strip()))
                
                # Get all unique keys from all metric entries
                all_keys = set()
                for metrics in all_metrics:
                    all_keys.update(metrics.keys())
                
                # Sort keys for consistent column ordering (put timestep columns first)
                priority_keys = ['update_step', 'total_timesteps']
                other_keys = sorted([k for k in all_keys if k not in priority_keys])
                fieldnames = [k for k in priority_keys if k in all_keys] + other_keys
                
                # Write to CSV
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for metrics in all_metrics:
                        writer.writerow(metrics)
                
                logger.info("Wrote CSV metrics to %s", csv_path)
            except Exception as e:
                logger.warning("Warning: Failed to create CSV metrics file: %s", e)

    # Use evaluation mean reward if available (from deterministic evaluation episodes)
    # This provides a reproducible metric consistent with PyTorch HPO methodology
    # Fallback to training metrics only if evaluation was not performed
    # NOTE: "mean_reward" here means MEAN EPISODE RETURN across all episodes (including failures)
    # This is NOT the best episode return - it's the average performance, which is the standard RL metric
    # NOTE: This block MUST run regardless of DEBUG/WANDB settings for HPO/evolutionary optimization
    import json  # Needed for metrics extraction
    final_mean_reward = eval_mean_reward
    
    if final_mean_reward is None:
        # Fallback: Compute from training metrics file if evaluation was not performed
        # For RandKey environments, use FINAL rollout/ep_rew_mean (sliding window over last 100 episodes)
        # This is the most reliable metric - it's already averaged over a sliding window
        try:
            metrics_path = wandb_metrics_path
            last_rollout_ep_rew_mean = None
            metric_source = None
            if os.path.exists(metrics_path):
                # For rollout/ep_rew_mean, use the FINAL value (already a sliding window average)
                # For other metrics, collect all values for averaging
                rollout_values = []
                other_reward_values = []
                
                with open(metrics_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                m = json.loads(line.strip())
                            except Exception:
                                continue
                            
                            # Track the LATEST rollout/ep_rew_mean (sliding window average)
                            if 'rollout/ep_rew_mean' in m and m['rollout/ep_rew_mean'] is not None:
                                try:
                                    v = float(m['rollout/ep_rew_mean'])
                                    if not (np.isnan(v) or np.isinf(v)):
                                        last_rollout_ep_rew_mean = v
                                        metric_source = 'rollout/ep_rew_mean'
                                except Exception:
                                    pass
                            
                            # Also collect other metrics as fallback
                            for key in ['mean_episode_return', 'mean_reward', 'episode_return', 'returned_episode_returns', 'episode_returns']:
                                if key in m and m[key] is not None:
                                    try:
                                        v = m[key]
                                        if hasattr(v, '__len__') and not isinstance(v, str):
                                            try:
                                                v_mean = float(np.mean(v))
                                                if not (np.isnan(v_mean) or np.isinf(v_mean)):
                                                    other_reward_values.append(v_mean)
                                            except Exception:
                                                pass
                                        else:
                                            val = float(v)
                                            if not (np.isnan(val) or np.isinf(val)):
                                                other_reward_values.append(val)
                                    except Exception:
                                        pass
                
            # Prefer the final rollout/ep_rew_mean (SB3-style sliding window)
            if last_rollout_ep_rew_mean is not None:
                final_mean_reward = last_rollout_ep_rew_mean
            elif other_reward_values:
                final_mean_reward = float(np.mean(other_reward_values))
                metric_source = 'training_metrics_average'
        except Exception:
            pass

        # If no reward values found in metrics.jsonl, try progress CSV (progress.csv)
        if final_mean_reward is None:
            try:
                import csv as _csv
                csv_candidates = []
                # Prefer the csv_path created above if available
                if 'csv_path' in locals() and os.path.exists(csv_path):
                    csv_candidates.append(csv_path)
                    # Also check RECORD_DIR (usually wandb.run.dir)
                    record_dir = config.get('RECORD_DIR')
                    if record_dir:
                        candidate = os.path.join(record_dir, 'progress.csv')
                        if os.path.exists(candidate):
                            csv_candidates.append(candidate)
                    # Fallback to generate_recording_dir
                    fallback_csv = os.path.join(generate_recording_dir(config), 'progress.csv')
                    if os.path.exists(fallback_csv) and fallback_csv not in csv_candidates:
                        csv_candidates.append(fallback_csv)

                    csv_vals = []
                    for cpath in csv_candidates:
                        try:
                            with open(cpath, 'r', encoding='utf-8') as cf:
                                reader = _csv.DictReader(cf)
                                for row in reader:
                                    for key in ['episode_return', 'mean_episode_return', 'episode_return_mean', 'mean_reward']:
                                        if key in row and row[key] not in (None, '', 'null'):
                                            try:
                                                v = float(row[key])
                                                if not (np.isnan(v) or np.isinf(v)):
                                                    csv_vals.append(v)
                                            except Exception:
                                                continue
                        except Exception:
                            continue

                if csv_vals:
                    final_mean_reward = float(np.mean(csv_vals))
            except Exception:
                pass

        # As a last resort, use best policy tracker metric if available
        if final_mean_reward is None:
            try:
                if 'best_policy_tracker_container' in locals() and best_policy_tracker_container is not None:
                    bt = best_policy_tracker_container[0]
                    best_metric = bt.get('best_metric', None)
                    if best_metric is not None and best_metric > -1e9:
                        final_mean_reward = float(best_metric)
            except Exception:
                pass

    if final_mean_reward is not None:
        # Log at WARNING and print to stdout so downstream evaluators capture it
        metric_source_str = f" (from {metric_source})" if 'metric_source' in locals() and metric_source else ""
        if eval_mean_reward is not None:
            metric_source_str = " (from post-training evaluation)"
        
        # Always report final rollout/ep_rew_mean as the primary metric
        # NOTE: This is the MEAN across all episodes (including failures), not the best episode return
        # This is the standard RL metric for agent performance (like SB3's rollout/ep_rew_mean)
        reported_reward = final_mean_reward
        
        # Log the primary metric with clarification
        logger.warning("Mean episode return: %.6f%s (averaged across last ~100 episodes)", reported_reward, metric_source_str)
        
        try:
            print(f"Mean episode return: {reported_reward:.6f}", flush=True)
        except Exception:
            pass
    else:
        logger.warning("Could not compute final mean episode return from evaluation or training metrics")
        reported_reward = None  # Ensure it's defined even if no metrics found

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            # Extract train_state from runner_state
            # When NUM_REPEATS > 1, we need to index with [rs_index][0]
            # When NUM_REPEATS = 1, we just need to index with [rs_index]
            if config["NUM_REPEATS"] > 1:
                train_states = out["runner_state"][rs_index]
                train_state = jax.tree.map(lambda x: x[0], train_states)
            else:
                # No batch dimension when NUM_REPEATS=1
                runner_state = out["runner_state"]
                train_state = runner_state[rs_index]
            
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            logger.info("saved runner state to %s", path)
            save_args = orbax_utils.save_args_from_target(train_state)
            # Ensure NUM_UPDATES is an integer and not a float (avoid orbax checkpoint naming floats)
            final_update_step = int(max(0, int(config.get("NUM_UPDATES", 0)) - 1))
            checkpoint_manager.save(
                final_update_step,
                train_state,
                save_kwargs={"save_args": save_args},
            )
            logger.info("Final policy saved at update_step %s", final_update_step)

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")

        if config.get("SAVE_BEST_POLICY", False):
            # Check if we have a best policy from training
            # Note: Due to JAX vmap/jit, the best_policy_tracker returned from training
            # may not have the updated values. However, the callback saves policies during training.
            # We check if best_policies directory exists instead.
            best_policies_path = os.path.join(wandb.run.dir, "best_policies") if config["USE_WANDB"] else None
            if best_policies_path and os.path.exists(best_policies_path) and os.listdir(best_policies_path):
                logger.info("Best policy was saved during training to %s", best_policies_path)
                logger.info("Best policy is available in the 'best_policies' directory")
            else:
                # No best policy was found during training, so don't save anything to best_policies
                logger.info("No improvement detected during training - no best policy to save")
                logger.info("Note: Use --save-policy to save the final policy to 'policies' directory")

        # Attempt to upload any recorded videos that were written into the run dir
        try:
            _upload_videos_to_wandb(getattr(wandb.run, "dir", None))
        except Exception as e:
            logger.warning("Warning: failed to upload videos to wandb: %s", e)
        else:
            try:
                # Force a commit so uploaded videos appear in the run even if no
                # other logs follow. This is a no-op if wandb is not connected.
                wandb.log({}, commit=True)
            except Exception:
                pass
    
    # Return training results and metrics for evolutionary/HPO usage
    return {
        'out': out,
        'eval_mean_reward': eval_mean_reward,
        'final_mean_reward': final_mean_reward if 'final_mean_reward' in locals() else None,
        'reported_reward': reported_reward if 'reported_reward' in locals() else None
    }

