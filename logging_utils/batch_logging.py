import time
from collections import deque

import jax.numpy as jnp
import numpy as np
import wandb

batch_logs = {}
log_times = []

# Rolling window buffers for SB3-style rollout metrics
# Track last stats_window_size episode returns and lengths
episode_returns_buffer = deque(maxlen=100)  # Default stats_window_size
episode_lengths_buffer = deque(maxlen=100)


def create_log_dict(info, config):
    # Check if any episodes completed in this batch
    completed_episodes = jnp.sum(info.get("returned_episode", 0))
    
    if completed_episodes > 0:
        # Calculate mean episode return and length only for completed episodes
        total_return = jnp.sum(info["returned_episode_returns"] * info.get("returned_episode", 1))
        total_length = jnp.sum(info["returned_episode_lengths"] * info.get("returned_episode", 1))
        mean_return = total_return / completed_episodes
        mean_length = total_length / completed_episodes
        
        # Add individual completed episodes to rolling window buffers
        # Extract the individual episode returns/lengths where episodes completed
        returned_episode_mask = info.get("returned_episode", 0)
        episode_returns_flat = info["returned_episode_returns"].flatten()
        episode_lengths_flat = info["returned_episode_lengths"].flatten()
        returned_episode_flat = returned_episode_mask.flatten()
        
        # Add each completed episode to the buffer
        for i in range(len(returned_episode_flat)):
            if returned_episode_flat[i] > 0:  # Episode completed
                episode_returns_buffer.append(float(episode_returns_flat[i]))
                episode_lengths_buffer.append(float(episode_lengths_flat[i]))
    else:
        # No episodes completed - try to report running episode stats instead of zeros
        # Prefer the per-env running episode returns/lengths if available in info
        if "episode_returns" in info:
            try:
                mean_return = float(jnp.mean(info["episode_returns"]))
            except Exception:
                # Fallback to 0.0 if the structure is unexpected
                mean_return = 0.0
        else:
            mean_return = 0.0

        if "episode_lengths" in info:
            try:
                mean_length = float(jnp.mean(info["episode_lengths"]))
            except Exception:
                mean_length = float(config.get("NUM_STEPS", 1000))
        else:
            # Episodes are running to max length since none completed early
            mean_length = float(config.get("NUM_STEPS", 1000))
    
    to_log = {
        "episode_return": mean_return,
        "episode_length": mean_length,
    }
    
    # Add SB3-style rollout metrics: rolling mean over last stats_window_size episodes
    if len(episode_returns_buffer) > 0:
        to_log["rollout/ep_rew_mean"] = np.mean(episode_returns_buffer)
        to_log["rollout/ep_len_mean"] = np.mean(episode_lengths_buffer)
        # Also log the number of episodes in the buffer for transparency
        to_log["rollout/ep_count"] = len(episode_returns_buffer)
    else:
        # No completed episodes yet - don't log rollout metrics
        # This is better than logging 0.0 which could be misleading
        pass

    sum_achievements = 0
    for k, v in info.items():
        if "achievements" in k.lower():
            to_log[k] = v
            sum_achievements += v / 100.0

    to_log["achievements"] = sum_achievements

    # Add MiniGrid-specific metrics
    if "MiniGrid" in config.get("ENV_NAME", ""):
        # Calculate achievement rates for MiniGrid
        if "returned_episode" in info:
            total_episodes = jnp.sum(info["returned_episode"])
            if total_episodes > 0:
                if "large_reward_goal" in info:
                    large_goal_rate = jnp.sum(info["large_reward_goal"] * info["returned_episode"]) / total_episodes
                    to_log["large_reward_goal_rate"] = large_goal_rate
                    # Add as achievement chart for wandb
                    to_log["Achievements/large_goal"] = large_goal_rate
                
                if "small_reward_goal" in info:
                    small_goal_rate = jnp.sum(info["small_reward_goal"] * info["returned_episode"]) / total_episodes  
                    to_log["small_reward_goal_rate"] = small_goal_rate
                    # Add as achievement chart for wandb
                    to_log["Achievements/small_goal"] = small_goal_rate
                
                # Overall achievement rate (either goal)
                if "large_reward_goal" in info and "small_reward_goal" in info:
                    any_goal_rate = jnp.sum((info["large_reward_goal"] + info["small_reward_goal"]) * info["returned_episode"]) / total_episodes
                    to_log["any_goal_achievement_rate"] = jnp.minimum(any_goal_rate, 1.0)  # Cap at 1.0
                    # Add as achievement chart for wandb
                    to_log["Achievements/any_goal"] = jnp.minimum(any_goal_rate, 1.0)
            else:
                # No episodes completed - provide default values
                to_log["large_reward_goal_rate"] = 0.0
                to_log["small_reward_goal_rate"] = 0.0
                to_log["any_goal_achievement_rate"] = 0.0
                to_log["Achievements/large_goal"] = 0.0
                to_log["Achievements/small_goal"] = 0.0
                to_log["Achievements/any_goal"] = 0.0

    if config.get("TRAIN_ICM") or config.get("USE_RND") or config.get("USE_JAX_INTRINSIC_REWARDS"):
        # Log intrinsic/extrinsic rewards for any intrinsic reward system
        if "reward_i" in info:
            to_log["intrinsic_reward"] = info["reward_i"]
        if "reward_e" in info:
            to_log["extrinsic_reward"] = info["reward_e"]

        if config.get("TRAIN_ICM"):
            to_log["icm_inverse_loss"] = info["icm_inverse_loss"]
            to_log["icm_forward_loss"] = info["icm_forward_loss"]
        elif config.get("USE_RND"):
            to_log["rnd_loss"] = info["rnd_loss"]

    return to_log


def batch_log(update_step, log, config, total_timesteps=None):
    update_step = int(update_step)
    if update_step not in batch_logs:
        batch_logs[update_step] = []

    batch_logs[update_step].append(log)

    if len(batch_logs[update_step]) == config["NUM_REPEATS"]:
        agg_logs = {}
        for key in batch_logs[update_step][0]:
            agg = []
            if key in ["goal_heatmap"]:
                agg = [batch_logs[update_step][0][key]]
            else:
                for i in range(config["NUM_REPEATS"]):
                    val = batch_logs[update_step][i][key]
                    # Handle both scalar and array values
                    if jnp.isscalar(val) or val.ndim == 0:
                        if not jnp.isnan(val):
                            agg.append(val)
                    else:
                        # For arrays, check if any element is not nan
                        if not jnp.isnan(val).all():
                            agg.append(val)

            if len(agg) > 0:
                if key in [
                    "episode_length",
                    "episode_return",
                    "exploration_bonus",
                    "e_mean",
                    "e_std",
                    "rnd_loss",
                ]:
                    agg_logs[key] = np.mean(agg)
                else:
                    agg_logs[key] = np.array(agg)
            else:
                # Skip logging this key if all values were NaN
                # This is especially important for episode_length and episode_return
                # when no episodes have completed yet
                pass

        log_times.append(time.time())

        if config["DEBUG"]:
            if len(log_times) == 1:
                print("Started logging")
            elif len(log_times) > 1:
                dt = log_times[-1] - log_times[-2]
                steps_between_updates = (
                    config["NUM_STEPS"] * config["NUM_ENVS"] * config["NUM_REPEATS"]
                )
                sps = steps_between_updates / dt
                agg_logs["sps"] = sps

        # Log to wandb if we have meaningful metrics AND wandb is enabled
        # Include total_timesteps so it can be used as x-axis in WandB UI
        if len(agg_logs) > 0 and config.get("USE_WANDB", False):
            if total_timesteps is not None:
                agg_logs["total_timesteps"] = int(total_timesteps)
            wandb.log(agg_logs)


