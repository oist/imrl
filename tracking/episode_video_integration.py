"""
Integration module for recording best episode videos during training
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import scire

logger = logging.getLogger(__name__)

def _extract_intrinsic_components_from_vae_result(vae_result):
    """Return a small dict mapping viewer keys to scalar values from a VAE return value.

    The VAE may return in several formats: (rewards, info_dict), (components_dict, normalizer),
    or a direct dict. We prefer info_dict if present. The function returns None if no
    usable component dict found.
    """
    vae_components = None
    if isinstance(vae_result, tuple) and len(vae_result) == 2:
        vae_output, vae_info = vae_result
        # Prefer explicitly returned 'info' dict when non-empty; otherwise fall back
        # to the primary VAE output if it contains component keys.
        if isinstance(vae_info, dict) and len(vae_info) > 0:
            vae_components = vae_info
        elif isinstance(vae_output, dict):
            vae_components = vae_output
    elif isinstance(vae_result, dict):
        vae_components = vae_result
    if not isinstance(vae_components, dict):
        return None

    # Map keys to those expected by viewer
    # Ensure keys are preserved consistently so the replay saver records both raw
    # and normalized versions when available. Previously we accidentally mapped
    # raw keys to normalized keys which caused viewer recomputation to miss the
    # raw values and display saturated normalized numbers.
    key_mapping = {
        'novelty_raw': 'novelty_raw',
        'novelty_normalized': 'novelty_normalized',
        'novelty_weighted': 'novelty_weighted',
        'surprise_raw': 'surprise_raw',
        'surprise_normalized': 'surprise_normalized',
        'surprise_weighted': 'surprise_weighted',
        'empowerment_raw': 'empowerment_raw',
        'empowerment_normalized': 'empowerment_normalized',
        'empowerment_weighted': 'empowerment_weighted',
        'intrinsic_reward_total': 'total',
    }

    result = {}
    for src_key, dst_key in key_mapping.items():
        if src_key in vae_components and dst_key not in result:
            val = vae_components[src_key]
            try:
                if hasattr(val, 'shape') and len(getattr(val, 'shape', ())) >= 1:
                    result[dst_key] = float(val[0])
                else:
                    result[dst_key] = float(val)
            except Exception:
                result[dst_key] = vae_components[src_key]
    return result
from tracking.episode_video_recorder import EpisodeVideoRecorder
from tracking.episode_replay_saver import EpisodeReplaySaver


def create_episode_recorder_if_enabled(config: Dict, env, env_params) -> Optional[EpisodeVideoRecorder]:
    """
       # Step 1: PARALLEL EVALUATION - evaluate all episodes simultaneously using vmap
    print(f"⚡ Running 10 episodes in parallel...")
    
    # Generate all random seeds - use uint32 to handle large seed values
    eval_seeds = jnp.array([seed + 100000 + i * 1000 for i in range(num_eval_episodes)], dtype=jnp.uint32)
    rng_keys = jax.vmap(jax.random.PRNGKey)(eval_seeds)e episode video recorder if video recording is enabled in config
    
    Args:
        config: Training configuration dictionary
        env: Environment instance
        env_params: Environment parameters
        
    Returns:
        EpisodeVideoRecorder instance or None
    """
    if not config.get("RECORD_EPISODES", False):
        return None
    
    # Only enable for Craftax Symbolic or MiniGrid environments (not raw pixels-only setups)
    env_name = config.get("ENV_NAME", "")
    is_craftax_symbolic = ("Symbolic" in env_name and "Craftax" in env_name)
    is_minigrid = ("MiniGrid" in env_name) or hasattr(env, 'get_obs') or hasattr(env, 'render')
    if not (is_craftax_symbolic or is_minigrid):
        print("⚠️  Episode video recording only supported for Craftax Symbolic or MiniGrid environments")
        return None
    
    # Save videos to a temp directory outside wandb to prevent auto-upload
    # We'll explicitly log only the final best video
    import tempfile
    output_dir = config.get("RECORD_DIR", None)
    if output_dir is None:
        # Create a temporary directory for episode videos
        output_dir = tempfile.mkdtemp(prefix="craftax_episodes_")
        print(f"📁 Episode videos will be saved to: {output_dir}")
    
    if hasattr(jax.random, 'PRNGKey'):  # Ensure we're in a JAX environment
        try:
            recorder = EpisodeVideoRecorder(
                env=env,
                env_params=env_params,
                env_name=env_name,
                output_dir=output_dir,
                frame_size=(1024, 1024),
                record_full_map=True,
                record_fov=True
            )
            print(f"✅ Episode video recorder initialized for {env_name}")
            return recorder
        except Exception as e:
            print(f"⚠️  Failed to initialize episode video recorder: {e}")
            return None
    return None


def record_best_episode(
    config: Dict,
    env,
    env_params,
    train_state,
    network,
    rng_key,
    update_step: int,
    recorder: Optional[EpisodeVideoRecorder] = None,
    replay_saver: Optional[EpisodeReplaySaver] = None,
    normalizer_state: Optional[object] = None,
    achievement_count: Optional[int] = None,
    episode_return: Optional[float] = None
) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
    """
    Record a single episode using the current policy
    
    Args:
        config: Training configuration
        env: Environment instance
        env_params: Environment parameters
        train_state: Current training state
        network: Policy network
        rng_key: JAX random key
        update_step: Current training step
        recorder: Optional pre-initialized recorder
        replay_saver: Optional pre-initialized replay saver
        achievement_count: Optional pre-computed achievement count (if None, will extract from env_state)
        episode_return: Optional pre-computed episode return (if None, will extract from env_state)
        
    Returns:
        Tuple of (video_path, replay_paths) where replay_paths is (json_path, pickle_path)
        Returns (None, None) if recording failed
    """
    if recorder is None:
        recorder = create_episode_recorder_if_enabled(config, env, env_params)
    # If we do not have a recorder and replay sequences are disabled, there's nothing to record
    if recorder is None and not config.get("SAVE_REPLAY_SEQUENCES", True):
        return None, None
    
    # Create replay saver if enabled
    if replay_saver is None and config.get("SAVE_REPLAY_SEQUENCES", True):
        # Save to same directory as videos
        replay_dir = recorder.output_dir if recorder else config.get("RECORD_DIR", "episode_replays")
        env_name = config.get("ENV_NAME", "")
        # CRITICAL FIX: Always save state trajectory for perfect replay reproduction
        # This ensures replay viewer shows exact same behavior as training videos
        save_states = True  # Force state history saving for accurate replay
        # Save intrinsic rewards if JAX intrinsic rewards are enabled
        save_intrinsic_rewards = config.get("USE_JAX_INTRINSIC_REWARDS", False)
        replay_saver = EpisodeReplaySaver(replay_dir, env_name, save_states=save_states, save_intrinsic_rewards=save_intrinsic_rewards)
    
    try:
        # Start recording if we have a recorder (video); otherwise we still proceed to record the replay saver
        if recorder is not None:
            recorder.start_recording()
        
        # CRITICAL: Save the original rng_key BEFORE any splits for exact replay
        # This ensures replay can follow the exact same RNG sequence
        original_rng_key = rng_key
        
        # Reset environment - convert to numpy for rendering
        rng_key, reset_key = jax.random.split(rng_key)
        reset_result = env.reset(reset_key, env_params)
        
        # Handle different return formats
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, env_state = reset_result
        else:
            raise ValueError(f"Unexpected reset result: {type(reset_result)}")
        
        done = False
        step_count = 0
        max_steps = config.get("EPISODE_MAX_STEPS", 1000)
        
        # Add initial frame - convert JAX arrays to numpy
        if recorder is not None:
            recorder.add_frame(env_state, obs)
        
        # Start replay recording if enabled - use ORIGINAL rng_key for exact replay
        if replay_saver is not None:
            replay_saver.start_recording(original_rng_key, env_state, obs)

        # Local normalizer snapshot used for evaluation-time normalization
        # We want to compute and save normalized intrinsic components during
        # evaluation/recording, matching the same normalization used in training
        # (Welford updates). Use a copy so we don't mutate the training normalizer.
        supplied_normalizer = normalizer_state or config.get('INTRINSIC_REWARD_NORMALIZER')
        if supplied_normalizer is None:
            local_normalizer = scire.NormalizerState.init()
        else:
            # Use as-is (NamedTuple) - we'll reassign to updated versions as we step
            local_normalizer = supplied_normalizer
        
        # Run episode
        prev_obs = obs
        while not done and step_count < max_steps:
            # Get action from policy
            rng_key, action_key = jax.random.split(rng_key)
            
            # Handle observation shape
            if isinstance(obs, tuple):
                actual_obs = obs[1] if len(obs) > 1 else obs[0]
            else:
                actual_obs = obs
            
            # Get policy distribution and sample action
            params = train_state.params if hasattr(train_state, 'params') else train_state.get('params')
            network_output = network.apply(params, actual_obs[None, ...])  # Add batch dim
            
            # Handle both single and dual value head networks
            if len(network_output) == 3:
                # Dual value heads: (pi, value_ext, value_int)
                pi, value_ext, value_int = network_output
                value = value_ext + value_int  # Combined value
            else:
                # Single value head: (pi, value)
                pi, value = network_output
            
            action = pi.sample(seed=action_key)
            action = action[0]  # Remove batch dim
            
            # Prepare intrinsic_components variable; we compute it AFTER stepping (so we have next_obs)
            intrinsic_components = None
            
            # Step environment
            rng_key, step_key = jax.random.split(rng_key)
            step_result = env.step(step_key, env_state, action, env_params)
            
            # Handle different return formats
            if isinstance(step_result, tuple):
                if len(step_result) == 5:
                    obs, env_state, reward, done, info = step_result
                else:
                    raise ValueError(f"Unexpected step result length: {len(step_result)}")
            else:
                raise ValueError(f"Unexpected step result type: {type(step_result)}")
            
            # Check for early termination BEFORE adding frame
            # Handle both scalar and array done values
            if hasattr(done, 'shape') and len(done.shape) > 0:
                # Array of done values (e.g., from vectorized envs)
                done = bool(jnp.any(done))
            elif hasattr(done, 'item'):
                done = bool(done.item())
            else:
                done = bool(done)
            
            # Only add frame if episode is not done
            # This prevents recording the post-termination/reset state
            if not done and recorder is not None:
                recorder.add_frame(env_state, obs)
            
            # Record action and reward for replay; include observation so we can compute seen-object counts
            if replay_saver is not None:
                # Compute intrinsic components using prev_obs -> obs transition and record them
                if config.get("USE_JAX_INTRINSIC_REWARDS", False) and replay_saver.save_intrinsic_rewards:
                    intrinsic_components = None
                    try:
                        obs_prev_flat = jnp.reshape(jnp.asarray(prev_obs).reshape(-1), (1, config["OBS_DIM"]))
                        obs_next_flat = jnp.reshape(jnp.asarray(obs).reshape(-1), (1, config["OBS_DIM"]))
                        action_idx = int(action.item()) if hasattr(action, 'item') else int(action)
                        action_batch = jnp.array([action_idx])

                        if config.get("USE_EMI", False):
                            # EMI is independent from the SCIRE normalizer; its own reward/component
                            # names are returned as-is.
                            action_onehot = jax.nn.one_hot(action_batch, config["ACTION_DIM"])
                            vae_apply = config.get("VAE_APPLY_FN") or config.get("VAE_APPLY_FN_JIT")
                            _, emi_info = vae_apply(
                                config["VAE_PARAMS"],
                                obs_prev_flat,
                                action_onehot,
                                obs_next_flat,
                                config["NOVELTY_WEIGHT"],
                                config["SURPRISE_WEIGHT"],
                                empowerment_weight=config["EMPOWERMENT_WEIGHT"],
                                return_components=True,
                            )
                            intrinsic_components = dict(emi_info)
                        else:
                            rng_key, scire_eval_key = jax.random.split(rng_key)
                            _, scire_info = scire.compute_intrinsic_reward(
                                config["SCIRE_MODEL"],
                                config["SCIRE_PARAMS"],
                                obs_prev_flat,
                                action_batch,
                                obs_next_flat,
                                config["SCIRE_ACTION_SEQUENCES"],
                                scire_eval_key,
                                novelty_weight=config["NOVELTY_WEIGHT"],
                                surprise_weight=config["SURPRISE_WEIGHT"],
                                empowerment_weight=config["EMPOWERMENT_WEIGHT"],
                                normalizer_state=local_normalizer,
                                update_normalizer=True,
                            )
                            local_normalizer = scire_info["normalizer_state"]
                            intrinsic_components = {
                                "novelty_raw": float(scire_info["novelty_raw"][0]),
                                "novelty_normalized": float(scire_info["novelty_normalized"][0]),
                                "novelty_weighted": float(config["NOVELTY_WEIGHT"] * scire_info["novelty_normalized"][0]),
                                "surprise_raw": float(scire_info["surprise_raw"][0]),
                                "surprise_normalized": float(scire_info["surprise_normalized"][0]),
                                "surprise_weighted": float(config["SURPRISE_WEIGHT"] * scire_info["surprise_normalized"][0]),
                                "empowerment_raw": float(scire_info["empowerment_raw"][0]),
                                "empowerment_normalized": float(scire_info["empowerment_normalized"][0]),
                                "empowerment_weighted": float(config["EMPOWERMENT_WEIGHT"] * scire_info["empowerment_normalized"][0]),
                            }
                    except Exception as e:
                        logging.warning("Warning: Failed to compute intrinsic rewards at step %s: %s", step_count, e)
                        intrinsic_components = None
                # Only record step if episode is not done
                # This prevents recording the post-termination/reset state which may have
                # a different landscape/player position due to environment auto-reset
                if not done:
                    replay_saver.add_step(action, reward, env_state, done, intrinsic_components, observation=obs)
                else:
                    # For the final step (done=True), only record the action/reward/done flag
                    # but NOT the env_state (which would be post-reset)
                    # This ensures state_history doesn't include the reset state
                    replay_saver.add_step(action, reward, None, done, intrinsic_components, observation=obs)
            
            prev_obs = obs
            step_count += 1
        
        # Stop recording (guard for recorder None)
        if recorder is not None:
            recorder.stop_recording()
        if replay_saver is not None:
            replay_saver.stop_recording()
        
        # Save video - extract info from potentially wrapped state
        # Use provided values if available, otherwise extract from env_state
        if achievement_count is None or episode_return is None:
            extracted_ach = 0
            extracted_ret = 0.0
            
            # Unwrap state if needed (e.g., LogEnvState)
            actual_state = env_state.env_state if hasattr(env_state, 'env_state') else env_state
            
            if hasattr(actual_state, 'achievements'):
                achievements = np.array(actual_state.achievements)
                extracted_ach = int(np.sum(achievements))
            
            # Try to get episode return from wrapper first, then from state
            if hasattr(env_state, 'episode_returns'):
                returns = np.array(env_state.episode_returns)
                # Handle both scalar and array returns
                extracted_ret = float(returns.item() if returns.ndim == 0 else returns[0] if returns.ndim == 1 and len(returns) > 0 else 0.0)
            elif hasattr(actual_state, 'episode_return'):
                ret = np.array(actual_state.episode_return)
                extracted_ret = float(ret.item() if ret.ndim == 0 else ret[0] if ret.ndim == 1 and len(ret) > 0 else 0.0)
            
            # Use extracted values if provided values are None
            if achievement_count is None:
                achievement_count = extracted_ach
            if episode_return is None:
                episode_return = extracted_ret
        
        filename = f"step{update_step:06d}_ach{achievement_count}_ret{episode_return:.1f}"
        video_path = None
        if recorder is not None:
            video_path = recorder.save_video(filename, fps=10)
        
        # Save replay data if enabled
        replay_paths = None
        if replay_saver is not None:
            try:
                # Supply a snapshot of the current normalizer if provided, else use config's normalizer
                supplied_normalizer = normalizer_state or config.get('INTRINSIC_REWARD_NORMALIZER')
                replay_paths = replay_saver.save_replay(filename, update_step, normalizer_state=supplied_normalizer)
                print(f"   Achievements: {achievement_count}, Return: {episode_return:.1f}")
            except Exception as e:
                print(f"⚠️  Failed to save replay data: {e}")
        
        # Don't log here - let the caller decide whether/how to log
        # This avoids duplicate logging when called from record_final_best_episode
        
        return video_path, replay_paths
        
    except Exception as e:
        print(f"⚠️  Failed to record episode: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def record_and_log_best_episode_on_improvement(
    config: Dict,
    env,
    env_params,
    train_state,
    network,
    update_step: int,
    best_policy_tracker: Dict,
    recorder: Optional[EpisodeVideoRecorder] = None,
    replay_saver: Optional[EpisodeReplaySaver] = None
) -> None:
    """
    Record and log best episode video when a new best policy is found
    
    This should be called right after a new best policy is detected.
    
    Args:
        config: Training configuration
        env: Environment instance  
        env_params: Environment parameters
        train_state: Current training state (should be the best one)
        network: Policy network
        update_step: Current training step
        best_policy_tracker: Best policy tracking dictionary
        recorder: Optional pre-initialized recorder
        replay_saver: Optional pre-initialized replay saver
    """
    if not config.get("RECORD_EPISODES", False):
        return
    
    print(f"🎥 Recording best episode at update_step {update_step}...")
    
    # Generate a random key for episode rollout
    seed = config.get("SEED", 42)
    rng_key = jax.random.PRNGKey(seed + update_step)
    
    # Record episode
    video_path, replay_paths = record_best_episode(
        config=config,
        env=env,
        env_params=env_params,
        train_state=train_state,
        network=network,
        rng_key=rng_key,
        update_step=update_step,
        recorder=recorder,
        replay_saver=replay_saver
        , normalizer_state=(train_state.get('intrinsic_reward_normalizer') if isinstance(train_state, dict) else getattr(train_state, 'intrinsic_reward_normalizer', None))
    )
    
    if video_path:
        print(f"✅ Best episode video saved: {video_path}")
        
        # Log to wandb with special key for best episode
        if config.get("USE_WANDB", False):
            import wandb
            if wandb.run is not None:
                metric = best_policy_tracker.get('best_metric', 0.0)
                wandb.log({
                    "video/best_episode": wandb.Video(
                        video_path,
                        caption=f"Best Episode (metric: {metric:.4f})",
                        format="mp4"
                    )
                }, commit=False)
                print("✅ Best episode video logged to wandb")


def evaluate_episode_jax(rng_key, env, env_params, train_state, network, max_steps: int = 1000):
    """
    Fast JAX-only episode evaluation (no video recording)
    
    Returns: (achievement_count, episode_return, done, step_count)
    """
    # Reset environment
    # CRITICAL: Use same split order as record_best_episode for deterministic replay
    rng_key, reset_key = jax.random.split(rng_key)

    # Support both single-env and vectorized env adapters. The training loop may
    # pass a VectorizedGymnaxAdapter (which returns batched obs/state). In that
    # case we'll use the underlying single env instance (`_single`) for
    # deterministic per-episode evaluation so that rewards and dones are scalar
    # and accumulate correctly.
    use_single_env = False
    single_env = None
    
    # Determine environment type
    env_type_str = str(type(env))
    is_minigrid = 'MiniGrid' in env_type_str or 'Minigrid' in env_type_str
    
    # Vectorized adapter exposes `_vec._single` (JAXMinigridFourRooms)
    if hasattr(env, '_vec') and hasattr(env._vec, '_single'):
        single_env = env._vec._single
        use_single_env = True

    if use_single_env:
        # single_env.reset returns (state, obs)
        env_state, obs = single_env.reset(reset_key)
    else:
        obs, env_state = env.reset(reset_key, env_params)
    
    # Helper to extract achievements from state
    # MiniGrid (all wrappers): state.achievements or state.env_state.achievements
    # Craftax: state.achievements directly (no env_state wrapper in EnvState)
    if is_minigrid:
        # MiniGrid path: check if wrapped (has env_state) or direct
        def get_achievements(state):
            # If wrapped (LogEnvState), unwrap to get inner state
            if hasattr(state, 'env_state'):
                ach = state.env_state.achievements
                return state.env_state.achievements
            else:
                return state.achievements
    else:
        # Craftax path: achievements are directly on state, but may be wrapped in LogEnvState
        def get_achievements(state):
            # Check if wrapped (e.g., LogEnvState)
            if hasattr(state, 'env_state'):
                # Wrapped state - check if inner state has achievements
                if hasattr(state.env_state, 'achievements'):
                    return state.env_state.achievements
                else:
                    # Inner state doesn't have achievements, return zeros
                    return jnp.zeros(22, dtype=jnp.bool_)
            else:
                # Direct state - achievements should be on state itself
                if hasattr(state, 'achievements'):
                    return state.achievements
                else:
                    # No achievements attribute, return zeros
                    return jnp.zeros(22, dtype=jnp.bool_)
    
    def step_fn(carry, _):
        rng_key, env_state, obs, cumulative_reward, done, max_achievements = carry
        
        # Get action from policy
        # CRITICAL: Use same split order as record_best_episode for deterministic replay
        rng_key, action_key = jax.random.split(rng_key)
        
        # Handle observation - if it's a tuple, use the appropriate element
        # For Craftax-Symbolic, obs is typically a tuple (image, flat_obs)
        actual_obs = obs[1] if isinstance(obs, tuple) and len(obs) > 1 else obs
        
        # Get policy action
        params = train_state.params if hasattr(train_state, 'params') else train_state.get('params')
        network_output = network.apply(params, actual_obs[None, ...])
        
        # Handle both single and dual value head networks
        if len(network_output) == 3:
            # Dual value heads: (pi, value_ext, value_int)
            pi, _, _ = network_output
        else:
            # Single value head: (pi, value)
            pi, _ = network_output
        
        action = pi.sample(seed=action_key)[0]
        
        # Step environment
        # CRITICAL: Use same split order as record_best_episode for deterministic replay
        rng_key, step_key = jax.random.split(rng_key)
        if use_single_env:
            # single_env.step signature: (state, action) -> (state, obs, reward, done, info)
            # Avoid Python `int(...)` on traced JAX values, pass a JAX integer scalar instead.
            env_state_next, obs_next, reward, done_next, _ = single_env.step(env_state, action.astype(jnp.int32))
        else:
            obs_next, env_state_next, reward, done_next, _ = env.step(
                step_key, env_state, action, env_params
            )
        
        # Extract achievements from current state
        current_achievements = get_achievements(env_state_next)
        
        # Accumulate reward (only if not already done before this step)
        cumulative_reward = jnp.where(done, cumulative_reward, cumulative_reward + reward)
        
        # CRITICAL FIX: Track maximum achievements INCLUDING the step where done becomes True
        # Achievements are set in the same step that done becomes True, so we must capture them
        # Only skip updating if episode was already done in the previous step
        max_achievements = jnp.where(done, max_achievements, jnp.maximum(max_achievements, current_achievements))
        
        # Update done status AFTER tracking achievements
        new_done = done | done_next
        
        return (rng_key, env_state_next, obs_next, cumulative_reward, new_done, max_achievements), None
    
    # Run episode for max_steps
    # Initialize max_achievements with zeros
    init_achievements = get_achievements(env_state)
    init_carry = (rng_key, env_state, obs, jnp.float32(0.0), jnp.bool_(False), init_achievements)
    (_, _, _, total_reward, done_flag, max_achievements), _ = jax.lax.scan(
        step_fn, init_carry, None, length=max_steps
    )
    
    # Count achievements from the max_achievements array
    achievement_count = jnp.sum(max_achievements.astype(jnp.int32))
    
    # Use accumulated reward as episode return
    episode_return = total_reward
    
    return achievement_count, episode_return, done_flag, max_steps


def record_final_best_episode(
    config: Dict,
    env,
    env_params,
    train_state,
    network,
    best_policy_tracker: Dict,
    recorder: Optional[EpisodeVideoRecorder] = None,
    replay_saver: Optional[EpisodeReplaySaver] = None,
    num_eval_episodes: int = 10
) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
    """
    Record and log the BEST episode from multiple evaluation runs
    
    This evaluates the best policy across multiple episodes with different
    random seeds and saves only the episode with the highest achievement count
    or episode return.
    
    Args:
        config: Training configuration
        env: Environment instance  
        env_params: Environment parameters
        train_state: Best training state from the entire run
        network: Policy network
        best_policy_tracker: Best policy tracking dictionary
        recorder: Optional pre-initialized recorder
        replay_saver: Optional pre-initialized replay saver
        num_eval_episodes: Number of episodes to evaluate (default: 10)
        
    Returns:
        Tuple of (video_path, replay_paths) where replay_paths is (json_path, pickle_path)
    """
    if not config.get("RECORD_EPISODES", False):
        return None, None

    # If environment is not Craftax Symbolic or MiniGrid, skip video recording
    env_name = config.get("ENV_NAME", "")
    if not (("Craftax" in env_name and "Symbolic" in env_name) or ("MiniGrid" in env_name)):
        print("⚠️  Skipping video recording: recording only supported for Craftax Symbolic and MiniGrid environments")
        return None, None
    
    print(f"\n{'='*70}")
    print(f"🎬 Evaluating best policy across {num_eval_episodes} episodes (PARALLEL)...")
    print(f"   Using JAX vmap for fast parallel evaluation")
    print(f"   Looking for highest achievement count and episode return...")
    if config.get("FIXED_LANDSCAPE", False):
        print(f"   🗺️  Using fixed landscape from training")
    print(f"{'='*70}\n")
    
    seed = config.get("SEED", 42)
    max_steps = config.get("EPISODE_MAX_STEPS", 1000)
    
    # Step 1: PARALLEL EVALUATION - evaluate all episodes simultaneously using vmap
    print(f"⚡ Running {num_eval_episodes} episodes in parallel...")
    
    # Generate all random seeds - use modulo to keep within int32 range while maintaining uniqueness
    # Instead of adding large offsets, use base_offset that wraps around if needed
    INT32_MAX = 2**31 - 1
    base_offset = 100000 % INT32_MAX  # Keep offset within range
    eval_seeds = jnp.array([(seed + base_offset + i * 1000) % INT32_MAX for i in range(num_eval_episodes)], dtype=jnp.int32)
    rng_keys = jax.vmap(jax.random.PRNGKey)(eval_seeds)
    
    # Vectorized evaluation function
    def eval_single_episode(rng_key):
        return evaluate_episode_jax(rng_key, env, env_params, train_state, network, max_steps)
    
    # Run all episodes in parallel using vmap
    import time
    start_time = time.time()
    achievement_counts, episode_returns, _, _ = jax.vmap(eval_single_episode)(rng_keys)
    
    # Convert to numpy for easier processing
    achievement_counts = np.array(achievement_counts)
    episode_returns = np.array(episode_returns)
    eval_time = time.time() - start_time
    
    print(f"✅ Parallel evaluation complete in {eval_time:.2f}s ({num_eval_episodes/eval_time:.2f} episodes/sec)")
    print(f"\n📊 Episode Results:")
    for i in range(num_eval_episodes):
        print(f"   Episode {i+1:2d}: ach={achievement_counts[i]:2d}, return={episode_returns[i]:6.1f}")
    
    # Step 2: SELECT BEST EPISODE - find the episode with highest episode return, then highest achievements
    # Priority: 1) Higher episode return, 2) Higher achievement count
    # np.lexsort uses the last key as the primary sort key; we want episode_returns as primary,
    # so pass the keys as (achievement_counts, episode_returns).
    best_idx = np.lexsort((achievement_counts, episode_returns))[-1]  # Last element after sorting
    # Extract the seed (already in int32 range due to modulo)
    best_seed = int(eval_seeds[best_idx])
    best_ach = int(achievement_counts[best_idx])
    best_ret = float(episode_returns[best_idx])
    
    print(f"\n{'='*70}")
    print(f"🏆 BEST EPISODE IDENTIFIED: Episode {best_idx + 1}")
    print(f"   Seed: {best_seed}")
    print(f"   Achievements (eval): {best_ach}")
    print(f"   Episode Return (eval): {best_ret:.1f}")
    # Confirm the best candidate by running a deterministic single episode
    # using the chosen RNG seed and the same evaluation function, to ensure
    # that the achievements and returns are consistent with the recording
    # step. This prevents vmap/vectorization subtleties from causing
    # inconsistent counts between evaluation and recorded replays.
    try:
        recon_rng = jax.random.PRNGKey(best_seed)
        recon_ach, recon_ret, recon_done, _ = evaluate_episode_jax(recon_rng, env, env_params, train_state, network, max_steps)
        # Convert to Python types reliably
        recon_ach_int = int(np.array(recon_ach))
        recon_ret_float = float(np.array(recon_ret))
        if recon_ach_int != best_ach or abs(recon_ret_float - best_ret) > 1e-6:
            print(f"⚠️  Evaluation mismatch detected for seed {best_seed}: eval(ach={best_ach}, ret={best_ret:.1f}) vs single-run(ach={recon_ach_int}, ret={recon_ret_float:.1f}). Using single-run values for recording.")
            best_ach = recon_ach_int
            best_ret = recon_ret_float
    except Exception as e:
        # If confirmation fails for any reason (e.g. jitted function mismatch),
        # just proceed with the vectorized results to avoid blocking recording.
        print(f"⚠️  Could not confirm best episode via single-run evaluation: {e}")
    print(f"{'='*70}\n")
    
    # Step 3: RECORD BEST EPISODE ONLY - now record with video/replay
    print(f"🎥 Recording best episode (episode {best_idx + 1}) with video and replay data...")
    
    best_rng_key = jax.random.PRNGKey(best_seed)
    video_path, replay_paths = record_best_episode(
        config=config,
        env=env,
        env_params=env_params,
        train_state=train_state,
        network=network,
        rng_key=best_rng_key,
        update_step=999999,  # Temporary step for initial filename
        recorder=recorder,
        replay_saver=replay_saver,
        achievement_count=best_ach,  # Pass the correct achievements from parallel evaluation
        episode_return=best_ret  # Pass the correct return from parallel evaluation
        , normalizer_state=(train_state.get('intrinsic_reward_normalizer') if isinstance(train_state, dict) else getattr(train_state, 'intrinsic_reward_normalizer', None))
    )
    
    if not video_path or not replay_paths:
        print(f"❌ Failed to record best episode!")
        return None, None
    
                # Update tracking data
    best_episode_data = {
        'video_path': video_path,
        'replay_paths': replay_paths,
        'achievement_count': best_ach,
        'episode_return': best_ret,
        'seed_used': best_seed,
        'episode_num': best_idx
    }
    # Prefer recorded metadata over evaluation values for naming & logs if available
    if replay_saver is not None:
        recorded_meta = replay_saver.episode_metadata
        try:
            # If the recorded metadata has achievement count or total reward, use them
            recorded_ach = int(recorded_meta.get('achievement_count', best_episode_data['achievement_count']))
            recorded_ret = float(recorded_meta.get('total_reward', best_episode_data['episode_return']))
            best_episode_data['achievement_count'] = recorded_ach
            best_episode_data['episode_return'] = recorded_ret
        except Exception:
            # Ignore and use evaluation values
            pass
    
    # Step 4: RENAME TO FINAL FORMAT - rename best episode to final_best_episode_achN_retX.X
    # Also record the evaluated metrics inside the saved metadata for traceability.
    if replay_saver is not None:
        try:
            replay_saver.episode_metadata['evaluated_achievement_count'] = int(best_ach)
            replay_saver.episode_metadata['evaluated_episode_return'] = float(best_ret)
        except Exception:
            pass
    final_video_path = best_episode_data['video_path']
    final_replay_paths = best_episode_data['replay_paths']
    
    if final_video_path and os.path.exists(final_video_path):
        # Rename to final_best_episode_achN_retX.X format (more explicit than step999999)
        directory = os.path.dirname(final_video_path)
        final_filename = f"final_best_episode_ach{best_episode_data['achievement_count']}_ret{best_episode_data['episode_return']:.1f}"
        
        new_video_path = os.path.join(directory, final_filename + ".mp4")
        new_json_path = os.path.join(directory, final_filename + "_replay.json")
        new_pkl_path = os.path.join(directory, final_filename + "_state.pkl")
        
        try:
            # Rename video
            if os.path.exists(final_video_path) and final_video_path != new_video_path:
                os.rename(final_video_path, new_video_path)
                final_video_path = new_video_path
            
            # Rename replay files
            if final_replay_paths:
                old_json, old_pkl = final_replay_paths
                if old_json and os.path.exists(old_json) and old_json != new_json_path:
                    os.rename(old_json, new_json_path)
                if old_pkl and os.path.exists(old_pkl) and old_pkl != new_pkl_path:
                    os.rename(old_pkl, new_pkl_path)
                final_replay_paths = (new_json_path, new_pkl_path)
        except Exception as e:
            print(f"⚠️  Could not rename final files: {e}")
        
        print(f"\n{'='*60}")
        print(f"✅ BEST EPISODE FOUND!")
        print(f"   Episode {best_episode_data['episode_num'] + 1}/{num_eval_episodes} (seed: {best_episode_data['seed_used']})")
        print(f"   Achievements: {best_episode_data['achievement_count']}")
        print(f"   Episode Return: {best_episode_data['episode_return']:.1f}")
        print(f"   Video: {os.path.basename(final_video_path)}")
        if final_replay_paths:
            print(f"   Replay: {os.path.basename(final_replay_paths[0])}")
        print(f"{'='*60}")
        
        # Log to wandb with special key for final episode (single upload only)
        if config.get("USE_WANDB", False) and recorder is not None:
            policy_metric = best_policy_tracker.get('best_metric', 0.0)
            caption = f"Best Episode - Achievements: {best_episode_data['achievement_count']}, Return: {best_episode_data['episode_return']:.1f}, Policy Metric: {policy_metric:.4f}"
            recorder.log_to_wandb(
                final_video_path,
                key="video/final_best_episode",
                caption=caption
            )
            print("✅ Best episode video logged to wandb as 'video/final_best_episode'")
        
        print(f"\n{'='*60}\n")
        return final_video_path, final_replay_paths
    else:
        print(f"\n{'='*60}")
        print("⚠️  Failed to record any successful episodes")
        print(f"{'='*60}\n")
        return None, None
