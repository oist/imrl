"""
Episode Replay Saver - Save action sequences and environment state for exact replay

This module saves all information needed to exactly replay a recorded episode:
- Action sequence taken during the episode
- Initial RNG seed used
- Initial environment state
- Episode metadata (length, reward, achievements, etc.)
"""

import os
import json
import pickle
import numpy as np
try:
    import jax
    import jax.numpy as jnp
except Exception:
    # Minimal jax shim for environments/tests without jax
    import types
    import numpy as np
    jax = types.SimpleNamespace()
    jax.device_get = lambda x: x
    jax.random = types.SimpleNamespace(PRNGKey=lambda x: np.array([x, 0], dtype=np.uint32), split=lambda k: (k, k))
    jax.numpy = np
    jnp = np
    class _TreeUtil:
        @staticmethod
        def tree_map(fn, tree):
            if isinstance(tree, dict):
                return {k: _TreeUtil.tree_map(fn, v) for k, v in tree.items()}
            elif isinstance(tree, (list, tuple)):
                return type(tree)(_TreeUtil.tree_map(fn, v) for v in tree)
            else:
                try:
                    return fn(tree)
                except Exception:
                    return tree
    jax.tree_util = _TreeUtil()
from typing import Dict, Any, Optional, List, Tuple


class EpisodeReplaySaver:
    """Saves episode action sequences and states for exact replay"""
    
    def __init__(self, output_dir: str, env_name: Optional[str] = None, save_states: bool = False, save_intrinsic_rewards: bool = False):
        """
        Initialize the replay saver
        
        Args:
            output_dir: Directory to save replay files
            env_name: Name of the environment (optional, for metadata)
            save_states: If True, save complete state trajectory for perfect replay.
                        Warning: This creates large files (~10-100MB per episode)
                        NOTE: This is now forced to True in episode_video_integration.py
                        for accurate replay reproduction
            save_intrinsic_rewards: If True, save intrinsic reward components for each timestep
        """
        self.output_dir = output_dir
        self.env_name = env_name
        self.save_states = save_states
        self.save_intrinsic_rewards = save_intrinsic_rewards
        os.makedirs(output_dir, exist_ok=True)
        
        # Episode recording state
        self.recording = False
        self.actions = []
        self.rewards = []
        self.intrinsic_components = [] if save_intrinsic_rewards else None  # Store intrinsic reward components
        # Store seen object counts per-step (block counts + mob counts) if present
        self.seen_objects = []
        self.state_history = [] if save_states else None  # Store full states if enabled
        self.initial_rng_seed = None
        self.initial_env_state = None
        self.episode_metadata = {}
        # Keep track of the most recent env_state (pre-reset) for metadata
        self._last_env_state = None
        
    def start_recording(self, rng_key, initial_env_state, initial_obs):
        """
        Start recording a new episode
        
        Args:
            rng_key: JAX random key used for episode
            initial_env_state: Initial environment state after reset
            initial_obs: Initial observation
        """
        self.recording = True
        self.actions = []
        self.rewards = []
        if self.save_intrinsic_rewards:
            self.intrinsic_components = []
        if self.save_states:
            self.state_history = []
            # Save initial state
            self.state_history.append(self._convert_to_serializable(initial_env_state))
        
        # Extract seed from rng_key
        # JAX PRNGKey is internally represented as an array of shape (2,) in the new format
        # or scalar in the old format
        if hasattr(rng_key, 'shape') and len(rng_key.shape) > 0:
            # New JAX PRNG format - extract both values
            key_data = np.array(rng_key)
            # Store as a tuple that can be used to recreate the key
            self.initial_rng_seed = tuple(int(x) for x in key_data.flatten())
        elif hasattr(rng_key, 'item'):
            self.initial_rng_seed = int(rng_key.item())
        else:
            self.initial_rng_seed = int(rng_key) if rng_key is not None else 0
        
        # Save initial state - convert JAX arrays to numpy for serialization
        self.initial_env_state = self._convert_to_serializable(initial_env_state)
        self.initial_obs = self._convert_to_serializable(initial_obs)
        
        # Initialize metadata
        self.episode_metadata = {
            'initial_seed': self.initial_rng_seed,
            'step_count': 0,
            'total_reward': 0.0,
        }
        
        # Add environment name if available
        if self.env_name:
            self.episode_metadata['env_name'] = self.env_name
        
        # Save the initial map if available (for fixed landscape replay)
        if hasattr(initial_env_state, 'map'):
            # Save the map separately for easier access
            self.initial_map = np.array(initial_env_state.map)
        else:
            self.initial_map = None
        
    def add_step(self, action, reward, env_state, done, intrinsic_components=None, observation=None):
        """
        Record a step in the episode
        
        Args:
            action: Action taken (int or JAX array)
            reward: Reward received
            env_state: Environment state after action
            done: Whether episode is done
            intrinsic_components: Optional dict of intrinsic reward components
        """
        if not self.recording:
            return
        
        # Convert action to int
        if hasattr(action, 'item'):
            action_int = int(action.item())
        else:
            action_int = int(action)
        
        # Convert reward to float
        if hasattr(reward, 'item'):
            reward_float = float(reward.item())
        else:
            reward_float = float(reward)
        
        self.actions.append(action_int)
        self.rewards.append(reward_float)
        
        # Save intrinsic components if enabled
        if self.save_intrinsic_rewards and self.intrinsic_components is not None:
            if intrinsic_components is not None:
                # Convert JAX arrays to numpy for serialization
                serializable_components = {}
                for key, value in intrinsic_components.items():
                    if hasattr(value, '__array__'):
                        serializable_components[key] = np.array(value)
                    elif hasattr(value, 'item'):
                        serializable_components[key] = float(value.item())
                    else:
                        serializable_components[key] = value
                self.intrinsic_components.append(serializable_components)
            else:
                # No intrinsic components for this step
                self.intrinsic_components.append({})
        
        # Save state if enabled (for perfect replay)
        # Skip saving state when env_state is None (e.g., post-done reset state)
        if self.save_states and self.state_history is not None and env_state is not None:
            self.state_history.append(self._convert_to_serializable(env_state))
        
        # Update metadata
        self.episode_metadata['step_count'] += 1
        self.episode_metadata['total_reward'] += reward_float

        # Keep track of the most recent env_state so we can finalize metadata at the end
        # Only update if env_state is not None (avoid overwriting with post-reset state)
        if env_state is not None:
            self._last_env_state = env_state
        
        # Extract final achievements and stats from env_state
        # Only save final metadata once when the episode is done
        # Use _last_env_state which preserves the pre-reset state
        if done and not hasattr(self, '_final_state_saved'):
            # Be robust if _last_env_state was never set (defensive programming)
            state_to_use = getattr(self, '_last_env_state', None) if getattr(self, '_last_env_state', None) is not None else env_state
            if state_to_use is None:
                self._final_state_saved = True
                return
            actual_state = state_to_use.env_state if hasattr(state_to_use, 'env_state') else state_to_use
            
            if hasattr(actual_state, 'achievements'):
                achievements = np.array(actual_state.achievements)
                self.episode_metadata['achievement_count'] = int(np.sum(achievements))
                self.episode_metadata['achievements'] = achievements.tolist()
            
            # Save player stats if available
            if hasattr(actual_state, 'player_health'):
                self.episode_metadata['final_health'] = float(actual_state.player_health)
            if hasattr(actual_state, 'player_food'):
                self.episode_metadata['final_food'] = float(actual_state.player_food)
            if hasattr(actual_state, 'player_drink'):
                self.episode_metadata['final_drink'] = float(actual_state.player_drink)
            if hasattr(actual_state, 'player_energy'):
                self.episode_metadata['final_energy'] = float(actual_state.player_energy)
            
            self._final_state_saved = True

        # Record seen object counts from observation if provided
        try:
            if observation is not None:
                # Convert to numpy array if needed
                if hasattr(observation, '__array__'):
                    obs_np = np.array(observation)
                else:
                    obs_np = observation

                # Ensure obs has spatial layout: try to extract 9x9x21 spatial map
                # For symbolic Craftax, spatial map occupies the first part of flattened obs
                # Detect shape based on length: if flattened, reshape accordingly
                spatial = None
                try:
                    # Known shapes: 9*9*(#channels) -> 9*9*(len(BlockType)+4mobs)
                    if obs_np.ndim == 1:
                        # Heuristic: if shape corresponds to 9*9*21 + inventory etc, take first part
                        if obs_np.shape[0] >= 9*9*21:
                            spatial_flat = obs_np[:9*9*21]
                            spatial = spatial_flat.reshape((9, 9, 21))
                    elif obs_np.ndim == 3:
                        # Already spatial
                        spatial = obs_np
                except Exception:
                    spatial = None

                if spatial is not None:
                    # Count blocks by summing visibility values across the grid and rounding
                    seen_block_counts = {}
                    mob_seen_counts = {'zombie': 0, 'cow': 0, 'skeleton': 0, 'arrow': 0}
                    # Block channels: first 17 entries
                    for block_type in range(17):
                        val = int(np.sum(spatial[..., block_type] > 0.05))
                        seen_block_counts[str(block_type)] = int(val)
                    # Mob channels 17-20
                    mob_channel_map = {'zombie': 17, 'cow': 18, 'skeleton': 19, 'arrow': 20}
                    for mob_type, ch in mob_channel_map.items():
                        val = int(np.sum(spatial[..., ch] > 0.05))
                        mob_seen_counts[mob_type] = int(val)
                    self.seen_objects.append({'blocks': seen_block_counts, 'mobs': mob_seen_counts})
                else:
                    self.seen_objects.append({'blocks': {}, 'mobs': {}})
            else:
                self.seen_objects.append({'blocks': {}, 'mobs': {}})
        except Exception:
            # On any failure, append an empty marker to keep length consistent
            self.seen_objects.append({'blocks': {}, 'mobs': {}})
    
    def stop_recording(self):
        """Stop recording the episode"""
        self.recording = False
        # If final state metadata not saved yet (e.g., stop_recording called without 'done'), try to save from last known env_state
        try:
            if not hasattr(self, '_final_state_saved') and hasattr(self, '_last_env_state') and self._last_env_state is not None:
                actual_state = self._last_env_state.env_state if hasattr(self._last_env_state, 'env_state') else self._last_env_state
                if hasattr(actual_state, 'achievements'):
                    achievements = np.array(actual_state.achievements)
                    self.episode_metadata['achievement_count'] = int(np.sum(achievements))
                    self.episode_metadata['achievements'] = achievements.tolist()
                # Also set other final fields if available
                try:
                    if hasattr(actual_state, 'episode_returns'):
                        returns = np.array(actual_state.episode_returns)
                        self.episode_metadata['total_reward'] = float(returns.item() if returns.ndim == 0 else returns[0] if returns.ndim == 1 and len(returns) > 0 else self.episode_metadata.get('total_reward', 0.0))
                    elif hasattr(actual_state, 'episode_return'):
                        ret = np.array(actual_state.episode_return)
                        self.episode_metadata['total_reward'] = float(ret.item() if ret.ndim == 0 else ret[0] if ret.ndim == 1 and len(ret) > 0 else self.episode_metadata.get('total_reward', 0.0))
                except Exception:
                    pass
                self._final_state_saved = True
        except Exception:
            pass

        # NOTE: Ensure JSON metadata update is done inside save_replay
    
    def save_replay(self, filename_prefix: str, update_step: Optional[int] = None, normalizer_state: Optional[object] = None) -> Tuple[str, str]:
        """
        Save the recorded episode replay data
        
        Args:
            filename_prefix: Prefix for the saved files
            update_step: Training step when this episode was recorded (optional)
            
        Returns:
            Tuple of (json_path, pickle_path) - paths to saved files
        """
        if update_step is not None:
            self.episode_metadata['update_step'] = update_step
        
        # Create filenames
        json_path = os.path.join(self.output_dir, f"{filename_prefix}_replay.json")
        pickle_path = os.path.join(self.output_dir, f"{filename_prefix}_state.pkl")
        
        # Prepare JSON data (human-readable metadata and actions)
        json_data = {
            'metadata': self.episode_metadata,
            'actions': self.actions,
            'rewards': self.rewards,
            'num_steps': len(self.actions),
        }
        
        # Add intrinsic components to JSON if available
        if self.save_intrinsic_rewards and self.intrinsic_components is not None:
            json_data['intrinsic_components'] = self.intrinsic_components

        # Add seen object counts per-step if recorded
        if self.seen_objects:
            json_data['seen_objects'] = self.seen_objects

        # Add a serialized normalizer_state to metadata if provided
        if normalizer_state is not None:
            try:
                # Convert to serializable structure: namedtuple->dict, arrays -> lists
                norm_serial = self._convert_to_serializable(normalizer_state)
                # Convert numpy arrays to lists for JSON serialization
                def _to_python(obj):
                    import numpy as _np
                    if isinstance(obj, _np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, dict):
                        return {k: _to_python(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_to_python(v) for v in obj]
                    return obj
                json_data['metadata']['normalizer_state'] = _to_python(norm_serial)
            except Exception:
                # Silently ignore failures to add normalizer snapshot
                pass
        
        # Convert everything in json_data to JSON serializable Python types
        def _to_python_json(obj):
            import numpy as _np
            # Convert numpy arrays to python lists
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            # Convert JAX arrays / array-like objects with the numpy protocol
            if hasattr(obj, '__array__'):
                try:
                    as_np = _np.array(obj)
                    return _to_python_json(as_np)
                except Exception:
                    pass
            # numpy scalar types: convert to Python types
            if isinstance(obj, (_np.generic,)):
                return obj.item()
            # Recursively convert dicts
            if isinstance(obj, dict):
                return {k: _to_python_json(v) for k, v in obj.items()}
            # Convert lists/tuples recursively
            if isinstance(obj, list):
                return [_to_python_json(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_to_python_json(v) for v in obj)
            # Fallback: return as-is (strings, ints, floats OK)
            return obj

        # Ensure achievement count in metadata matches final saved state (if present)
        try:
            # If we stored state_history in pickle_data, prefer that
            if self.save_states and self.state_history is not None and len(self.state_history) > 0:
                final_state = self.state_history[-1]
                # Support both namedtuple-like states with attributes and
                # serialized state dictionaries produced by `_convert_to_serializable`.
                if hasattr(final_state, 'achievements'):
                        final_ach_np = np.array(final_state.achievements)
                        self.episode_metadata['achievement_count'] = int(np.sum(final_ach_np))
                        self.episode_metadata['achievements'] = final_ach_np.tolist()
                elif isinstance(final_state, dict):
                        # New serialized format may either include achievements directly
                        if 'achievements' in final_state and final_state['achievements'] is not None:
                            try:
                                final_ach_np = np.array(final_state['achievements'])
                                self.episode_metadata['achievement_count'] = int(np.sum(final_ach_np))
                                self.episode_metadata['achievements'] = final_ach_np.tolist()
                            except Exception:
                                # If conversion fails, try to detect nested namedtuple-like serialization
                                # Format produced by _convert_to_serializable: keys '__namedtuple__', '__fields__', '__values__'
                                pass
                        elif '__namedtuple__' in final_state and 'achievements' in final_state.get('__fields__', []):
                            try:
                                idx = list(final_state.get('__fields__', [])).index('achievements')
                                val = final_state.get('__values__', [])[idx]
                                final_ach_np = np.array(val)
                                self.episode_metadata['achievement_count'] = int(np.sum(final_ach_np))
                                self.episode_metadata['achievements'] = final_ach_np.tolist()
                            except Exception:
                                pass
            elif hasattr(self, '_last_env_state') and self._last_env_state is not None:
                final_state = self._last_env_state
                if hasattr(final_state, 'achievements'):
                    final_ach_np = np.array(final_state.achievements)
                    self.episode_metadata['achievement_count'] = int(np.sum(final_ach_np))
                    self.episode_metadata['achievements'] = final_ach_np.tolist()
        except Exception:
            pass

        # CRITICAL: Ensure that the metadata['achievement_count'] and
        # metadata['achievements'] are always consistent. If 'achievements'
        # is present and contains a list-like structure, recompute and
        # normalize the 'achievement_count' to reduce mismatches when
        # renaming or displaying the replay in the viewer.
        try:
            # Use the current in-memory episode metadata (self.episode_metadata)
            # so the recomputation logic reflects any updates from final state
            # inspection we have just performed above.
            md = dict(self.episode_metadata) if isinstance(self.episode_metadata, dict) else json_data.get('metadata', {})
            if 'achievements' in md and md['achievements'] is not None:
                # Count entries that are truthy (bools or ints), matching how the
                # viewer counts unlocked achievements. This avoids differences
                # when an achievements vector contains non-boolean values.
                achs = md['achievements']
                try:
                    # Accept numpy arrays and lists; coerce to Python list
                    ach_list = list(np.array(achs).tolist())
                except Exception:
                    ach_list = list(achs)
                md['achievement_count'] = int(sum(1 for a in ach_list if bool(a)))
                # Make sure we write back into both self.episode_metadata and
                # the local json_data's metadata object.
                self.episode_metadata['achievement_count'] = md['achievement_count']
                self.episode_metadata['achievements'] = ach_list
                json_data['metadata'] = md
        except Exception:
            # Do not fail saving over a metadata consistency issue
            pass

        json_data = _to_python_json(json_data)
        # Save JSON file
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Prepare pickle data (full state for exact replay)
        pickle_data = {
            'initial_rng_seed': self.initial_rng_seed,
            'initial_env_state': self.initial_env_state,
            'initial_obs': self.initial_obs,
            'initial_map': self.initial_map,  # Save map separately for fixed landscape
            'actions': self.actions,
            'rewards': self.rewards,
            'metadata': self.episode_metadata,
            'has_state_history': self.save_states,  # Flag to indicate if states are saved
            'has_intrinsic_components': self.save_intrinsic_rewards,  # Flag to indicate if intrinsic components are saved
        }
        
        # Add intrinsic components to pickle if available
        if self.save_intrinsic_rewards and self.intrinsic_components is not None:
            pickle_data['intrinsic_components'] = self.intrinsic_components
        # Add the raw normalizer snapshot to pickle as well if provided
        if normalizer_state is not None:
            pickle_data['normalizer_state'] = normalizer_state

        # Add seen objects list to pickle data
        if self.seen_objects:
            pickle_data['seen_objects'] = self.seen_objects
        
        # Add state history if available (for perfect deterministic replay)
        if self.save_states and self.state_history is not None:
            pickle_data['state_history'] = self.state_history
            print(f"   💾 Saved {len(self.state_history)} states for perfect replay")
        
        # Save pickle file
        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✅ Episode replay saved:")
        print(f"   JSON: {json_path}")
        print(f"   State: {pickle_path}")
        print(f"   Steps: {len(self.actions)}, Reward: {self.episode_metadata['total_reward']:.2f}")
        if self.save_intrinsic_rewards:
            ic_count = len(self.intrinsic_components) if self.intrinsic_components else 0
            print(f"   Intrinsic Components: {ic_count} timesteps")
            # If all intrinsic components are empty dicts, warn the user to re-record
            if ic_count > 0:
                try:
                    non_empty = sum(1 for ic in self.intrinsic_components if ic and len(ic) > 0)
                    if non_empty == 0:
                        print("⚠️  Warning: Intrinsic components were recorded but all steps are empty, likely the VAE failed during recording. Re-record the episode with proper normalizer and VAE configuration.")
                except Exception:
                    pass
        
        return json_path, pickle_path
    
    def _convert_to_serializable(self, obj):
        """
        Convert JAX arrays and nested structures to serializable format
        
        Args:
            obj: Object to convert (can be pytree, array, etc.)
            
        Returns:
            Serializable version (numpy arrays, dicts, lists)
        """
        if obj is None:
            return None
        
        # Handle JAX arrays
        if hasattr(obj, '__array__'):
            return np.array(obj)
        
        # Handle namedtuples (must check before tuple since namedtuples are tuples)
        if hasattr(obj, '_fields'):
            # This is a namedtuple - preserve structure
            return {
                '__namedtuple__': True,
                '__class__': obj.__class__.__name__,
                '__module__': obj.__class__.__module__,
                '__fields__': obj._fields,
                '__values__': tuple(self._convert_to_serializable(getattr(obj, field)) for field in obj._fields)
            }
        
        # Handle regular tuples
        if isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        
        # Handle lists
        if isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        
        # Handle dicts
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        
        # Handle objects with __dict__ (like EnvState) - but these should be caught by namedtuple check above
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    try:
                        result[key] = self._convert_to_serializable(value)
                    except Exception:
                        # Skip attributes that can't be serialized
                        pass
            return result
        
        # Return primitive types as-is
        return obj


def _reconstruct_env_state(env, saved_state_dict):
    """
    Reconstruct environment state from saved dictionary
    
    Args:
        env: Environment instance
        saved_state_dict: Dictionary containing saved state data
        
    Returns:
        Reconstructed environment state object
    """
    # Import the appropriate EnvState class based on environment
    try:
        # Try to get the state class from the environment
        if hasattr(env, 'env'):
            # Unwrapped environment
            actual_env = env.env
        else:
            actual_env = env
        
        # Get the state class
        if hasattr(actual_env, 'EnvState'):
            EnvState = actual_env.EnvState
        else:
            # Try to import based on common patterns
            env_module = actual_env.__class__.__module__
            if 'craftax_classic' in env_module:
                from envs.craftax.craftax_classic.envs.craftax_state import EnvState
            else:
                from envs.craftax.craftax.envs.craftax_state import EnvState
        
        # Convert numpy arrays back to JAX arrays
        state_data = {}
        for key, value in saved_state_dict.items():
            if isinstance(value, np.ndarray):
                state_data[key] = jnp.array(value)
            elif isinstance(value, dict):
                # Recursively convert nested dicts
                state_data[key] = {k: jnp.array(v) if isinstance(v, np.ndarray) else v 
                                  for k, v in value.items()}
            else:
                state_data[key] = value
        
        # Create the state object
        env_state = EnvState(**state_data)
        return env_state
        
    except Exception as e:
        raise RuntimeError(f"Failed to reconstruct environment state: {e}")


def replay_from_state_history(
    env,
    env_params,
    actions: List[int],
    state_history: List[Any],
    rewards: Optional[List[float]] = None,
    fixed_landscape_map: Optional[Any] = None
):
    """
    Replay episode using saved state history for perfect deterministic replay
    
    This method provides frame-by-frame perfect reproduction by using the exact states
    that were recorded during the original episode. No RNG or action execution needed.
    
    Args:
        env: Environment instance (used for observation generation if needed)
        env_params: Environment parameters
        actions: List of actions (for metadata/verification)
        state_history: List of ALREADY RESTORED environment states (restored in load_replay_data)
        rewards: Optional list of rewards for each step
        fixed_landscape_map: Optional fixed landscape map to inject into all states
        
    Returns:
        List of (obs, env_state, reward, done, info) tuples for each state
    """
    print(f"🎬 Generating perfect replay from {len(state_history)} saved states...")
    
    trajectory = []
    
    # States are already restored to proper format in load_replay_data
    # Just need to generate observations
    for i, env_state in enumerate(state_history):
        
        # If we have a fixed landscape map, ensure all states use it
        if fixed_landscape_map is not None and hasattr(env_state, 'map'):
            env_state = env_state.replace(map=fixed_landscape_map)
        
        # Generate observation from state (if environment supports it)
        
        # Generate observation from state (if environment supports it)
        try:
            # Most Craftax environments have get_obs method
            if hasattr(env, 'get_obs'):
                obs = env.get_obs(env_state)
            elif hasattr(env, '_get_obs'):
                obs = env._get_obs(env_state)
            else:
                # Fallback: use the state's observation if available
                obs = getattr(env_state, 'obs', jnp.zeros((32, 32, 6)))
        except Exception as e:
            # If observation generation fails, use zero observation
            print(f"⚠️  Warning: Could not generate observation at step {i}: {e}")
            obs = jnp.zeros((32, 32, 6))
        
        # Extract reward and done from state (if available)
        reward = 0.0
        if rewards and i < len(rewards):
            reward = rewards[i]
        done = False
        
        # Check if episode is done (if state has done flag)
        if hasattr(env_state, 'done'):
            done = bool(env_state.done)
        
        trajectory.append((obs, env_state, reward, done, {}))
    
    print(f"✅ Perfect replay trajectory generated ({len(trajectory)} frames)")
    return trajectory


def _restore_state_from_serializable(saved_state):
    """
    Convert serializable state (numpy arrays) back to JAX format, including namedtuples
    
    Args:
        saved_state: Saved state with numpy arrays and serialized namedtuples
        
    Returns:
        State with JAX arrays and reconstructed namedtuples
    """
    import importlib
    from collections import namedtuple
    
    if saved_state is None:
        return None
    
    # Handle numpy arrays
    if isinstance(saved_state, np.ndarray):
        return jnp.array(saved_state)
    
    # Handle dictionaries - check if it's a serialized namedtuple
    if isinstance(saved_state, dict):
        if saved_state.get('__namedtuple__'):
            # Reconstruct the namedtuple
            try:
                module = importlib.import_module(saved_state['__module__'])
                cls = getattr(module, saved_state['__class__'])
            except (ImportError, AttributeError):
                # If we can't import the class, create a generic namedtuple
                cls = namedtuple(saved_state['__class__'], saved_state['__fields__'])
            
            # Recursively restore the field values
            restored_values = tuple(
                _restore_state_from_serializable(val) 
                for val in saved_state['__values__']
            )
            
            # Create the namedtuple instance
            return cls(*restored_values)
        else:
            # Regular dict - recursively restore values
            return {k: _restore_state_from_serializable(v) for k, v in saved_state.items()}
    
    # Handle tuples
    if isinstance(saved_state, tuple):
        # Check if this is a namedtuple by looking for _fields
        if hasattr(saved_state, '_fields'):
            # Reconstruct namedtuple with JAX arrays
            return type(saved_state)(*(_restore_state_from_serializable(item) for item in saved_state))
        else:
            return tuple(_restore_state_from_serializable(item) for item in saved_state)
    
    # Handle lists
    if isinstance(saved_state, list):
        return [_restore_state_from_serializable(item) for item in saved_state]
    
    # Return as-is for primitives
    return saved_state


def load_replay_data(replay_json_path: str, replay_pickle_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load saved replay data
    
    Args:
        replay_json_path: Path to JSON replay file
        replay_pickle_path: Optional path to pickle state file
        
    Returns:
        Dictionary containing replay data
    """
    # Load JSON data
    with open(replay_json_path, 'r') as f:
        json_data = json.load(f)
    
    result = {
        'actions': json_data['actions'],
        'rewards': json_data.get('rewards', []),
        'metadata': json_data['metadata'],
        'num_steps': json_data['num_steps'],
        'intrinsic_components': json_data.get('intrinsic_components', []),
        'has_intrinsic_components': len(json_data.get('intrinsic_components', [])) > 0,
        'seen_objects': json_data.get('seen_objects', []),
    }
    
    # Load pickle data if available
    if replay_pickle_path and os.path.exists(replay_pickle_path):
        try:
            with open(replay_pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
        except ModuleNotFoundError as e:
            # numpy 1.x stores certain objects in ``numpy._core.numeric``;
            # older installations (1.26) sometimes do not expose that module
            # when unpickling files created with numpy>=2.  Remap the name to
            # the canonical path and retry.
            if 'numpy._core.numeric' in str(e):
                class RenamingUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'numpy._core.numeric':
                            module = 'numpy.core.numeric'
                        return super().find_class(module, name)

                with open(replay_pickle_path, 'rb') as f:
                    pickle_data = RenamingUnpickler(f).load()
            else:
                raise
        except TypeError as e:
            # Handle backwards compatibility with old RunningStats format
            # Old format had fewer fields (mean, var, count) vs new format
            # (mean, var, count, min_val, max_val, sum_sq)
            if "missing" in str(e) and ("min_val" in str(e) or "max_val" in str(e) or "sum_sq" in str(e)):
                from typing import NamedTuple
                
                # Create a compatible old-style RunningStats for unpickling
                class OldRunningStats(NamedTuple):
                    """Old RunningStats format with only 3 fields."""
                    mean: np.ndarray
                    var: np.ndarray
                    count: np.ndarray
                
                class BackwardsCompatibleUnpickler(pickle.Unpickler):
                    """Custom unpickler that substitutes old RunningStats."""
                    
                    def find_class(self, module, name):
                        # Substitute RunningStats with old format version
                        if name == 'RunningStats' and 'normalization' in module:
                            return OldRunningStats
                        return super().find_class(module, name)
                
                # Re-read with backwards compatible unpickler
                with open(replay_pickle_path, 'rb') as f:
                    pickle_data = BackwardsCompatibleUnpickler(f).load()
                
                # Convert old RunningStats to new format if normalizer_state exists
                if 'normalizer_state' in pickle_data:
                    from scire import RunningStats, NormalizerState

                    def upgrade_running_stats(old_stats):
                        """Convert old 3-field RunningStats to new 6-field format."""
                        if isinstance(old_stats, OldRunningStats):
                            mean, var, count = old_stats
                            min_val = np.array(float('inf'), dtype=np.float32)
                            max_val = np.array(float('-inf'), dtype=np.float32)
                            sum_sq = np.zeros_like(mean) if hasattr(mean, 'shape') else np.zeros((), dtype=np.float32)
                            return RunningStats(mean, var, count, min_val, max_val, sum_sq)
                        return old_stats

                    old_norm = pickle_data['normalizer_state']
                    if hasattr(old_norm, '_fields'):
                        # It's a namedtuple (old IntrinsicRewardNormalizerState); keep only the
                        # novelty/surprise/empowerment fields that scire.NormalizerState still has.
                        upgraded_fields = {
                            field: upgrade_running_stats(getattr(old_norm, field))
                            for field in old_norm._fields
                            if field in NormalizerState._fields
                        }
                        pickle_data['normalizer_state'] = NormalizerState(**upgraded_fields)
            else:
                raise
        result['initial_rng_seed'] = pickle_data['initial_rng_seed']
        result['initial_env_state'] = pickle_data.get('initial_env_state')
        result['initial_obs'] = pickle_data.get('initial_obs')
        result['initial_map'] = pickle_data.get('initial_map')  # Load the saved map
        result['has_state_history'] = pickle_data.get('has_state_history', False)
        result['has_intrinsic_components'] = pickle_data.get('has_intrinsic_components', False)
        
        # Load intrinsic components if available
        if result['has_intrinsic_components']:
            result['intrinsic_components'] = pickle_data.get('intrinsic_components', [])
        # Load seen objects if available
        result['seen_objects'] = pickle_data.get('seen_objects', []) if 'seen_objects' in pickle_data else result.get('seen_objects', [])

        # Load normalizer_state from pickle if present and not already in metadata
        if 'normalizer_state' in pickle_data:
            norm_obj = pickle_data['normalizer_state']
            # If metadata does not already contain a serialized version, try to build one
            if 'normalizer_state' not in result['metadata']:
                try:
                    # If this is a namedtuple, call _asdict and convert gen arrays to lists
                    if hasattr(norm_obj, '_asdict'):
                        nd = norm_obj._asdict()
                        serial = {}
                        for k, v in nd.items():
                            # v is a RunningStats namedtuple
                            serial[k] = {
                                'mean': np.array(v.mean).tolist(),
                                'var': np.array(v.var).tolist(),
                                'count': float(v.count)
                            }
                        result['metadata']['normalizer_state'] = serial
                    else:
                        # Not a namedtuple; just use as-is
                        result['metadata']['normalizer_state'] = norm_obj
                except Exception:
                    result['metadata']['normalizer_state'] = norm_obj
            # Also include the raw normalizer in top-level result for consumer convenience
            result['normalizer_state'] = norm_obj
        
        # CRITICAL: Restore state history from serialized format (dicts) to proper namedtuples
        raw_state_history = pickle_data.get('state_history', None)
        if raw_state_history is not None:
            print(f"🔄 Restoring {len(raw_state_history)} states from serialized format...")
            
            # Check format: new format has __namedtuple__ marker, old format is plain dict
            if raw_state_history and isinstance(raw_state_history[0], dict):
                if '__namedtuple__' in raw_state_history[0]:
                    # New format with namedtuple markers - use recursive restoration
                    print(f"   📦 Detected new format (with namedtuple markers)")
                    result['state_history'] = [
                        _restore_state_from_serializable(state) 
                        for state in raw_state_history
                    ]
                else:
                    # Old format - plain dicts without markers
                    # Need to reconstruct Flax structs using actual classes
                    print(f"   📦 Detected old format (plain dicts) - reconstructing Flax struct objects...")
                    
                    # Detect environment type from metadata or state structure
                    env_name = result.get('metadata', {}).get('env_name', '')
                    is_minigrid = 'minigrid' in env_name.lower() if env_name else False
                    
                    # Try to detect from state structure if env_name not available
                    if not env_name and raw_state_history:
                        sample_state = raw_state_history[0]
                        # MiniGrid states have: agent_pos, agent_dir, key_pos, etc.
                        # Craftax states have: player_position, player_direction, map, etc.
                        if 'agent_pos' in sample_state and 'agent_dir' in sample_state:
                            is_minigrid = True
                        elif 'player_position' in sample_state and 'player_direction' in sample_state:
                            is_minigrid = False
                    
                    # Import the appropriate EnvState class based on environment type
                    if is_minigrid:
                        # Try both MiniGrid state formats
                        try:
                            # Check which state format is used by examining sample state
                            sample_state = raw_state_history[0]
                            if 'key' in sample_state and 'obs' in sample_state:
                                # MiniGridEnvState format (has 'key', 'has_key', 'door_unlocked', 'obs')
                                from envs.minigrid.minigrid_jax_env import MiniGridEnvState as EnvState
                                print(f"   ✅ Imported MiniGridEnvState from minigrid_jax_env")
                            elif 'rng_key' in sample_state and 'grid' in sample_state:
                                # EnvState format (has 'rng_key', 'key_picked', 'door_open', 'grid')
                                from envs.minigrid.jax_minigrid_env import EnvState
                                print(f"   ✅ Imported EnvState from jax_minigrid_env")
                            else:
                                # Default to first format
                                from envs.minigrid.minigrid_jax_env import MiniGridEnvState as EnvState
                                print(f"   ✅ Imported MiniGridEnvState (default)")
                            # MiniGrid doesn't have nested structs like Inventory/Mobs
                            Inventory = None
                            Mobs = None
                        except ImportError:
                            raise ImportError("Could not import EnvState class from MiniGrid")
                    else:
                        # Import the actual EnvState classes from Craftax
                        try:
                            from envs.craftax.craftax_classic.envs.craftax_state import EnvState, Inventory, Mobs
                            print(f"   ✅ Imported EnvState from craftax_classic")
                        except ImportError:
                            try:
                                from envs.craftax.craftax.craftax_state import EnvState, Inventory, Mobs
                                print(f"   ✅ Imported EnvState from craftax")
                            except ImportError:
                                raise ImportError("Could not import EnvState class from Craftax")
                    
                    def reconstruct_flax_structs(obj, cls_hint=None):
                        """Recursively reconstruct Flax structs from plain dicts"""
                        if obj is None:
                            return None
                        
                        # Convert numpy arrays to JAX arrays
                        if isinstance(obj, np.ndarray):
                            return jnp.array(obj)
                        
                        # Reconstruct dicts as Flax structs
                        if isinstance(obj, dict):
                            # Determine the class for this dict
                            if cls_hint is None:
                                # Top level - must be EnvState
                                struct_cls = EnvState
                            elif not is_minigrid and 'wood' in obj and 'stone' in obj and 'coal' in obj:
                                # Inventory struct (Craftax only)
                                struct_cls = Inventory
                            elif not is_minigrid and 'position' in obj and 'health' in obj and 'mask' in obj:
                                # Mobs struct (Craftax only)
                                struct_cls = Mobs
                            else:
                                # Unknown - try to convert to dict with JAX arrays
                                return {k: reconstruct_flax_structs(v) for k, v in obj.items()}
                            
                            # Recursively process all fields
                            kwargs = {}
                            for key, value in obj.items():
                                # Determine hint for nested structures
                                hint = key if cls_hint is None else None
                                kwargs[key] = reconstruct_flax_structs(value, hint)
                            
                            # Create the struct instance
                            return struct_cls(**kwargs)
                        
                        # Handle tuples
                        if isinstance(obj, tuple):
                            return tuple(reconstruct_flax_structs(item) for item in obj)
                        
                        # Handle other types as-is
                        return obj
                    
                    # Reconstruct all states
                    print(f"   ⚙️  Reconstructing Flax struct hierarchy...")
                    result['state_history'] = [
                        reconstruct_flax_structs(state_dict)
                        for state_dict in raw_state_history
                    ]
            else:
                result['state_history'] = raw_state_history
            
            print("✅ States restored to proper namedtuple format")
            # Recompute and reconcile metadata from restored final state if present
            try:
                if result.get('state_history'):
                    final_state_restored = result['state_history'][-1]
                    # Try attribute-style first
                    if hasattr(final_state_restored, 'achievements'):
                        final_ach = np.array(final_state_restored.achievements)
                        result['metadata']['achievement_count'] = int(np.sum(final_ach))
                        result['metadata']['achievements'] = final_ach.tolist()
                    elif isinstance(final_state_restored, dict):
                        if 'achievements' in final_state_restored and final_state_restored['achievements'] is not None:
                            final_ach = np.array(final_state_restored['achievements'])
                            result['metadata']['achievement_count'] = int(np.sum(final_ach))
                            result['metadata']['achievements'] = final_ach.tolist()
                        elif '__namedtuple__' in final_state_restored and 'achievements' in final_state_restored.get('__fields__', []):
                            try:
                                idx = list(final_state_restored.get('__fields__', [])).index('achievements')
                                val = final_state_restored.get('__values__', [])[idx]
                                final_ach = np.array(val)
                                result['metadata']['achievement_count'] = int(np.sum(final_ach))
                                result['metadata']['achievements'] = final_ach.tolist()
                            except Exception:
                                pass
            except Exception:
                pass
        else:
            result['state_history'] = None
    
    return result


def replay_episode_with_actions(
    env,
    env_params,
    actions: List[int],
    initial_rng_seed,  # Can be int or tuple
    initial_env_state: Optional[Any] = None,
    fixed_landscape_map: Optional[Any] = None,
    state_history: Optional[List[Any]] = None,
    rewards: Optional[List[float]] = None
):
    """
    Replay an episode using saved actions or state history
    
    If state_history is provided, uses it for perfect deterministic replay.
    Otherwise, uses actions and RNG sequence (may have minor differences due to stochastic elements).
    
    CRITICAL: This function must follow the EXACT SAME RNG sequence as during recording:
    1. Start with original_rng_key (the one saved)
    2. Split to get (rng_key, reset_key)
    3. Use reset_key for env.reset()
    4. Use rng_key for all subsequent splits during stepping
    
    Args:
        env: Environment instance
        env_params: Environment parameters
        actions: List of actions to replay
        initial_rng_seed: ORIGINAL rng_key seed (before any splits during recording)
        initial_env_state: Optional saved initial state (if available, uses this instead of resetting)
        fixed_landscape_map: Optional fixed landscape map (JAX array) to manually inject after reset
        state_history: Optional list of saved environment states for perfect replay
        rewards: Optional list of rewards for each step (used with state_history)
        
    Returns:
        List of (obs, env_state, reward, done, info) tuples for each step
    """
    
    # If state history is available, use it for perfect replay
    if state_history is not None and len(state_history) > 0:
        print(f"✅ Using saved state history for perfect deterministic replay ({len(state_history)} states)")
        return replay_from_state_history(env, env_params, actions, state_history, rewards, fixed_landscape_map)
    
    # Otherwise, use action replay (may have minor differences due to stochasticity)
    print(f"⚠️  Using action replay (minor differences possible with stochastic elements)")
    # Reconstruct the ORIGINAL RNG key from saved seed (before any splits)
    if isinstance(initial_rng_seed, (tuple, list)):
        # Seed was saved as tuple (new JAX PRNG format)
        # Reconstruct the key directly
        original_rng_key = jnp.array(initial_rng_seed, dtype=jnp.uint32)
    else:
        # Seed was saved as single integer (fallback)
        original_rng_key = jax.random.PRNGKey(initial_rng_seed)
    
    # CRITICAL: Follow the EXACT SAME split sequence as during recording
    # Step 1: Split original key to get reset_key (same as recording)
    rng_key, reset_key = jax.random.split(original_rng_key)
    
    # Get the underlying environment (unwrap if needed)
    unwrapped_env = env
    if hasattr(env, 'env'):
        unwrapped_env = env.env
    
    # Step 2: Reset environment using the same reset_key as recording
    # Don't pass fixed_landscape_map in params to avoid hashability issues
    obs, env_state = unwrapped_env.reset(reset_key, env_params)
    
    # If we have a fixed landscape map, inject it into the state after reset
    if fixed_landscape_map is not None:
        env_state = env_state.replace(map=fixed_landscape_map)
    
    # Replay actions
    trajectory = [(obs, env_state, 0.0, False, {})]
    
    # CRITICAL: Use the SAME rng_key sequence as during recording
    # During recording, after reset, the rng_key is used for ALL subsequent splits
    # (action_key splits and step_key splits alternate)
    
    # Pre-compile first step for faster subsequent steps
    print(f"🔥 Compiling replay (first step may take 10-30 seconds)...")
    for i, action in enumerate(actions):
        # CRITICAL: Follow the SAME split sequence as during recording:
        # 1. Split for action_key (even though we don't use it - actions are saved)
        rng_key, action_key = jax.random.split(rng_key)
        # 2. Split for step_key (used in env.step)
        rng_key, step_key = jax.random.split(rng_key)
        
        # Use JIT-compiled step method (fast) - fixed_landscape_map is not in params so no hashability issues
        step_result = unwrapped_env.step(step_key, env_state, action, env_params)
        
        # Show progress for first compilation only
        if i == 0:
            print(f"✅ Compilation complete! Replaying {len(actions)} actions...")
        
        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs, env_state, reward, done, info = step_result
        else:
            raise ValueError(f"Unexpected step result: {type(step_result)}")
        
        trajectory.append((obs, env_state, reward, done, info))
        
        # Check if done
        if hasattr(done, 'item'):
            done_bool = bool(done.item())
        else:
            done_bool = bool(done)
        
        if done_bool:
            break
    
    return trajectory
