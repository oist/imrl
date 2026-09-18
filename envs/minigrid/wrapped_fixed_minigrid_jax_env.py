"""
Wrapped fixed MiniGrid JAX environment that uses the same wrappers as training.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Any
from gymnasium import make
from minigrid.wrappers import OneHotPartialObsWrapper
from envs.minigrid.minigrid_wrappers import DictToArrayObsWrapper

from .minigrid_jax_env import MiniGridJAXWrapper, MiniGridEnvState, MiniGridEnvParams


class WrappedFixedMiniGridJaxEnv(MiniGridJAXWrapper):
    """Fixed version that uses real MiniGrid environment with proper wrappers."""
    
    def __init__(self, env_name: str = "MiniGrid-DoorKey-5x5-v0"):
        # Initialize environment attributes without calling parent __init__
        self.env_id = env_name
        self._action_dim = 7  # MiniGrid standard action space  
        self._max_episode_steps = 1000
        
        # Create the real environment with the same wrappers as training
        self._gym_env = self._create_wrapped_env()
        
        # Get observation shape from wrapped environment
        obs, _ = self._gym_env.reset()
        obs_array = jnp.array(obs, dtype=jnp.float32)
        if obs_array.ndim == 0:
            obs_array = obs_array.reshape(1)
        self._obs_shape = obs_array.shape
        
        print(f"MiniGrid env initialized - obs shape: {self._obs_shape}, obs dtype: {obs_array.dtype}")
        print(f"Action space: {self._action_dim}")
        
        # Store current state for tracking
        self._current_obs = None
        self._current_reward = 0.0
        self._current_done = False
        self._step_count = 0
    
    def _create_wrapped_env(self):
        """Create a wrapped MiniGrid environment instance."""
        env = make(self.env_id)
        env = OneHotPartialObsWrapper(env)
        env = DictToArrayObsWrapper(env)
        return env
    
    @property
    def default_params(self):
        """Return default parameters."""
        return MiniGridEnvParams(max_steps=self._max_episode_steps)

    def observation_space(self, params = None):
        """Return the observation space of the wrapped environment."""
        from gymnasium import spaces
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._obs_shape,
            dtype=jnp.float32
        )
    
    def action_space(self, params = None):
        """Return the action space of the real environment."""
        from gymnasium import spaces
        return spaces.Discrete(self._action_dim)

    def step_env(
        self, rng: jax.Array, state: MiniGridEnvState, action: int, params: MiniGridEnvParams
    ) -> Tuple[jax.Array, MiniGridEnvState, float, bool, Dict[str, Any]]:
        """Step using the real wrapped environment."""
        
        # Convert JAX action to numpy
        action_np = int(action)
        
        # Step the real environment
        obs, reward, terminated, truncated, info = self._gym_env.step(action_np)
        done = terminated or truncated
        
        # Convert observation to JAX format
        obs_jax = jnp.array(obs, dtype=jnp.float32)
        
        # Update tracking variables
        self._current_obs = obs_jax
        self._current_reward = float(reward)
        self._current_done = done
        self._step_count += 1
        
        # Extract real environment state
        unwrapped = self._gym_env
        while hasattr(unwrapped, 'env'):
            unwrapped = unwrapped.env
        
        agent_pos = jnp.array([
            int(unwrapped.agent_pos[0]),
            int(unwrapped.agent_pos[1])
        ])
        agent_dir = int(unwrapped.agent_dir)
        
        # Check for key
        if hasattr(unwrapped, 'key_pos') and unwrapped.key_pos is not None:
            key_pos = jnp.array([
                int(unwrapped.key_pos[0]),
                int(unwrapped.key_pos[1])
            ])
        else:
            key_pos = jnp.array([-1, -1])
        
        # Check if carrying key
        has_key = getattr(unwrapped, 'carrying', None) is not None
        door_unlocked = getattr(unwrapped, 'door_unlocked', True)
        
        # Calculate achievements
        achievements = jnp.zeros(2)
        if has_key:
            achievements = achievements.at[0].set(1)  # Key achievement
        if done and reward > 0:
            achievements = achievements.at[1].set(1)  # Goal achievement
        
        # Create new state
        new_state = MiniGridEnvState(
            key=state.key,
            step_count=state.step_count + 1,
            done=done,
            obs=obs_jax,
            agent_dir=agent_dir,
            agent_pos=agent_pos,
            achievements=achievements,
            key_pos=key_pos,
            has_key=has_key,
            door_unlocked=door_unlocked
        )
        
        # Create info dict
        info_dict = {
            "discount": jax.lax.select(done, 0.0, 1.0),
            "achievement": jax.lax.select(done and reward > 0, 1, 0),
            "large_reward_goal": jax.lax.select(done and reward > 0, 1, 0),
            "small_reward_goal": 0
        }
        
        return obs_jax, new_state, float(reward), done, info_dict

    def reset_env(
        self, rng: jax.Array, params: MiniGridEnvParams
    ) -> Tuple[jax.Array, MiniGridEnvState]:
        """Reset using the real wrapped environment."""
        
        # Reset the real environment
        obs, info = self._gym_env.reset()
        obs_jax = jnp.array(obs, dtype=jnp.float32)
        
        # Update tracking
        self._current_obs = obs_jax
        self._current_reward = 0.0
        self._current_done = False
        self._step_count = 0
        
        # Extract initial state
        unwrapped = self._gym_env
        while hasattr(unwrapped, 'env'):
            unwrapped = unwrapped.env
        
        agent_pos = jnp.array([
            int(unwrapped.agent_pos[0]),
            int(unwrapped.agent_pos[1])
        ])
        agent_dir = int(unwrapped.agent_dir)
        
        # Check for key position
        if hasattr(unwrapped, 'key_pos') and unwrapped.key_pos is not None:
            key_pos = jnp.array([
                int(unwrapped.key_pos[0]),
                int(unwrapped.key_pos[1])
            ])
        else:
            key_pos = jnp.array([-1, -1])
        
        door_unlocked = getattr(unwrapped, 'door_unlocked', True)
        
        # Create initial state
        state = MiniGridEnvState(
            key=rng,
            step_count=0,
            done=False,
            obs=obs_jax,
            agent_dir=agent_dir,
            agent_pos=agent_pos,
            achievements=jnp.zeros(2),
            key_pos=key_pos,
            has_key=False,
            door_unlocked=door_unlocked
        )
        
        return obs_jax, state

    def is_terminal(self, state: MiniGridEnvState, params: MiniGridEnvParams) -> bool:
        """Check if the state is terminal."""
        return bool(state.done)

    def discount(self, state: MiniGridEnvState, params: MiniGridEnvParams) -> float:
        """Return discount factor."""
        return jax.lax.select(state.done, 0.0, 1.0)

    def render(self, state: MiniGridEnvState) -> np.ndarray:
        """Render the environment."""
        return self._gym_env.render()

    def close(self):
        """Close the environment."""
        if hasattr(self, '_gym_env'):
            self._gym_env.close()


def make_wrapped_fixed_minigrid_jax_env(env_name: str = "MiniGrid-DoorKey-5x5-v0"):
    """Factory function to create a wrapped fixed MiniGrid JAX environment."""
    env = WrappedFixedMiniGridJaxEnv(env_name)
    env_params = env.default_params
    return env, env_params