"""
Pure JAX Vectorized Minigrid Environment Wrapper

This wrapper provides a lightweight vectorized environment using the
JAX-native `JAXMinigridFourRooms` implementation. It intentionally
avoids any dependency on `gymnasium` or `stable-baselines3`.

The API is deliberately small and compatible with common training loops:
- reset() -> np.ndarray of observations
- step_async(actions)
- step_wait() -> (obs, rewards, dones, infos)

It uses JAX vmapped reset/step for efficient batch execution and minimizes
host-device transfers by keeping most work in JAX.
"""

from typing import Tuple, List, Optional, Any
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import os

# Make sure async dispatch is enabled for maximal throughput
os.environ.setdefault('JAX_ENABLE_ASYNC_DISPATCH', '1')
os.environ.setdefault('JAX_CHECK_TRACER_LEAKS', '0')
os.environ.setdefault('JAX_DEBUG_NANS', '0')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', '/tmp/jax_cache')

from .jax_minigrid_env import JAXMinigridFourRooms, EnvParams


# Constants for one-hot encoding
NUM_OBJECTS = 11
NUM_COLORS = 8
NUM_STATES = 3


def _make_box_space(shape, low=0.0, high=1.0, dtype=np.float32):
    class Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype
    return Box(low, high, shape, dtype)


def _make_discrete_space(n):
    class Discrete:
        def __init__(self, n):
            self.n = n
    return Discrete(n)


@jax.jit
def one_hot_encode_grid_jax(grid: jnp.ndarray, direction: jnp.ndarray) -> jnp.ndarray:
    H, W, _ = grid.shape
    obj_ids = grid[:, :, 0].astype(jnp.int32)
    color_ids = grid[:, :, 1].astype(jnp.int32)
    state_ids = grid[:, :, 2].astype(jnp.int32)
    obj_encoded = jax.nn.one_hot(obj_ids, NUM_OBJECTS, dtype=jnp.float32)
    color_encoded = jax.nn.one_hot(color_ids, NUM_COLORS, dtype=jnp.float32)
    state_encoded = jax.nn.one_hot(state_ids, NUM_STATES, dtype=jnp.float32)
    encoded = jnp.concatenate([obj_encoded, color_encoded, state_encoded], axis=-1)
    flat_grid = encoded.reshape(-1)
    direction_normalized = direction / 3.0
    direction_val = jnp.asarray([direction_normalized], dtype=jnp.float32)
    mission_val = jnp.array([0.0], dtype=jnp.float32)
    return jnp.concatenate([flat_grid, direction_val, mission_val])


_v_one_hot_encode = jax.vmap(one_hot_encode_grid_jax, in_axes=(0, 0), out_axes=0)


class VectorizedJAXMinigrid:
    """Minimal vectorized wrapper around `JAXMinigridFourRooms`.

    This class intentionally does not depend on gymnasium or stable-baselines3.
    It provides a small, fast wrapper for training loops that expect batched
    numpy arrays.
    """

    def __init__(self, env_id: str = 'FourRooms', n_envs: int = 4, seed: Optional[int] = None, flatten: bool = True):
        self._single = JAXMinigridFourRooms()
        self.flatten = flatten
        self.env_id = env_id
        self.num_envs = int(n_envs)
        if seed is None:
            self._seed = int(np.random.randint(0, 2**31 - 1))
        else:
            self._seed = int(seed)
        self.rng_keys = None
        self._base_key = None

        obs_shape = self._single.observation_shape
        if self.flatten:
            flat_size = obs_shape[0] * obs_shape[1] * (NUM_OBJECTS + NUM_COLORS + NUM_STATES) + 2
            self.observation_space = _make_box_space((flat_size,), low=0.0, high=1.0, dtype=np.float32)
            self._obs_buffer = np.zeros((self.num_envs, flat_size), dtype=np.float32)
        else:
            self.observation_space = _make_box_space(obs_shape, low=0, high=255, dtype=np.uint8)
            self._obs_buffer = np.zeros((self.num_envs, *obs_shape), dtype=np.uint8)

        self.action_space = _make_discrete_space(self._single.num_actions)

        # Pre-compile vmapped functions
        def _reset_single(k):
            st, obs = self._single.reset(k)
            return st, obs

        def _step_single(st, a):
            st2, obs, reward, done, info = self._single.step(st, a)
            return st2, obs, reward, done, info

        self._v_reset = jax.jit(jax.vmap(_reset_single, in_axes=0, out_axes=0))
        self._v_step = jax.jit(jax.vmap(_step_single, in_axes=(0, 0), out_axes=0))

        def _conditional_reset(state, obs, done, key):
            new_state, new_obs = self._single.reset(key)
            selected_state = jax.tree_map(
                lambda new, old: jax.lax.select(done, new, old),
                new_state, state
            )
            selected_obs = jax.lax.select(done, new_obs, obs)
            return selected_state, selected_obs

        self._v_conditional_reset = jax.jit(jax.vmap(_conditional_reset, in_axes=(0, 0, 0, 0)))

        self._rewards_buffer = np.zeros(self.num_envs, dtype=np.float32)
        self._dones_buffer = np.zeros(self.num_envs, dtype=bool)
        self._episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self._state = None

    def reset(self) -> np.ndarray:
        if self._base_key is None:
            base_key = random.PRNGKey(int(self._seed))
            self.rng_keys = random.split(base_key, self.num_envs + 1)
            self._base_key = self.rng_keys[-1]
            self.rng_keys = self.rng_keys[:-1]
        else:
            self._base_key, *new_keys = random.split(self._base_key, self.num_envs + 1)
            self.rng_keys = jnp.array(new_keys)

        keys = self.rng_keys
        sts, obs = self._v_reset(keys)
        self._state = sts
        if self.flatten:
            directions = sts.agent_dir
            flat = _v_one_hot_encode(obs, directions)
            np.copyto(self._obs_buffer, flat)
            return self._obs_buffer
        else:
            np.copyto(self._obs_buffer, obs)
            return self._obs_buffer

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        acts_j = jnp.asarray(self._actions, dtype=jnp.int32)
        sts2, obs_arr_j, rewards_j, dones_j, infos_j = self._v_step(self._state, acts_j)
        done_mask = dones_j
        if jnp.any(done_mask):
            n_dones = jnp.sum(done_mask).item()
            if n_dones > 0:
                self._base_key, *new_keys = random.split(self._base_key, self.num_envs + 1)
                reset_keys = jnp.array(new_keys)
                sts2, obs_arr_j = self._v_conditional_reset(sts2, obs_arr_j, done_mask, reset_keys)

        self._state = sts2
        if self.flatten:
            directions = sts2.agent_dir
            flat_jax = _v_one_hot_encode(obs_arr_j, directions)
            np.copyto(self._obs_buffer, flat_jax)
            np.copyto(self._rewards_buffer, rewards_j)
            np.copyto(self._dones_buffer, dones_j)
        else:
            np.copyto(self._obs_buffer, obs_arr_j)
            np.copyto(self._rewards_buffer, rewards_j)
            np.copyto(self._dones_buffer, dones_j)

        infos = []
        for i in range(self.num_envs):
            self._episode_rewards[i] += self._rewards_buffer[i]
            self._episode_lengths[i] += 1
            info = {}
            if self._dones_buffer[i]:
                info['episode'] = {
                    'r': float(self._episode_rewards[i]),
                    'l': int(self._episode_lengths[i])
                }
                self._episode_rewards[i] = 0.0
                self._episode_lengths[i] = 0
            infos.append(info)

        return self._obs_buffer, self._rewards_buffer, self._dones_buffer, infos

    def close(self):
        pass

    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        target_envs = self._get_indices(indices)
        if attr_name == "render_mode":
            return [None] * len(target_envs)
        if hasattr(self._single, attr_name):
            attr_value = getattr(self._single, attr_name)
            return [attr_value] * len(target_envs)
        raise AttributeError(f"Attribute '{attr_name}' not found")

    def _get_indices(self, indices: Optional[List[int]]) -> List[int]:
        if indices is None:
            return list(range(self.num_envs))
        return indices

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        if seed is not None:
            self._seed = int(seed)
            base_key = random.PRNGKey(int(seed))
            self.rng_keys = random.split(base_key, self.num_envs)
        return [seed] * self.num_envs

    @property
    def unwrapped(self):
        return self._single

    @property
    def num_envs_property(self):
        return self.num_envs


# --- Adapter to expose Gymnax-compatible JAX API for the vectorized env ---
class VectorizedGymnaxAdapter:
    """Adapter that exposes a Gymnax-like JAX interface for VectorizedJAXMinigrid.

    Methods:
      - reset(rng, params=None) -> obs, state
      - step(rng, state, action, params=None) -> obs, state, reward, done, info

    This allows the rest of the training code (which expects env.reset/step to be
    JAX-jittable and vmappable) to work with the efficient vectorized implementation.
    """

    def __init__(self, env_id: str = 'FourRooms', n_envs: int = 4, seed: Optional[int] = None, flatten: bool = True):
        self._vec = VectorizedJAXMinigrid(env_id=env_id, n_envs=n_envs, seed=seed, flatten=flatten)
        self.num_envs = self._vec.num_envs
        self.observation_space = self._vec.observation_space
        self.action_space = self._vec.action_space
        # expose single env params if available
        if hasattr(self._vec._single, 'default_params'):
            self.default_params = self._vec._single.default_params
        else:
            self.default_params = None

    @property
    def _base_key(self):
        # small helper to generate base random keys on the fly
        return jax.random.PRNGKey(int(self._vec._seed))

    def reset(self, rng: jax.Array, params=None):
        """Reset all vectorized environments using a single rng key."""
        # Create per-env keys from provided rng
        rngs = jax.random.split(rng, self.num_envs)
        # Use the underlying vmapped reset function
        states, obs = self._vec._v_reset(rngs)
        # If flattening is enabled, encode observations via _v_one_hot_encode
        if self._vec.flatten:
            directions = states.agent_dir
            flat = _v_one_hot_encode(obs, directions)
            return flat, states
        else:
            return obs, states

    def step(self, rng: jax.Array, state, action: jnp.ndarray, params=None):
        """Step all vectorized environments.

        rng is used to generate reset keys for environments that finished.
        """
        # action should be a jnp.ndarray shaped (num_envs,)
        st2, obs_arr_j, rewards_j, dones_j, infos_j = self._vec._v_step(state, action)

        # If any environment finished, perform conditional reset
        done_mask = dones_j
        if jnp.any(done_mask):
            reset_keys = jax.random.split(rng, self.num_envs)
            st2, obs_arr_j = self._vec._v_conditional_reset(st2, obs_arr_j, done_mask, reset_keys)

        if self._vec.flatten:
            directions = st2.agent_dir
            flat_jax = _v_one_hot_encode(obs_arr_j, directions)
            return flat_jax, st2, rewards_j, dones_j, infos_j
        else:
            return obs_arr_j, st2, rewards_j, dones_j, infos_j

    def seed(self, seed: Optional[int] = None):
        return self._vec.seed(seed)

    def close(self):
        return self._vec.close()


# Provide a convenience factory so env_factory can import directly
def make_vectorized_minigrid(env_id: str = 'MiniGrid-FourRooms-TwoGoals-RandKey-ViewSize-3x3-v0', n_envs: int = 128, seed: Optional[int] = None, flatten: bool = True):
    return VectorizedGymnaxAdapter(env_id=env_id, n_envs=n_envs, seed=seed, flatten=flatten)
