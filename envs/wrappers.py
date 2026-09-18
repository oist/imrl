import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial
from typing import Union, Any


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jnp.where(done, x, y), state_re, state_st
            )
            obs = jnp.where(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jnp.where(done, x, y), state_re, state_st
            )
            obs = jnp.where(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: float  # Changed from int to float for dtype consistency
    returned_episode_returns: float
    returned_episode_lengths: float  # Changed from int to float for dtype consistency
    episode_achievements: jnp.ndarray  # Current episode achievements
    returned_episode_achievements: jnp.ndarray  # Achievements from last completed episode
    timestep: float  # Changed from int to float for dtype consistency
    episode_actions: jnp.ndarray  # Accumulated actions for current episode (fixed-size buffer)
    action_count: float  # Number of actions accumulated in current episode


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)
        # Detect number of achievements based on environment type
        env_name = str(type(env).__name__)
        if 'MiniGrid' in env_name or 'Minigrid' in env_name:
            # MiniGrid has 2 achievements: [large_goal, small_goal]
            self.num_achievements = 2
            self.achievement_names = ['large_goal', 'small_goal']
        else:
            # Craftax Classic has 22 achievements
            self.num_achievements = 22
            self.achievement_names = [
                'collect_wood', 'place_table', 'eat_cow', 'collect_sapling', 'collect_drink',
                'make_wood_pickaxe', 'make_wood_sword', 'place_plant', 'defeat_zombie', 'collect_stone',
                'place_stone', 'eat_plant', 'defeat_skeleton', 'make_stone_pickaxe', 'make_stone_sword',
                'wake_up', 'place_furnace', 'collect_coal', 'collect_iron', 'collect_diamond',
                'make_iron_pickaxe', 'make_iron_sword'
            ]

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        # Initialize achievement arrays
        zero_achievements = jnp.zeros(self.num_achievements)
        # Initialize action buffer (max 512 actions per episode - can be adjusted)
        max_episode_length = 512
        zero_actions = jnp.zeros(max_episode_length, dtype=jnp.int32)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=0.0,
            episode_lengths=0.0,  # Changed to float
            returned_episode_returns=0.0,
            returned_episode_lengths=0.0,  # Changed to float
            episode_achievements=zero_achievements,
            returned_episode_achievements=zero_achievements,
            timestep=0.0,  # Changed to float
            episode_actions=zero_actions,
            action_count=0.0,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1.0  # Ensure float arithmetic

        # Accumulate actions for the current episode
        action_idx = jnp.clip(state.action_count, 0, state.episode_actions.shape[0] - 1).astype(jnp.int32)
        updated_actions = state.episode_actions.at[action_idx].set(action)
        new_action_count = state.action_count + 1.0

        # Extract achievements directly from the environment state
        # When done=True, achievements contain the cumulative achievements for the episode
        current_achievements = jnp.where(
            done,
            env_state.achievements,  # Use achievements from env_state when episode is done
            jnp.zeros(self.num_achievements),  # Zero otherwise
        )

        # When episode completes, extract the action sequence
        # Only extract up to action_count to avoid trailing zeros
        completed_actions = jnp.where(
            done,
            updated_actions,  # Full buffer with this episode's actions
            state.episode_actions,  # Keep previous
        )

        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            episode_achievements=env_state.achievements * (1 - done),  # Current episode achievements
            returned_episode_achievements=(current_achievements * done
                                           + state.returned_episode_achievements * (1 - done)),
            timestep=state.timestep + 1.0,  # Ensure float arithmetic
            episode_actions=updated_actions * (1 - done),  # Reset when episode ends
            action_count=new_action_count * (1 - done),  # Reset when episode ends
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode_achievements"] = state.returned_episode_achievements
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        # Also expose current (running) episode returns and lengths per environment
        info["episode_returns"] = state.episode_returns
        info["episode_lengths"] = state.episode_lengths
        # Include the full action sequence when episode completes
        info["episode_actions"] = completed_actions
        info["episode_action_count"] = new_action_count  # How many actions are valid

        return obs, state, reward, done, info
