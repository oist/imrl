import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import spaces, environment
from typing import Tuple, Optional

from envs.craftax.environment_base.environment_bases import EnvironmentNoAutoReset
from envs.craftax.craftax_classic.envs.common import compute_score
from envs.craftax.craftax_classic.constants import *
from envs.craftax.craftax_classic.game_logic import craftax_step, is_game_over
from envs.craftax.craftax_classic.envs.craftax_state import (
    EnvState,
    EnvParams,
    StaticEnvParams,
)
from envs.craftax.craftax_classic.renderer import render_craftax_symbolic
from envs.craftax.craftax_classic.world_gen import generate_world


def get_map_obs_shape():
    num_mobs = 4
    num_blocks = len(BlockType)

    return OBS_DIM[0], OBS_DIM[1], num_blocks + num_mobs


def get_flat_map_obs_shape():
    map_obs_shape = get_map_obs_shape()
    return map_obs_shape[0] * map_obs_shape[1] * map_obs_shape[2]


def get_inventory_obs_shape():
    inv_size = 12
    num_intrinsics = 4
    light_level = 1
    is_sleeping = 1
    direction = 4

    return inv_size + num_intrinsics + light_level + is_sleeping + direction


class CraftaxClassicSymbolicEnvNoAutoReset(EnvironmentNoAutoReset):
    def render(self, state=None, *args, **kwargs):
        """Render the current state as an image (for video recording or visualization)."""
        if state is None:
            raise ValueError("State must be provided to render().")
        return render_craftax_symbolic(state)
    def __init__(self, static_env_params: StaticEnvParams = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, EnvState, float, bool, dict]:
        state, reward = craftax_step(rng, state, action, params, self.static_env_params)

        done = self.is_terminal(state, params)
        info = compute_score(state, done)
        # Tracer-safe propagation of termination reason: store boolean and a small int code
        # (1 => achievement_reward, 0 => none). String reasons can't be JIT-traced.
        info['terminated_by_reward'] = state.terminated_by_reward
        info['termination_reason_code'] = jnp.where(state.terminated_by_reward, jnp.int32(1), jnp.int32(0))
        info["discount"] = self.discount(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        # If a fixed landscape state was pre-generated and attached to the env
        # instance (a plain Python attribute), use that prototype and update
        # ephemeral fields like RNG and timestep. This avoids branching on
        # tracer-valued params inside generate_world during JAX tracing.
        fixed_proto = getattr(self, '_fixed_landscape_state', None)
        use_fixed = getattr(self, '_use_fixed_landscape', False)
        if use_fixed and fixed_proto is not None:
            # split rng for state rng
            rng, _rng = jax.random.split(rng)
            state = fixed_proto.replace(state_rng=_rng, timestep=0, terminated_by_reward=False)
        else:
            state = generate_world(rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jax.Array:
        pixels = render_craftax_symbolic(state)
        return pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Symbolic-NoAutoReset-v1"

    @property
    def num_actions(self) -> int:
        return 16

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(16)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        flat_map_obs_shape = get_flat_map_obs_shape()
        inventory_obs_shape = get_inventory_obs_shape()

        obs_shape = flat_map_obs_shape + inventory_obs_shape

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )


class CraftaxClassicSymbolicEnv(environment.Environment):
    def render(self, state=None, *args, **kwargs):
        """Render the current state as an image (for video recording or visualization)."""
        if state is None:
            raise ValueError("State must be provided to render().")
        return render_craftax_symbolic(state)
    def __init__(self, static_env_params: StaticEnvParams = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, EnvState, float, bool, dict]:
        state, reward = craftax_step(rng, state, action, params, self.static_env_params)

        done = self.is_terminal(state, params)
        info = compute_score(state, done)
        info["discount"] = self.discount(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        # Use pre-generated fixed prototype state if available on the env
        fixed_proto = getattr(self, '_fixed_landscape_state', None)
        use_fixed = getattr(self, '_use_fixed_landscape', False)
        if use_fixed and fixed_proto is not None:
            rng, _rng = jax.random.split(rng)
            state = fixed_proto.replace(state_rng=_rng, timestep=0, terminated_by_reward=False)
        else:
            state = generate_world(rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jax.Array:
        pixels = render_craftax_symbolic(state)
        return pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Symbolic-v1"

    @property
    def num_actions(self) -> int:
        return 16

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(16)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        flat_map_obs_shape = get_flat_map_obs_shape()
        inventory_obs_shape = get_inventory_obs_shape()

        obs_shape = flat_map_obs_shape + inventory_obs_shape

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )
