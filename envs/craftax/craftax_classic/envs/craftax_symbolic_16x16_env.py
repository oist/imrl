import jax
from jax import lax
from gymnax.environments import spaces, environment
from typing import Tuple, Optional

import jax.numpy as jnp
from envs.craftax.craftax_classic.envs.common import compute_score
from envs.craftax.craftax_classic.game_logic import craftax_step, is_game_over
from envs.craftax.craftax_classic.envs.craftax_state import (
    EnvState,
    EnvParams,
    StaticEnvParams,
)
from envs.craftax.craftax_classic.renderer import render_craftax_symbolic
from envs.craftax.craftax_classic.world_gen import generate_world
from envs.craftax.craftax_classic.envs.craftax_symbolic_env import (
    get_flat_map_obs_shape,
    get_inventory_obs_shape,
)

class CraftaxClassicSymbolic16x16Env(environment.Environment):
    def render(self, state=None, *args, **kwargs):
        """Render the current state as an image (for video recording or visualization)."""
        # If state is not provided, raise error (since this env is functional)
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
        # Set map_size to 16x16
        return StaticEnvParams(map_size=(16, 16))

    def step_env(
        self, rng, state: EnvState, action: int, params: EnvParams
    ) -> Tuple:
        state, reward = craftax_step(rng, state, action, params, self.static_env_params)
        done = self.is_terminal(state, params)
        info = compute_score(state, done)
        # Tracer-safe propagation of termination reason
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
        self, rng, params: EnvParams
    ) -> Tuple:
        state = generate_world(rng, params, self.static_env_params)
        fixed_proto = getattr(self, '_fixed_landscape_state', None)
        use_fixed = getattr(self, '_use_fixed_landscape', False)
        if use_fixed and fixed_proto is not None:
            rng, _rng = jax.random.split(rng)
            state = fixed_proto.replace(state_rng=_rng, timestep=0, terminated_by_reward=False)
        else:
            state = generate_world(rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState):
        pixels = render_craftax_symbolic(state)
        return pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Symbolic-16x16-v1"

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
