"""
JAX-Native MiniGrid FourRooms Environment (ported copy)

This file is a copy of the JAX-native MiniGrid implementation taken from
the RL-HParams-Optim repository. It provides JIT-able reset/step and
batched reset/step for GPU-accelerated training.

Note: The implementation is independent from gymnasium and is pure JAX.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from functools import partial
from typing import Tuple, Dict, Any, Optional
import numpy as np
from flax import struct
import chex


# Constants matching original MiniGrid
OBJECT_TO_IDX = {
    'unseen': 0,
    'empty': 1,
    'wall': 2,
    'floor': 3,
    'door': 4,
    'key': 5,
    'ball': 6,
    'box': 7,
    'goal': 8,
    'lava': 9,
    'agent': 10,
}

COLOR_TO_IDX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'purple': 3,
    'yellow': 4,
    'grey': 5,
    'emeraldgreen': 6,
    'neongreen': 7,
}

# Actions
ACTIONS = {
    'left': 0,
    'right': 1,
    'forward': 2,
    'pickup': 3,
    'drop': 4,
    'toggle': 5,
    'done': 6,
}

# Default goal positions for FourRooms-TwoGoals (stored as NumPy, converted to JAX when needed)
# Positions are in (x, y) format matching Gymnasium convention
DEFAULT_GOAL1_POS = np.array([1, 11], dtype=np.int32)  # Goal at (x=1, y=11)
DEFAULT_GOAL2_POS = np.array([11, 11], dtype=np.int32)  # Goal2 at (x=11, y=11)
DEFAULT_DOOR_POS = np.array([6, 3], dtype=np.int32)  # Door at (x=6, y=3)

# Directions
DIR_TO_VEC = np.array([
    [1, 0],   # right
    [0, 1],   # down
    [-1, 0],  # left
    [0, -1],  # up
], dtype=int)


@struct.dataclass
class EnvState:
    agent_pos: chex.Array
    agent_dir: int
    key_pos: chex.Array
    key_picked: bool
    door_open: bool
    grid: chex.Array
    step_count: int
    done: bool
    rng_key: chex.PRNGKey
    achievements: chex.Array


@struct.dataclass
class EnvParams:
    width: int = 13
    height: int = 13
    agent_view_size: int = 3
    max_steps: int = 1000
    goal1_pos: tuple = (1, 11)
    goal2_pos: tuple = (11, 11)
    door_pos: tuple = (6, 3)
    terminate_on_reward: bool = True  # Terminate episode when reaching goal (True=goal-based, False=open-ended)


class JAXMinigridFourRooms:
    def __init__(self, params: Optional[EnvParams] = None):
        base_params = params if params is not None else EnvParams()
        self.params = base_params.replace(
            goal1_pos=jnp.array(base_params.goal1_pos, dtype=jnp.int32),
            goal2_pos=jnp.array(base_params.goal2_pos, dtype=jnp.int32),
            door_pos=jnp.array(base_params.door_pos, dtype=jnp.int32)
        )
        self.dir_to_vec = DIR_TO_VEC
        self.goal1_pos_jax = jnp.array(base_params.goal1_pos, dtype=jnp.int32)
        self.goal2_pos_jax = jnp.array(base_params.goal2_pos, dtype=jnp.int32)
        self.goal1_pos = np.array(base_params.goal1_pos, dtype=int)
        self.goal2_pos = np.array(base_params.goal2_pos, dtype=int)
        self.door_pos = np.array(base_params.door_pos, dtype=int)
        self.observation_shape = (
            self.params.agent_view_size,
            self.params.agent_view_size,
            3
        )
        self.action_space_size = 7

    @property
    def num_actions(self) -> int:
        return self.action_space_size

    def _create_base_grid(self) -> chex.Array:
        grid = jnp.zeros((self.params.height, self.params.width, 3), dtype=jnp.int32)
        grid = grid.at[:, :, 0].set(OBJECT_TO_IDX['empty'])
        grid = grid.at[0, :, 0].set(OBJECT_TO_IDX['wall'])
        grid = grid.at[-1, :, 0].set(OBJECT_TO_IDX['wall'])
        grid = grid.at[:, 0, 0].set(OBJECT_TO_IDX['wall'])
        grid = grid.at[:, -1, 0].set(OBJECT_TO_IDX['wall'])
        grid = grid.at[0, :, 1].set(COLOR_TO_IDX['grey'])
        grid = grid.at[-1, :, 1].set(COLOR_TO_IDX['grey'])
        grid = grid.at[:, 0, 1].set(COLOR_TO_IDX['grey'])
        grid = grid.at[:, -1, 1].set(COLOR_TO_IDX['grey'])
        grid = grid.at[1:12, 6, 0].set(OBJECT_TO_IDX['wall'])
        grid = grid.at[1:12, 6, 1].set(COLOR_TO_IDX['grey'])
        grid = grid.at[6, 1:12, 0].set(OBJECT_TO_IDX['wall'])
        grid = grid.at[6, 1:12, 1].set(COLOR_TO_IDX['grey'])
        grid = grid.at[3, 6, 0].set(OBJECT_TO_IDX['door'])
        grid = grid.at[3, 6, 1].set(COLOR_TO_IDX['yellow'])
        grid = grid.at[3, 6, 2].set(0)
        grid = grid.at[6, 3, 0].set(OBJECT_TO_IDX['empty'])
        grid = grid.at[6, 9, 0].set(OBJECT_TO_IDX['empty'])
        g1x, g1y = int(self.goal1_pos[0]), int(self.goal1_pos[1])
        g2x, g2y = int(self.goal2_pos[0]), int(self.goal2_pos[1])
        grid = grid.at[g1y, g1x, 0].set(OBJECT_TO_IDX['goal'])
        grid = grid.at[g1y, g1x, 1].set(COLOR_TO_IDX['emeraldgreen'])
        grid = grid.at[g2y, g2x, 0].set(OBJECT_TO_IDX['goal'])
        grid = grid.at[g2y, g2x, 1].set(COLOR_TO_IDX['neongreen'])
        return grid

    @partial(jit, static_argnums=(0,))
    def reset(self, rng_key: chex.PRNGKey) -> Tuple[EnvState, chex.Array]:
        key1, key2, key3 = random.split(rng_key, 3)
        grid = self._create_base_grid()
        # Generate key position in range (1, 6)
        # Use modulo to ensure we avoid agent spawn position at (1, 1)
        key_x = random.randint(key1, (), 1, 6)
        key_y = random.randint(key2, (), 1, 6)
        # Ensure key doesn't spawn on agent position (1, 1)
        # If it does, shift it by 1 (wrapping within valid range)
        agent_pos = jnp.array([1, 1])
        key_on_agent = (key_x == 1) & (key_y == 1)
        key_x = jnp.where(key_on_agent, 2, key_x)
        key_pos = jnp.array([key_x, key_y])
        grid = grid.at[key_y, key_x, 0].set(OBJECT_TO_IDX['key'])
        grid = grid.at[key_y, key_x, 1].set(COLOR_TO_IDX['yellow'])
        agent_dir = 0
        state = EnvState(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            key_pos=key_pos,
            key_picked=False,
            door_open=False,
            grid=grid,
            step_count=0,
            done=False,
            rng_key=key3,
            achievements=jnp.zeros(2, dtype=jnp.int32),  # [large_goal, small_goal]
        )
        obs = self._get_observation(state)
        return state, obs

    @partial(jit, static_argnums=(0,))
    def step(self, state: EnvState, action: int) -> Tuple[EnvState, chex.Array, float, bool, Dict]:
        state = state.replace(step_count=state.step_count + 1)
        state, reward = self._process_action(state, action)
        done = state.done | (state.step_count >= self.params.max_steps)
        state = state.replace(done=done)
        obs = self._get_observation(state)
        info = {'step_count': state.step_count}
        return state, obs, reward, done, info

    def _process_action(self, state: EnvState, action: int) -> Tuple[EnvState, float]:
        reward = 0.0
        state = jax.lax.cond(
            action == ACTIONS['left'],
            lambda s: s.replace(agent_dir=(s.agent_dir - 1) % 4),
            lambda s: s,
            state
        )
        state = jax.lax.cond(
            action == ACTIONS['right'],
            lambda s: s.replace(agent_dir=(s.agent_dir + 1) % 4),
            lambda s: s,
            state
        )
        dir_vecs = jnp.array(self.dir_to_vec)

        def move_forward(s):
            dir_vec = jnp.take(dir_vecs, s.agent_dir, axis=0)
            new_pos = s.agent_pos + dir_vec
            cell = s.grid[new_pos[1], new_pos[0]]
            is_wall = cell[0] == OBJECT_TO_IDX['wall']
            is_closed_door = (cell[0] == OBJECT_TO_IDX['door']) & (cell[2] == 0) & (~s.door_open)
            can_move = ~is_wall & ~is_closed_door
            final_pos = jnp.where(can_move, new_pos, s.agent_pos)
            return s.replace(agent_pos=final_pos)

        state = jax.lax.cond(
            action == ACTIONS['forward'],
            move_forward,
            lambda s: s,
            state
        )

        def pickup_key(s):
            on_key = jnp.all(s.agent_pos == s.key_pos)
            can_pickup = on_key & (~s.key_picked)
            grid = jnp.where(
                can_pickup,
                s.grid.at[s.key_pos[1], s.key_pos[0], 0].set(OBJECT_TO_IDX['empty']),
                s.grid
            )
            return s.replace(key_picked=s.key_picked | can_pickup, grid=grid)

        state = jax.lax.cond(
            action == ACTIONS['pickup'],
            pickup_key,
            lambda s: s,
            state
        )

        def toggle_door(s):
            door_pos = self.door_pos
            dist = jnp.abs(s.agent_pos - door_pos).sum()
            next_to_door = dist == 1
            can_toggle = next_to_door & s.key_picked & (~s.door_open)
            grid = jnp.where(
                can_toggle,
                s.grid.at[door_pos[1], door_pos[0], 2].set(1),
                s.grid
            )
            return s.replace(door_open=s.door_open | can_toggle, grid=grid)

        state = jax.lax.cond(
            action == ACTIONS['toggle'],
            toggle_door,
            lambda s: s,
            state
        )

        goal1_pos_jax = jnp.array(DEFAULT_GOAL1_POS, dtype=jnp.int32)
        goal2_pos_jax = jnp.array(DEFAULT_GOAL2_POS, dtype=jnp.int32)
        on_goal1 = jnp.all(state.agent_pos == goal1_pos_jax)
        on_goal2 = jnp.all(state.agent_pos == goal2_pos_jax)
        base_reward = 1.0 - 0.9 * (state.step_count / self.params.max_steps)
        small_goal_reward = base_reward
        large_goal_reward = base_reward * 10.0
        reward = jnp.where(on_goal1, small_goal_reward, reward)
        reward = jnp.where(on_goal2, large_goal_reward, reward)
        
        # Update achievements: [0]=large_goal, [1]=small_goal
        new_achievements = state.achievements.at[0].add(jax.lax.select(on_goal2, 1, 0))
        new_achievements = new_achievements.at[1].add(jax.lax.select(on_goal1, 1, 0))
        
        # Only terminate on goal if terminate_on_reward is True
        # If False, agent can continue exploring and potentially reach both goals
        goal_reached = on_goal1 | on_goal2
        should_terminate = jnp.where(self.params.terminate_on_reward, goal_reached, False)
        state = state.replace(done=should_terminate, achievements=new_achievements)
        return state, reward

    def _get_observation(self, state: EnvState) -> chex.Array:
        view_size = self.params.agent_view_size
        grid_size = self.params.width
        agent_x, agent_y = state.agent_pos[0], state.agent_pos[1]
        half_view = view_size // 2
        x_start = jnp.maximum(0, agent_x - half_view)
        x_end = jnp.minimum(grid_size, agent_x + half_view + 1)
        y_start = jnp.maximum(0, agent_y - half_view)
        y_end = jnp.minimum(grid_size, agent_y + half_view + 1)
        view = jax.lax.dynamic_slice(
            state.grid,
            (y_start, x_start, 0),
            (view_size, view_size, 3)
        )
        return view

    @partial(jit, static_argnums=(0, 2))
    def batch_reset(self, rng_key: chex.PRNGKey, batch_size: int) -> Tuple[EnvState, chex.Array]:
        keys = random.split(rng_key, batch_size)
        reset_fn = lambda key: self.reset(key)
        states, obs = vmap(reset_fn)(keys)
        return states, obs

    @partial(jit, static_argnums=(0,))
    def batch_step(self, states: EnvState, actions: chex.Array) -> Tuple[EnvState, chex.Array, chex.Array, chex.Array, Dict]:
        step_fn = lambda s, a: self.step(s, a)
        next_states, obs, rewards, dones, infos = vmap(step_fn)(states, actions)
        return next_states, obs, rewards, dones, infos

    def render(self, state, mode='rgb_array', tile_size=32):
        """
        Render the environment using MiniGrid's native Grid rendering.
        
        Args:
            state: Environment state to render (EnvState or PyTree)
            mode: Rendering mode (only 'rgb_array' supported)
            tile_size: Size of each tile in pixels
            
        Returns:
            RGB array of the rendered environment
        """
        if mode != 'rgb_array':
            raise ValueError(f"Unsupported render mode: {mode}")
        
        # Handle both dataclass and PyTree forms of state
        # When JAX traces the state, it becomes a PyTree without attribute access
        if hasattr(state, 'grid'):
            # Direct dataclass access
            grid_array = state.grid
            agent_pos = state.agent_pos
            agent_dir = state.agent_dir
        elif hasattr(state, '__dict__') and 'grid' in state.__dict__:
            # Dictionary-based access
            grid_array = state.__dict__['grid']
            agent_pos = state.__dict__['agent_pos']
            agent_dir = state.__dict__['agent_dir']
        else:
            # Assume it's a flax struct/PyTree - try to reconstruct
            # Convert to EnvState if it's a PyTree
            try:
                state = EnvState(*jax.tree_util.tree_leaves(state))
                grid_array = state.grid
                agent_pos = state.agent_pos
                agent_dir = state.agent_dir
            except:
                raise TypeError(f"Cannot extract grid from state of type {type(state)}. State must be an EnvState object or compatible PyTree.")
        
        try:
            from minigrid.core.grid import Grid
            from minigrid.core.world_object import Wall, Door, Key, Goal
            
            # Convert JAX state to MiniGrid Grid
            grid_np = np.array(grid_array)  # (H, W, 3)
            height, width = grid_np.shape[0], grid_np.shape[1]
            
            # Create MiniGrid Grid object
            grid = Grid(width, height)
            
            # Populate grid with objects based on state
            for y in range(height):
                for x in range(width):
                    obj_type = int(grid_np[y, x, 0])
                    color_idx = int(grid_np[y, x, 1])
                    
                    # Map color index to color name
                    color_map = {v: k for k, v in COLOR_TO_IDX.items()}
                    color = color_map.get(color_idx, 'grey')
                    
                    # Create appropriate object
                    if obj_type == OBJECT_TO_IDX['wall']:
                        grid.set(x, y, Wall())
                    elif obj_type == OBJECT_TO_IDX['door']:
                        # For door state, we need to check if we still have access to door_open
                        if hasattr(state, 'door_open'):
                            is_locked = not state.door_open
                        else:
                            # Fallback: assume locked if we can't access state
                            is_locked = True
                        door = Door(color, is_locked=is_locked)
                        grid.set(x, y, door)
                    elif obj_type == OBJECT_TO_IDX['key']:
                        grid.set(x, y, Key(color))
                    elif obj_type == OBJECT_TO_IDX['goal']:
                        grid.set(x, y, Goal())
            
            # Render the grid
            agent_pos_np = np.array(agent_pos)
            agent_dir_np = int(agent_dir)
            
            img = grid.render(
                tile_size=tile_size,
                agent_pos=tuple(agent_pos_np),
                agent_dir=agent_dir_np,
                highlight_mask=None
            )
            
            return img
            
        except ImportError:
            # Fallback if MiniGrid is not available
            raise ImportError("MiniGrid library required for rendering. Install with: pip install minigrid")


def make_jax_fourrooms(seed: Optional[int] = None) -> JAXMinigridFourRooms:
    env = JAXMinigridFourRooms()
    return env


if __name__ == "__main__":
    print("Testing JAX MiniGrid FourRooms environment...")
    env = make_jax_fourrooms()
    key = random.PRNGKey(0)
    state, obs = env.reset(key)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space size: {env.num_actions}")
    state, obs, reward, done, info = env.step(state, ACTIONS['forward'])
    print(f"After step - reward: {reward}, done: {done}")
    print("\nTesting batched environments...")
    batch_size = 16
    states, obs_batch = env.batch_reset(key, batch_size)
    print(f"Batch observation shape: {obs_batch.shape}")
    actions = jnp.zeros(batch_size, dtype=jnp.int32)
    states, obs_batch, rewards, dones, infos = env.batch_step(states, actions)
    print(f"Batch rewards shape: {rewards.shape}")
    print("\n✓ JAX environment tests passed!")
