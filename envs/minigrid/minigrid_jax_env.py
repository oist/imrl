import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Any, Dict
from gymnax.environments import spaces, environment
from flax import struct
from gymnasium.envs.registration import make
from minigrid.wrappers import OneHotPartialObsWrapper
from envs.minigrid.minigrid_wrappers import DictToArrayObsWrapper
from envs.minigrid.minigrid_observation_generator import generate_fourrooms_observation
# Import to register custom environments
import envs.minigrid.minigrid_envs  # noqa: F401


@struct.dataclass
class MiniGridEnvState:
    """JAX-compatible state structure for MiniGrid environments.
    
    Uses optimistic reset approach - only stores what the agent can observe.
    The agent only knows its direction and gets local observations.
    """
    key: jax.Array  # Random key for this environment instance
    step_count: int
    done: bool
    obs: jax.Array  # Current local observation from MiniGrid
    agent_dir: int  # Agent direction (0=right, 1=down, 2=left, 3=up)
    agent_pos: jax.Array  # Agent position (x, y) for goal detection
    achievements: jax.Array  # Achievement tracking for compatibility
    # Key-related state for RandKey environments
    key_pos: jax.Array  # Key position (x, y) - (-1, -1) if collected
    has_key: bool  # Whether agent has collected the key
    door_unlocked: bool  # Whether the door has been unlocked


@struct.dataclass
class MiniGridEnvParams:
    """JAX-compatible parameters for MiniGrid environments."""
    max_steps: int = 1000
    goal1_pos: Tuple[int, int] = (1, 11)  # Goal1 (emerald green) - small reward goal
    goal2_pos: Tuple[int, int] = (11, 11)  # Goal2 (neon green) - large reward goal (×10)
    size: int = 13  # Grid size for the environment


class MiniGridJAXWrapper(environment.Environment):
    """JAX-compatible wrapper for MiniGrid environments using optimistic resets.
    
    This wrapper makes MiniGrid environments compatible with JAX by:
    1. Using optimistic resets for efficient vectorization like Craftax
    2. Leveraging actual MiniGrid environment for proper observations and rewards
    3. Caching environment states for fast stepping
    4. Using JAX callbacks only during reset to get initial true state
    
    The approach:
    - Reset: Use real MiniGrid environment to get initial state and observation
    - Step: Use cached state with minimal callbacks for performance
    - Maintain compatibility with existing MiniGrid environment definitions
    """
    
    def __init__(self, env_id: str):
        self.env_id = env_id
        self._obs_shape = None
        self._action_dim = 7  # MiniGrid standard action space
        self._max_episode_steps = 1000
        self._agent_view_size = 3  # Default, will be updated from env
        self._init_env_metadata()
        
    def _init_env_metadata(self):
        """Initialize environment metadata by creating a temporary environment instance."""
        env = make(self.env_id)
        env = OneHotPartialObsWrapper(env)
        env = DictToArrayObsWrapper(env)
        
        # Get observation and action space info
        obs, _ = env.reset()
        obs_array = jnp.array(obs, dtype=jnp.float32)
        # Ensure observation is at least 1D
        if obs_array.ndim == 0:
            obs_array = obs_array.reshape(1)
        self._obs_shape = obs_array.shape
        self._action_dim = env.action_space.n
        
        # Get max episode steps from the environment
        if hasattr(env, 'spec') and env.spec and hasattr(env.spec, 'max_episode_steps'):
            self._max_episode_steps = env.spec.max_episode_steps
        else:
            # Try to get from unwrapped environment
            unwrapped_env = env
            while hasattr(unwrapped_env, 'env'):
                unwrapped_env = unwrapped_env.env
            if hasattr(unwrapped_env, 'max_steps'):
                self._max_episode_steps = unwrapped_env.max_steps
            else:
                self._max_episode_steps = 1000
        
        if self._max_episode_steps is None:
            self._max_episode_steps = 1000
        
        # Extract agent_view_size from the unwrapped environment
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        if hasattr(unwrapped_env, 'agent_view_size'):
            self._agent_view_size = unwrapped_env.agent_view_size
        else:
            # Parse from env_id if available (e.g., ViewSize-3x3)
            if 'ViewSize-' in self.env_id:
                try:
                    view_str = self.env_id.split('ViewSize-')[1].split('x')[0]
                    self._agent_view_size = int(view_str)
                except:
                    self._agent_view_size = 3  # Default fallback
            else:
                self._agent_view_size = 7  # Standard MiniGrid default
            
        env.close()
        
        print(f"MiniGrid env initialized - obs shape: {self._obs_shape}, obs dtype: {obs_array.dtype}")
        print(f"Action space: {self._action_dim}")
        print(f"Agent view size: {self._agent_view_size}")
        
        env.close()  # Clean up the temporary environment
    
    def _create_wrapped_env(self):
        """Create a wrapped MiniGrid environment instance."""
        env = make(self.env_id)
        env = OneHotPartialObsWrapper(env)
        env = DictToArrayObsWrapper(env)
        return env
        

    
    @property
    def default_params(self) -> MiniGridEnvParams:
        return MiniGridEnvParams(max_steps=self._max_episode_steps, size=13)

    def step(
        self,
        key: jax.Array,
        state: MiniGridEnvState,
        action: int,
        params: Optional[MiniGridEnvParams] = None,
    ) -> Tuple[jax.Array, MiniGridEnvState, float, bool, Dict[str, Any]]:
        """Override step to disable auto-reset and preserve achievements."""
        if params is None:
            params = self.default_params
        # Call step_env without auto-reset logic
        obs, state, reward, done, info = self.step_env(key, state, action, params)
        return obs, state, reward, done, info

    def _is_valid_position(self, pos: jax.Array) -> jax.Array:
        """Check if a position is valid in the Four Rooms environment.
        
        This matches the exact layout from the viewer's _draw_four_rooms_grid:
        - 13x13 grid with walls around the perimeter (x=0,12 and y=0,12)
        - Vertical wall at x=6 with door at (6,3)  
        - Horizontal wall at y=6 with openings at x=3 and x=9
        - Floor everywhere else
        """
        x, y = pos[0], pos[1]
        
        # Check boundaries - outer walls at edges (x=0,12 and y=0,12)
        is_outer_wall = (x == 0) | (x == 12) | (y == 0) | (y == 12)
        
        # Vertical wall at x=6 (except door position at y=3)
        is_vertical_wall = (x == 6) & (y != 3)
        
        # Horizontal wall at y=6 (except openings at x=3 and x=9)
        is_horizontal_wall = (y == 6) & (x != 3) & (x != 9)
        
        # Position is valid if it's not a wall
        return ~(is_outer_wall | is_vertical_wall | is_horizontal_wall)
    
    def step_env(
        self, rng: jax.Array, state: MiniGridEnvState, action: int, params: MiniGridEnvParams
    ) -> Tuple[jax.Array, MiniGridEnvState, float, bool, Dict[str, Any]]:
        """Take a step in the environment using optimistic simulation."""
        # Early termination checks
        done_already = state.done
        done_max_steps = state.step_count >= params.max_steps

        done_by_action = (action == 6)  # Explicit DONE action
        should_terminate = done_already | done_max_steps | done_by_action

        def return_done_state():
            # If episode ended (by DONE action or already done), compute sparse goal reward
            goal1_pos_array = jnp.array(params.goal1_pos)
            goal2_pos_array = jnp.array(params.goal2_pos)
            at_goal1 = jnp.all(state.agent_pos == goal1_pos_array)
            at_goal2 = jnp.all(state.agent_pos == goal2_pos_array)
            goal1_reward = jax.lax.select(at_goal1, 1.0, 0.0)
            goal2_reward = jax.lax.select(at_goal2, 10.0, 0.0)
            reward = goal1_reward + goal2_reward
            achievement = jax.lax.select(at_goal1 | at_goal2, 1, 0)
            
            # Update achievements when episode ends at a goal
            final_achievements = state.achievements.at[0].add(jax.lax.select(at_goal2, 1, 0))  # Large goal
            final_achievements = final_achievements.at[1].add(jax.lax.select(at_goal1, 1, 0))  # Small goal
            
            info = {
                "discount": 0.0,
                "achievement": achievement,
                "large_reward_goal": jax.lax.select(at_goal2, 1, 0),
                "small_reward_goal": jax.lax.select(at_goal1, 1, 0)
            }
            done_state = MiniGridEnvState(
                key=state.key,
                step_count=state.step_count,
                done=jnp.bool_(True),
                obs=state.obs,
                agent_dir=state.agent_dir,
                agent_pos=state.agent_pos,
                achievements=final_achievements,
                key_pos=state.key_pos,
                has_key=state.has_key,
                door_unlocked=state.door_unlocked
            )
            return state.obs, done_state, reward, True, info

        def continue_episode():
            # Optimistic step - update agent state based on action
            new_step_count = state.step_count + 1
            
            # Update agent direction based on action
            new_agent_dir = jax.lax.select(
                action == 0,  # Turn left
                (state.agent_dir - 1) % 4,
                jax.lax.select(
                    action == 1,  # Turn right  
                    (state.agent_dir + 1) % 4,
                    state.agent_dir  # All other actions keep same direction
                )
            )
            
            # Update agent position based on forward movement
            # Direction vectors: 0=right(1,0), 1=down(0,1), 2=left(-1,0), 3=up(0,-1)
            dir_vectors = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
            dir_vec = dir_vectors[new_agent_dir]
            
            # Calculate proposed new position
            proposed_pos = state.agent_pos + dir_vec
            
            # For RandKey environments, door at (6,3) is locked unless key is collected
            is_door_position = (proposed_pos[0] == 6) & (proposed_pos[1] == 3)
            door_accessible = state.door_unlocked | ~is_door_position
            
            # Can't move into a cell that contains the key (unless picking it up)
            key_blocking = jnp.all(proposed_pos == state.key_pos) & (state.key_pos[0] >= 0)
            
            # Check if the proposed position is valid (not a wall), accessible, and not blocked by objects
            valid_move = self._is_valid_position(proposed_pos) & door_accessible & ~key_blocking
            
            # Only move if it's a forward action and the move is valid
            new_agent_pos = jax.lax.select(
                (action == 2) & valid_move,  # Forward action and valid move
                proposed_pos,
                state.agent_pos  # Stay in place if invalid or not forward action
            )
            
            # Handle key pickup mechanics (action 3 = pickup)
            # In MiniGrid, agent picks up objects in the cell they're facing, not the cell they're in
            # Calculate the cell the agent is facing
            dir_vectors = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
            facing_dir_vec = dir_vectors[new_agent_dir]
            facing_pos = new_agent_pos + facing_dir_vec
            
            # Key is available if it exists and agent doesn't already have it
            key_available = ~state.has_key & (state.key_pos[0] >= 0)  # Key hasn't been collected and exists
            at_key_position = jnp.all(facing_pos == state.key_pos)
            pickup_action = (action == 3)
            
            # Pick up key if facing the key position and using pickup action
            key_collected_now = pickup_action & at_key_position & key_available
            new_has_key = jnp.bool_(state.has_key | key_collected_now)
            
            # Remove key from map when collected (set to invalid position)
            new_key_pos = jax.lax.select(
                key_collected_now,
                jnp.array([-1, -1], dtype=jnp.int32),  # Invalid position indicates collected
                state.key_pos
            )
            
            # Handle key drop mechanics (action 4 = drop)
            drop_action = (action == 4)
            can_drop = new_has_key & drop_action
            
            # Drop key in the cell the agent is facing (if empty and valid)
            drop_pos = facing_pos
            drop_pos_valid = self._is_valid_position(drop_pos)
            
            # Only drop if the drop position is valid and not already occupied
            key_dropped_now = can_drop & drop_pos_valid
            
            # Update key possession and position
            new_has_key = jnp.bool_(new_has_key & ~key_dropped_now)  # Lose key if dropped
            new_key_pos = jax.lax.select(
                key_dropped_now,
                drop_pos,  # Place key at drop position
                new_key_pos  # Keep existing key position
            )
            
            # Handle door unlocking (action 5 = toggle)
            # Agent must be facing the door, not standing on it
            facing_door = jnp.all(facing_pos == jnp.array([6, 3]))
            toggle_action = (action == 5)
            can_unlock = new_has_key & facing_door & toggle_action & ~state.door_unlocked
            
            new_door_unlocked = jnp.bool_(state.door_unlocked | can_unlock)
            
            # Goal detection and sparse reward calculation
            # Reward only when reaching a goal: small goal = 1.0, large goal = 10.0
            goal1_pos_array = jnp.array(params.goal1_pos)
            goal2_pos_array = jnp.array(params.goal2_pos)

            at_goal1 = jnp.all(new_agent_pos == goal1_pos_array)
            at_goal2 = jnp.all(new_agent_pos == goal2_pos_array)

            goal1_reward = jax.lax.select(at_goal1, 1.0, 0.0)  # Small goal reward
            goal2_reward = jax.lax.select(at_goal2, 10.0, 0.0)  # Large goal reward
            goal_reward = goal1_reward + goal2_reward

            # No per-step base reward to avoid inflated cumulative returns
            reward = goal_reward
            
            # Achievement tracking
            achievement = jax.lax.select(at_goal1 | at_goal2, 1, 0)
            
            # Episode terminates when reaching goals or max steps
            goal_done = at_goal1 | at_goal2
            done = goal_done | (new_step_count >= params.max_steps)
            
            # Update achievements
            new_achievements = state.achievements.at[0].add(jax.lax.select(at_goal2, 1, 0))  # Large goal
            new_achievements = new_achievements.at[1].add(jax.lax.select(at_goal1, 1, 0))  # Small goal
            
            # Generate proper MiniGrid observation from current state
            rng_key, _ = jax.random.split(rng)
            new_obs = generate_fourrooms_observation(
                agent_pos=new_agent_pos,
                agent_dir=new_agent_dir,
                has_key=new_has_key,
                door_unlocked=new_door_unlocked,
                key_pos=new_key_pos,
                goal1_pos=params.goal1_pos,
                goal2_pos=params.goal2_pos,
                size=params.size
            )
            
            new_state = MiniGridEnvState(
                key=rng_key,
                step_count=new_step_count,
                done=done,
                obs=new_obs,
                agent_dir=new_agent_dir,
                agent_pos=new_agent_pos,
                achievements=new_achievements,
                key_pos=new_key_pos,
                has_key=new_has_key,
                door_unlocked=new_door_unlocked
            )
            
            info = {
                "discount": jax.lax.select(done, 0.0, 1.0),
                "achievement": achievement,
                "large_reward_goal": jax.lax.select(at_goal2, 1, 0),
                "small_reward_goal": jax.lax.select(at_goal1, 1, 0)
            }
            
            return new_obs, new_state, reward, done, info
        
        return jax.lax.cond(should_terminate, return_done_state, continue_episode)
    
    def reset_env(
        self, rng: jax.Array, params: MiniGridEnvParams
    ) -> Tuple[jax.Array, MiniGridEnvState]:
        """Reset the environment using pure JAX operations for maximum speed."""
        # Agent starts at position (1, 1) in the top-left room, facing right
        agent_pos = jnp.array([1, 1], dtype=jnp.int32)
        
        # Initial direction: 0=right (consistent starting direction)
        agent_dir = jnp.int32(0)
        
        # Determine initial door state based on environment variant
        # "RandKey" variant: door starts LOCKED (needs key to unlock)
        # "Rand" variant: door starts UNLOCKED (no key mechanic)
        initial_door_unlocked = "Rand-ViewSize" in self.env_id and "RandKey" not in self.env_id
        
        # Generate random key position in top-left room only (matching original MiniGrid environment)
        # From minigrid_envs.py: pos_key = self.generate_random_position((1, 6), (1, 6))
        # This means x in range [1, 5] and y in range [1, 5] (top-left room only)
        rng_key, subkey = jax.random.split(rng)
        key_x = jax.random.randint(subkey, (), 1, 6, dtype=jnp.int32)  # [1, 5] inclusive
        rng_key, subkey = jax.random.split(rng_key)
        key_y = jax.random.randint(subkey, (), 1, 6, dtype=jnp.int32)  # [1, 5] inclusive
        
        # Ensure key doesn't spawn on agent position (agent starts at (1,1))
        proposed_key_pos = jnp.array([key_x, key_y], dtype=jnp.int32)
        key_on_agent = jnp.all(proposed_key_pos == agent_pos)
        
        # If key would spawn on agent, use fallback position (2,2) in top-left room
        key_pos = jnp.where(
            key_on_agent,
            jnp.array([2, 2], dtype=jnp.int32),  # Safe fallback position in top-left room
            proposed_key_pos
        )
        
        # Generate proper MiniGrid observation from initial state
        obs = generate_fourrooms_observation(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            has_key=False,
            door_unlocked=initial_door_unlocked,
            key_pos=key_pos,
            goal1_pos=params.goal1_pos,
            goal2_pos=params.goal2_pos,
            size=params.size
        )
        
        # Create empty achievements array for MiniGrid (2 elements: [large_goal, small_goal])
        achievements = jnp.zeros(2, dtype=jnp.int32)
        
        state = MiniGridEnvState(
            key=rng_key,
            step_count=0,
            done=False,
            obs=obs,
            agent_dir=agent_dir,
            agent_pos=agent_pos,
            achievements=achievements,
            key_pos=key_pos,
            has_key=False,
            door_unlocked=initial_door_unlocked
        )
        
        return obs, state
    
    def get_obs(self, state: MiniGridEnvState) -> jax.Array:
        """Get observation from state."""
        return state.obs

    def is_terminal(self, state: MiniGridEnvState, params: MiniGridEnvParams) -> bool:
        """Check if environment is in terminal state."""
        return state.done

    def render(self, state: MiniGridEnvState, mode='rgb_array', tile_size=32, highlight=True):
        """
        Render the environment using MiniGrid's native Grid rendering.
        
        Args:
            state: Environment state to render
            mode: Rendering mode (only 'rgb_array' supported)
            tile_size: Size of each tile in pixels
            highlight: Whether to highlight the agent's field of view
            
        Returns:
            RGB array of the rendered environment
        """
        if mode != 'rgb_array':
            raise ValueError(f"Unsupported render mode: {mode}")
        
        try:
            import numpy as np
            from minigrid.core.grid import Grid
            from minigrid.core.world_object import Wall, Door, Key, Floor
            from envs.minigrid.minigrid_envs import Goal, Goal2  # Use custom Goal classes with different colors
            from envs.minigrid.jax_minigrid_env import OBJECT_TO_IDX, COLOR_TO_IDX
            
            # Create a 13x13 grid (FourRooms layout)
            width, height = 13, 13
            grid = Grid(width, height)
            
            # Build the Four Rooms layout with walls
            # Outer walls
            for x in range(width):
                grid.set(x, 0, Wall())
                grid.set(x, height - 1, Wall())
            for y in range(height):
                grid.set(0, y, Wall())
                grid.set(width - 1, y, Wall())
            
            # Vertical wall at x=6 with door at y=3
            for y in range(1, height - 1):
                if y != 3:
                    grid.set(6, y, Wall())
                else:
                    # Door at (6, 3) - locked unless door_unlocked is True
                    # In MiniGrid: is_locked=True means locked (needs key), is_open=False means closed
                    is_door_locked = not bool(state.door_unlocked)
                    door = Door('yellow', is_open=False, is_locked=is_door_locked)
                    grid.set(6, y, door)
            
            # Horizontal wall at y=6 with openings at x=3 and x=9
            for x in range(1, width - 1):
                if x != 3 and x != 9:
                    grid.set(x, 6, Wall())
            
            # Add key if it hasn't been collected
            key_pos_np = np.array(state.key_pos)
            if key_pos_np[0] >= 0:  # Valid position means key exists
                kx, ky = int(key_pos_np[0]), int(key_pos_np[1])
                if 0 <= kx < width and 0 <= ky < height:
                    grid.set(kx, ky, Key('yellow'))
            
            # Add goals at fixed positions
            # Goal at (1, 11): small reward goal (emerald green)
            # Goal2 at (11, 11): large reward goal (neon green)
            grid.set(1, 11, Goal())    # Small reward goal - emerald green
            grid.set(11, 11, Goal2())  # Large reward goal - neon green
            
            # Render the grid with agent and FOV highlighting
            agent_pos_np = np.array(state.agent_pos)
            agent_dir_np = int(state.agent_dir)
            
            # Create FOV highlight mask using actual agent_view_size
            # Grid.render() expects mask shape (width, height) and accesses as mask[x, y]
            if highlight:
                highlight_mask = np.zeros((width, height), dtype=bool)
                agent_x, agent_y = int(agent_pos_np[0]), int(agent_pos_np[1])
                
                # Use the actual view size from the environment (e.g., 3 for ViewSize-3x3)
                view_size = self._agent_view_size
                
                # MiniGrid FOV logic: agent sees a view_size x view_size grid
                # agent_pos = (x, y) where x=column, y=row
                # highlight_mask[row, col] corresponds to highlight_mask[y, x]
                # 
                # The agent triangle points in the MOVEMENT direction, and FOV should extend
                # in the same direction as movement (where the triangle points).
                #
                # For a 3x3 view:
                # - dir=0 (moves RIGHT): FOV extends RIGHT, columns [x, x+1, x+2], rows centered [y-1, y, y+1]
                # - dir=1 (moves DOWN): FOV extends DOWN, rows [y, y+1, y+2], columns centered [x-1, x, x+1]
                # - dir=2 (moves LEFT): FOV extends LEFT, columns [x-2, x-1, x], rows centered [y-1, y, y+1]
                # - dir=3 (moves UP): FOV extends UP, rows [y-2, y-1, y], columns centered [x-1, x, x+1]
                
                # Calculate visible area based on movement/facing direction
                # CRITICAL: Grid.render() interprets highlight_mask with SWAPPED indices!
                # mask[r, c] renders at output position [c, r]
                # So to highlight output[row, col], we must set mask[col, row]
                # 
                # Our coordinate system: agent_x = column, agent_y = row
                # To set highlight_mask correctly, we swap: mask[agent_x±, agent_y±]
                
                if agent_dir_np == 0:  # Facing RIGHT (moves right)
                    # Want to highlight: rows [agent_y-1, agent_y, agent_y+1], cols [agent_x, agent_x+1, agent_x+2]
                    # Set mask[cols, rows] = mask[agent_x:agent_x+3, agent_y-1:agent_y+2]
                    min_x = agent_x
                    max_x = min(width, agent_x + view_size)
                    min_y = max(0, agent_y - view_size // 2)
                    max_y = min(height, agent_y + view_size // 2 + 1)
                elif agent_dir_np == 1:  # Facing DOWN (moves down)
                    # Want: rows [agent_y, agent_y+1, agent_y+2], cols [agent_x-1, agent_x, agent_x+1]
                    min_x = max(0, agent_x - view_size // 2)
                    max_x = min(width, agent_x + view_size // 2 + 1)
                    min_y = agent_y
                    max_y = min(height, agent_y + view_size)
                elif agent_dir_np == 2:  # Facing LEFT (moves left)
                    # Want: rows [agent_y-1, agent_y, agent_y+1], cols [agent_x-2, agent_x-1, agent_x]
                    min_x = max(0, agent_x - view_size + 1)
                    max_x = agent_x + 1
                    min_y = max(0, agent_y - view_size // 2)
                    max_y = min(height, agent_y + view_size // 2 + 1)
                else:  # agent_dir_np == 3, Facing UP (moves up)
                    # Want: rows [agent_y-2, agent_y-1, agent_y], cols [agent_x-1, agent_x, agent_x+1]
                    min_x = max(0, agent_x - view_size // 2)
                    max_x = min(width, agent_x + view_size // 2 + 1)
                    min_y = max(0, agent_y - view_size + 1)
                    max_y = agent_y + 1
                
                # Set highlight_mask[x, y] where x=column, y=row
                # Grid.render() expects mask[(width, height)] and accesses as mask[i, j] in loop over cols, rows
                highlight_mask[min_x:max_x, min_y:max_y] = True
            else:
                highlight_mask = None
            
            img = grid.render(
                tile_size=tile_size,
                agent_pos=tuple(agent_pos_np),
                agent_dir=agent_dir_np,
                highlight_mask=highlight_mask
            )
            
            return img
            
        except Exception as e:
            raise RuntimeError(f"Failed to render MiniGrid environment: {e}")

    @property
    def name(self) -> str:
        return f"MiniGridJAX-{self.env_id}"

    @property
    def num_actions(self) -> int:
        return self._action_dim

    def action_space(self, params: Optional[MiniGridEnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(self._action_dim)

    def observation_space(self, params: Optional[MiniGridEnvParams] = None) -> spaces.Box:
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._obs_shape,
            dtype=jnp.float32
        )

    def state_space(self, params: Optional[MiniGridEnvParams] = None) -> spaces.Dict:
        """Return the state space for the environment."""
        return spaces.Dict({
            'key': spaces.Box(low=0, high=1, shape=(), dtype=jnp.uint32),
            'step_count': spaces.Box(low=0, high=10000, shape=(), dtype=jnp.int32),
            'done': spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool_),
            'obs': spaces.Box(low=0.0, high=1.0, shape=self._obs_shape, dtype=jnp.float32),
            'agent_dir': spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
            'agent_pos': spaces.Box(low=0, high=12, shape=(2,), dtype=jnp.int32),
            'achievements': spaces.Box(low=0, high=1000, shape=(22,), dtype=jnp.int32),
            'key_pos': spaces.Box(low=-1, high=12, shape=(2,), dtype=jnp.int32),
            'has_key': spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool_),
            'door_unlocked': spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool_),
        })

    def discount(self, state: MiniGridEnvState, params: MiniGridEnvParams) -> float:
        """Return discount factor."""
        return 1.0 if not state.done else 0.0


# Factory function for creating MiniGrid JAX environments  
def make_minigrid_jax_env(env_id: str) -> MiniGridJAXWrapper:
    """Create a JAX-compatible MiniGrid environment."""
    return MiniGridJAXWrapper(env_id)


# Pre-defined environment creators for common MiniGrid environments
def MiniGridFourRoomsTwoGoalsRandKeyJAX():
    return make_minigrid_jax_env("MiniGrid-FourRooms-TwoGoals-RandKey-ViewSize-3x3-v0")


def MiniGridFourRoomsTwoGoalsFixedJAX():
    return make_minigrid_jax_env("MiniGrid-FourRooms-TwoGoals-Fixed-ViewSize-3x3-v0")


def MiniGridFourRoomsTwoGoalsRandJAX():
    return make_minigrid_jax_env("MiniGrid-FourRooms-TwoGoals-Rand-ViewSize-3x3-v0")


def MiniGridFourRoomsDebugNoGoalFixedKeyJAX():
    return make_minigrid_jax_env("MiniGrid-FourRooms-Debug-NoGoal-FixedKey-ViewSize-3x3-v0")


def MiniGridFourRoomsDebugNoGoalRandKeyJAX():
    return make_minigrid_jax_env("MiniGrid-FourRooms-Debug-NoGoal-RandKey-ViewSize-3x3-v0")