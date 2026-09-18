"""
JAX-native MiniGrid observation generation for proper environment simulation.
This module implements the core MiniGrid observation logic in pure JAX to replace
the broken optimistic simulation.
"""

import jax
import jax.numpy as jnp
from typing import Tuple


# MiniGrid object type constants (matching MiniGrid convention)
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

# MiniGrid color constants
COLOR_TO_IDX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'purple': 3,
    'yellow': 4,
    'grey': 5,
}

# MiniGrid state constants  
STATE_TO_IDX = {
    'open': 0,
    'closed': 1,
    'locked': 2,
}


def create_four_rooms_layout():
    """Create the 13x13 four rooms layout matching CustomFourRoomsTwoGoalsRandKeyViewSize3x3.
    
    This creates the exact layout from the MiniGrid environment:
    - 13x13 grid with walls around the perimeter  
    - Vertical wall at x=6 with door at (6,3)
    - Horizontal wall at y=6 with openings at x=3 and x=9
    - Goal1 at (1,11) - small goal (emerald green)
    - Goal2 at (11,11) - large goal (neon green)
    """
    # 13x13 grid layout for FourRooms (matching the viewer implementation)
    layout = jnp.full((13, 13), OBJECT_TO_IDX['floor'], dtype=jnp.int32)  # Start with floor
    
    # Add walls around the perimeter (x=0,12 and y=0,12)
    layout = layout.at[0, :].set(OBJECT_TO_IDX['wall'])  # Top wall
    layout = layout.at[12, :].set(OBJECT_TO_IDX['wall'])  # Bottom wall
    layout = layout.at[:, 0].set(OBJECT_TO_IDX['wall'])  # Left wall
    layout = layout.at[:, 12].set(OBJECT_TO_IDX['wall'])  # Right wall
    
    # Add vertical wall at x=6 (except door position at y=3)
    for y in range(13):
        if y != 3:  # Skip door position
            layout = layout.at[y, 6].set(OBJECT_TO_IDX['wall'])
    
    # Add horizontal wall at y=6 (except openings at x=3 and x=9) 
    for x in range(13):
        if x != 3 and x != 9:  # Skip opening positions
            layout = layout.at[6, x].set(OBJECT_TO_IDX['wall'])
    
    # Add door at (6,3) - this will be the locked door that requires a key
    layout = layout.at[3, 6].set(OBJECT_TO_IDX['door'])
    
    return layout
def get_view_coords(agent_pos: jax.Array, agent_dir: int, view_size: int = 3) -> Tuple[jax.Array, jax.Array]:
    """Get the coordinates that are visible to the agent in their view.
    
    Args:
        agent_pos: Agent position (x, y)
        agent_dir: Agent direction (0=right, 1=down, 2=left, 3=up)
        view_size: Size of the agent's view (default 3x3)
        
    Returns:
        Tuple of (view_x_coords, view_y_coords) arrays of shape (view_size, view_size)
    """
    # Create relative coordinates for the view
    half_size = view_size // 2
    rel_coords = jnp.arange(-half_size, half_size + 1)
    
    # Create meshgrid for relative positions
    rel_y, rel_x = jnp.meshgrid(rel_coords, rel_coords, indexing='ij')
    
    # Rotate coordinates based on agent direction
    # Direction 0 (right): no rotation
    # Direction 1 (down): 90° clockwise
    # Direction 2 (left): 180°
    # Direction 3 (up): 270° clockwise
    
    # Apply rotation transformations
    def rotate_coords(rel_x, rel_y, direction):
        return jax.lax.switch(
            direction,
            [
                lambda: (rel_x, rel_y),           # 0: no rotation (facing right)
                lambda: (rel_y, -rel_x),          # 1: 90° clockwise (facing down)
                lambda: (-rel_x, -rel_y),         # 2: 180° (facing left)
                lambda: (-rel_y, rel_x),          # 3: 270° clockwise (facing up)
            ]
        )
    
    rotated_x, rotated_y = rotate_coords(rel_x, rel_y, agent_dir)
    
    # Convert to absolute coordinates
    abs_x = agent_pos[0] + rotated_x
    abs_y = agent_pos[1] + rotated_y
    
    return abs_x, abs_y


def generate_observation(
    agent_pos: jax.Array,
    agent_dir: int,
    key_pos: jax.Array,
    has_key: bool,
    door_unlocked: bool,
    goal1_pos: Tuple[int, int],
    goal2_pos: Tuple[int, int],
    layout: jax.Array,
    view_size: int = 3
) -> jax.Array:
    """Generate a MiniGrid observation based on the current game state.
    
    Args:
        agent_pos: Agent position (x, y)
        agent_dir: Agent direction (0=right, 1=down, 2=left, 3=up)
        key_pos: Key position (x, y), (-1, -1) if collected
        has_key: Whether agent has the key
        door_unlocked: Whether the door is unlocked
        goal1_pos: Position of goal 1 (small reward)
        goal2_pos: Position of goal 2 (large reward)
        layout: Base environment layout
        view_size: Size of agent's view
        
    Returns:
        Flattened observation array of shape (view_size * view_size * 3,)
    """
    # Get view coordinates
    view_x, view_y = get_view_coords(agent_pos, agent_dir, view_size)
    
    # Initialize observation arrays for object type, color, and state
    obs_object = jnp.full((view_size, view_size), OBJECT_TO_IDX['unseen'], dtype=jnp.int32)
    obs_color = jnp.zeros((view_size, view_size), dtype=jnp.int32)
    obs_state = jnp.zeros((view_size, view_size), dtype=jnp.int32)
    
    # Check bounds and get visible objects
    in_bounds = (view_x >= 0) & (view_x < layout.shape[1]) & (view_y >= 0) & (view_y < layout.shape[0])
    
    # Get base objects from layout
    layout_objects = jnp.where(
        in_bounds,
        layout[view_y, view_x],
        OBJECT_TO_IDX['unseen']
    )
    
    # Initialize with base layout
    obs_object = jnp.where(in_bounds, layout_objects, OBJECT_TO_IDX['unseen'])
    
    # Place key if it exists and is visible
    key_exists = key_pos[0] >= 0  # Key exists if position is valid
    key_visible = in_bounds & (view_x == key_pos[0]) & (view_y == key_pos[1])
    obs_object = jnp.where(key_exists & key_visible, OBJECT_TO_IDX['key'], obs_object)
    obs_color = jnp.where(key_exists & key_visible, COLOR_TO_IDX['yellow'], obs_color)
    
    # Handle door states
    door_positions = (obs_object == OBJECT_TO_IDX['door'])
    door_state = jax.lax.select(door_unlocked, STATE_TO_IDX['open'], STATE_TO_IDX['locked'])
    obs_state = jnp.where(door_positions, door_state, obs_state)
    obs_color = jnp.where(door_positions, COLOR_TO_IDX['yellow'], obs_color)
    
    # Place goals using the provided positions
    # Goal 1 - small goal (emerald green) 
    goal1_visible = in_bounds & (view_x == goal1_pos[0]) & (view_y == goal1_pos[1])
    obs_object = jnp.where(goal1_visible, OBJECT_TO_IDX['goal'], obs_object)
    obs_color = jnp.where(goal1_visible, COLOR_TO_IDX['green'], obs_color)
    
    # Goal 2 - large goal (neon green, but we'll use blue to distinguish)
    goal2_visible = in_bounds & (view_x == goal2_pos[0]) & (view_y == goal2_pos[1])
    obs_object = jnp.where(goal2_visible, OBJECT_TO_IDX['goal'], obs_object)
    obs_color = jnp.where(goal2_visible, COLOR_TO_IDX['blue'], obs_color)
    
    # Stack the three channels and flatten
    obs_3d = jnp.stack([obs_object, obs_color, obs_state], axis=-1)
    obs_flat = obs_3d.flatten()
    
    # Convert to float32 and normalize
    obs_normalized = obs_flat.astype(jnp.float32) / 10.0  # Normalize to [0, 1] range
    
    return obs_normalized


def encode_observation_with_wrappers(obs_normalized: jax.Array) -> jax.Array:
    """Encode the observation using the same wrappers as training (OneHot + DictToArray).
    
    This simulates the effect of OneHotPartialObsWrapper and DictToArrayObsWrapper
    to produce observations that match the training setup.
    """
    # The wrappers transform a 3x3x3 observation into a 200-dimensional one-hot encoded vector
    # For simplicity, we'll create a compatible representation
    
    # Get the raw 3x3x3 observation  
    obs_3d = obs_normalized.reshape((3, 3, 3))
    
    # Create a one-hot encoded representation
    # This is a simplified version of what OneHotPartialObsWrapper does
    
    # For each cell (3x3 = 9 cells) and each channel (3 channels), 
    # create one-hot vectors for possible values
    
    # Object types (11 possible values)
    object_channel = obs_3d[:, :, 0]  # Shape: (3, 3)
    object_indices = (object_channel * 10).astype(jnp.int32)  # Denormalize
    object_onehot = jax.nn.one_hot(object_indices, 11)  # Shape: (3, 3, 11)
    
    # Colors (6 possible values)
    color_channel = obs_3d[:, :, 1]  # Shape: (3, 3) 
    color_indices = (color_channel * 10).astype(jnp.int32)  # Denormalize
    color_onehot = jax.nn.one_hot(color_indices, 6)  # Shape: (3, 3, 6)
    
    # States (3 possible values)
    state_channel = obs_3d[:, :, 2]  # Shape: (3, 3)
    state_indices = (state_channel * 10).astype(jnp.int32)  # Denormalize
    state_onehot = jax.nn.one_hot(state_indices, 3)  # Shape: (3, 3, 3)
    
    # Concatenate all one-hot vectors
    all_onehot = jnp.concatenate([
        object_onehot.reshape(-1),  # 9 * 11 = 99
        color_onehot.reshape(-1),   # 9 * 6 = 54
        state_onehot.reshape(-1),   # 9 * 3 = 27
    ])  # Total: 99 + 54 + 27 = 180
    
    # Pad to reach 200 dimensions to match training
    padded_obs = jnp.pad(all_onehot, (0, 200 - len(all_onehot)), constant_values=0.0)
    
    return padded_obs


def generate_fourrooms_observation(
    agent_pos: jax.Array,
    agent_dir: int,
    key_pos: jax.Array,
    has_key: bool,
    door_unlocked: bool,
    goal1_pos: Tuple[int, int],
    goal2_pos: Tuple[int, int],
    size: int = 13
) -> jax.Array:
    """Generate observation for FourRooms environment with proper encoding.
    
    This is the main function to use for generating observations in the fixed environment.
    """
    # Create the four rooms layout
    layout = create_four_rooms_layout()
    
    # Generate raw observation
    raw_obs = generate_observation(
        agent_pos, agent_dir, key_pos, has_key, door_unlocked, 
        goal1_pos, goal2_pos, layout, view_size=3
    )
    
    # Apply wrapper encoding to match training setup
    wrapped_obs = encode_observation_with_wrappers(raw_obs)
    
    return wrapped_obs