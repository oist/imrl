"""
MiniGrid-specific environment factory and utilities.
"""

def make_minigrid_env_from_name(env_name):
    """
    Create MiniGrid environment from name - routes to JAX wrapper.
    
    Args:
        env_name: MiniGrid environment name
        
    Returns:
        JAX-compatible MiniGrid environment
    """
    from envs.minigrid.minigrid_jax_env import make_minigrid_jax_env
    return make_minigrid_jax_env(env_name)


def get_minigrid_env_variants():
    """
    Get list of available MiniGrid environment variants.
    
    Returns:
        List of available MiniGrid environment IDs
    """
    return [
        "MiniGrid-FourRooms-TwoGoals-Fixed-ViewSize-3x3-v0",
        "MiniGrid-FourRooms-TwoGoals-Rand-ViewSize-3x3-v0", 
        "MiniGrid-FourRooms-TwoGoals-RandKey-ViewSize-3x3-v0",
        "MiniGrid-FourRooms-Debug-NoGoal-FixedKey-ViewSize-3x3-v0",
        "MiniGrid-FourRooms-Debug-NoGoal-RandKey-ViewSize-3x3-v0"
    ]


def is_minigrid_env(env_name):
    """
    Check if environment name corresponds to a MiniGrid environment.
    
    Args:
        env_name: Environment name to check
        
    Returns:
        bool: True if MiniGrid environment
    """
    return "MiniGrid" in env_name