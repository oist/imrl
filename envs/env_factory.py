"""
Common environment factory for all environment types.
This centralizes environment creation logic and routes to appropriate handlers.
"""

def make_env_from_name(env_name, auto_reset=True, base_seed=None):
    """
    Universal environment factory that routes to appropriate environment creators.
    
    Args:
        env_name: Name of the environment to create
        auto_reset: Whether to use auto-reset wrapper (for Craftax environments)
        base_seed: Base random seed for deterministic behavior
        
    Returns:
        Environment instance compatible with JAX training
    """
    # Route to Craftax environments
    if any(craftax_id in env_name for craftax_id in ["Craftax-Classic", "Craftax-Symbolic", "Craftax-Pixels"]):
        from envs.craftax.craftax_env import make_craftax_env_from_name
        return make_craftax_env_from_name(env_name, auto_reset)
    
    # Route to MiniGrid environments  
    elif "MiniGrid" in env_name:
        from envs.minigrid.minigrid_jax_env import make_minigrid_jax_env
        return make_minigrid_jax_env(env_name)
    
    else:
        raise ValueError(f"Unknown environment type: {env_name}")


def get_env_type(env_name):
    """
    Determine the type of environment from its name.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        str: Environment type ('craftax' or 'minigrid')
    """
    if any(craftax_id in env_name for craftax_id in ["Craftax-Classic", "Craftax-Symbolic", "Craftax-Pixels"]):
        return "craftax"
    elif "MiniGrid" in env_name:
        return "minigrid"
    else:
        raise ValueError(f"Unknown environment type: {env_name}")