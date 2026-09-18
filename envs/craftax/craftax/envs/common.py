from envs.craftax.craftax.craftax_state import EnvState
from envs.craftax.craftax.constants import *


def log_achievements_to_info(state: EnvState, done: bool):
    # Store actual achievement completion status (0 or 100) for success rate calculation
    achievements = state.achievements.astype(jnp.float32) * done * 100.0
    info = {}
    for achievement in Achievement:
        name = f"Achievements/{achievement.name.lower()}"
        info[name] = achievements[achievement.value]
    return info
