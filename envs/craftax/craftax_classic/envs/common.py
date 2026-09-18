from envs.craftax.craftax_classic.envs.craftax_state import EnvState
from envs.craftax.craftax_classic.constants import *


def compute_score(state: EnvState, done: bool):
    # Use weighted achievement rewards scaled by 100x for metrics
    weighted_achievements = state.achievements.astype(jnp.float32) * ACHIEVEMENT_REWARDS * done * 100.0
    info = {}
    for achievement in Achievement:
        name = f"Achievements/{achievement.name.lower()}"
        # Store actual achievement completion status (0 or 100) for success rate calculation
        # This ensures zero-reward achievements are still properly tracked
        info[name] = state.achievements[achievement.value].astype(jnp.float32) * done * 100.0
    # Geometric mean with an offset of 1% using weighted achievements
    info["score"] = jnp.exp(jnp.mean(jnp.log(1 + weighted_achievements))) - 1.0
    return info
