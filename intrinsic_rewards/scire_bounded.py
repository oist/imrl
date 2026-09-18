"""imrl-specific SCIRE variant with the original sigmoid-bounded observation model.

The `scire` package generalizes VODM/VFDM to a Gaussian observation model (an
unconstrained real-valued decoder/predictor) so it works with observations in any
range. imrl's environments (MiniGrid, Craftax Classic Symbolic) produce observations
normalized to [0, 1], and the original imrl implementation relied on that: it squashed
the decoder's and forward-dynamics predictor's raw outputs through a sigmoid before
comparing them to the observation, i.e. a Bernoulli-style observation model.

This module restores that sigmoid squashing on top of the published `scire` package,
so imrl's novelty/surprise/empowerment numerics match the original implementation
exactly, without modifying `scire` itself (which must stay observation-range-agnostic
for other users).
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from scire import SCIREModule, VFDM, VODM
from scire.factory import generate_action_sequences

# Original imrl hardcoded defaults, exposed as configurable parameters when VODM/VFDM
# were generalized into the standalone scire package. Pinned back here explicitly so
# imrl's behavior does not depend on scire's current defaults.
LOGVAR_MIN = -10.0
LOGVAR_MAX = 2.0
LATENT_PRED_COEF = 0.1
MAX_EXHAUSTIVE_SEQUENCES = 216


class BoundedVODM(VODM):
    """VODM with a sigmoid-bounded [0, 1] observation model."""

    def decode(self, z: chex.Array) -> Tuple[chex.Array, chex.Array]:
        recon_mu, recon_logvar = super().decode(z)
        return jax.nn.sigmoid(recon_mu), recon_logvar


class BoundedVFDM(VFDM):
    """VFDM whose observation-space prediction is squashed to [0, 1]."""

    def predict_observation(self, z: chex.Array, action: chex.Array) -> chex.Array:
        return jax.nn.sigmoid(super().predict_observation(z, action))


class BoundedSCIREModule(SCIREModule):
    """SCIREModule wired to the sigmoid-bounded VODM/VFDM above."""

    def setup(self):
        self.vodm = BoundedVODM(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            logvar_min=self.logvar_min,
            logvar_max=self.logvar_max,
        )
        self.vfdm = BoundedVFDM(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            logvar_min=self.logvar_min,
            logvar_max=self.logvar_max,
        )


def create_bounded_scire_system(
    key: jax.random.PRNGKey,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 512,
    latent_dim: int = 128,
    planning_horizon: int = 3,
    num_action_sequences: int = 10,
    beta_vae: float = 1.0,
    logvar_min: float = LOGVAR_MIN,
    logvar_max: float = LOGVAR_MAX,
    latent_pred_coef: float = LATENT_PRED_COEF,
    max_exhaustive_sequences: int = MAX_EXHAUSTIVE_SEQUENCES,
) -> Tuple[BoundedSCIREModule, dict, jnp.ndarray]:
    """Same as `scire.create_scire_system`, but using the sigmoid-bounded model above."""
    init_key, action_seq_key = jax.random.split(key)

    model = BoundedSCIREModule(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        beta_vae=beta_vae,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
        latent_pred_coef=latent_pred_coef,
    )

    action_sequences = generate_action_sequences(
        action_seq_key, action_dim, planning_horizon, num_action_sequences, max_exhaustive_sequences
    )

    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_action = jnp.zeros((1,), dtype=jnp.int32)
    dummy_next_obs = jnp.zeros((1, obs_dim))

    params = model.init(init_key, dummy_obs, dummy_action, dummy_next_obs, action_sequences, init_key)

    return model, params, action_sequences
