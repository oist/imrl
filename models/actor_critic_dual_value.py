"""
Dual Value Head Actor-Critic Networks

Separates value estimation into two heads:
- v_extrinsic: estimates value from extrinsic (environment) rewards
- v_intrinsic: estimates value from intrinsic (curiosity) rewards

This separation prevents value function confusion when intrinsic reward magnitudes
change over training, and allows the agent to better balance exploration vs exploitation
for unlocking hidden achievements with zero extrinsic reward.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant
from models.initializers import orthogonal
from typing import Sequence, Dict, Tuple
import distrax


class ActorCriticDualValue(nn.Module):
    """Fully-connected actor-critic with dual value heads"""
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Shared embedding for actor and both critics
        embedding = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        embedding = activation(embedding)

        embedding = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        embedding = activation(embedding)

        embedding = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        embedding = activation(embedding)

        # Actor head
        actor_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        pi = distrax.Categorical(logits=actor_logits)

        # Extrinsic value head (for environment rewards)
        v_ext = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        v_ext = activation(v_ext)
        v_ext = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(v_ext)
        v_ext = activation(v_ext)
        v_ext = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v_ext)
        v_ext = jnp.squeeze(v_ext, axis=-1)

        # Intrinsic value head (for curiosity/exploration rewards)
        v_int = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        v_int = activation(v_int)
        v_int = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(v_int)
        v_int = activation(v_int)
        v_int = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v_int)
        v_int = jnp.squeeze(v_int, axis=-1)

        return pi, v_ext, v_int


class ActorCriticConvDualValue(nn.Module):
    """Convolutional actor-critic with dual value heads for image observations"""
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs):
        # Convolutional feature extraction
        x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        embedding = x.reshape(x.shape[0], -1)

        # Actor head
        actor_logits = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_logits = nn.relu(actor_logits)
        actor_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_logits)
        pi = distrax.Categorical(logits=actor_logits)

        # Extrinsic value head
        v_ext = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        v_ext = nn.relu(v_ext)
        v_ext = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v_ext)
        v_ext = jnp.squeeze(v_ext, axis=-1)

        # Intrinsic value head
        v_int = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        v_int = nn.relu(v_int)
        v_int = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v_int)
        v_int = jnp.squeeze(v_int, axis=-1)

        return pi, v_ext, v_int


class ActorCriticConvSymbolicCraftaxDualValue(nn.Module):
    """Craftax-specific actor-critic with dual value heads for mixed obs"""
    action_dim: Sequence[int]
    map_obs_shape: Sequence[int]
    layer_width: int

    @nn.compact
    def __call__(self, obs):
        # Split into map and flat obs
        flat_map_obs_shape = (
            self.map_obs_shape[0] * self.map_obs_shape[1] * self.map_obs_shape[2]
        )
        image_obs = obs[:, :flat_map_obs_shape]
        image_dim = self.map_obs_shape
        image_obs = image_obs.reshape((image_obs.shape[0], *image_dim))

        flat_obs = obs[:, flat_map_obs_shape:]

        # Convolutions on map
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_obs)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_embedding)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = image_embedding.reshape(image_embedding.shape[0], -1)

        # Combine embeddings
        embedding = jnp.concatenate([image_embedding, flat_obs], axis=-1)
        embedding = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        # Actor head
        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Extrinsic value head
        v_ext = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        v_ext = nn.relu(v_ext)
        v_ext = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(v_ext)
        v_ext = nn.relu(v_ext)
        v_ext = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(v_ext)
        v_ext = nn.relu(v_ext)
        v_ext = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v_ext)
        v_ext = jnp.squeeze(v_ext, axis=-1)

        # Intrinsic value head
        v_int = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        v_int = nn.relu(v_int)
        v_int = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(v_int)
        v_int = nn.relu(v_int)
        v_int = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(v_int)
        v_int = nn.relu(v_int)
        v_int = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v_int)
        v_int = jnp.squeeze(v_int, axis=-1)

        return pi, v_ext, v_int
