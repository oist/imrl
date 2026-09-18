"""
EMI (Exploration with Mutual Information) - JAX Implementation

Based on the paper "EMI: Exploration with Mutual Information" (Kim et al., ICML 2019)
Original implementation: https://github.com/snu-mllab/EMI

Core idea: Learn compact state and action embeddings φ(s) and ψ(a) such that
the forward dynamics becomes linear in embedding space:
    φ(s_{t+1}) ≈ φ(s_t) + ψ(a_t) + reconciler(φ(s_t), ψ(a_t))

Intrinsic rewards are computed based on:
1. Diversity-seeking: RBF kernel similarity in embedding space (exploration bonus)
2. Residual error: Prediction error as novelty signal

Mutual information terms are added to the training loss:
- I(ψ; φ' | φ): How much action embedding tells about next state given current state
- I(φ'; φ | ψ): How much next state tells about current state given action
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import xavier_uniform, zeros
from typing import Tuple, Dict, Any, Optional, Callable
import chex
import optax


# ============================================================================
# Neural Network Components
# ============================================================================

class StateEmbedding(nn.Module):
    """
    State embedding network: φ(s) → ℝ^{embedding_dim}
    
    Maps observations to a compact latent space where dynamics are (approximately) linear.
    """
    embedding_dim: int
    hidden_dim: int = 128
    num_layers: int = 3
    
    @nn.compact
    def __call__(self, obs: chex.Array) -> chex.Array:
        """
        Args:
            obs: Observation [batch, obs_dim]
        Returns:
            embedding: State embedding [batch, embedding_dim]
        """
        x = obs
        for i in range(self.num_layers - 1):
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=xavier_uniform(),
                bias_init=zeros,
                name=f'fc_{i}'
            )(x)
            x = nn.LayerNorm(name=f'ln_{i}')(x)
            x = nn.relu(x)
        
        # Final embedding layer
        embedding = nn.Dense(
            self.embedding_dim,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
            name='embedding'
        )(x)
        
        return embedding


class ActionEmbedding(nn.Module):
    """
    Action embedding network: ψ(a) → ℝ^{embedding_dim}
    
    Maps actions to the same embedding space as states.
    For discrete actions, this takes one-hot encoded actions.
    """
    embedding_dim: int
    action_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    
    @nn.compact
    def __call__(self, action: chex.Array) -> chex.Array:
        """
        Args:
            action: Action one-hot [batch, action_dim] or continuous [batch, action_dim]
        Returns:
            embedding: Action embedding [batch, embedding_dim]
        """
        x = action
        for i in range(self.num_layers - 1):
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=xavier_uniform(),
                bias_init=zeros,
                name=f'fc_{i}'
            )(x)
            x = nn.LayerNorm(name=f'ln_{i}')(x)
            x = nn.relu(x)
        
        # Final embedding layer
        embedding = nn.Dense(
            self.embedding_dim,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
            name='embedding'
        )(x)
        
        return embedding


class Reconciler(nn.Module):
    """
    Reconciler network for residual correction in dynamics prediction.
    
    r(φ(s), ψ(a)) predicts the residual: φ(s') - φ(s) - ψ(a)
    
    This allows for more complex dynamics while keeping the main structure linear.
    """
    embedding_dim: int
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, state_emb: chex.Array, action_emb: chex.Array) -> chex.Array:
        """
        Args:
            state_emb: State embedding [batch, embedding_dim]
            action_emb: Action embedding [batch, embedding_dim]
        Returns:
            residual: Dynamics residual [batch, embedding_dim]
        """
        x = jnp.concatenate([state_emb, action_emb], axis=-1)
        
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
            name='fc_1'
        )(x)
        x = nn.LayerNorm(name='ln_1')(x)
        x = nn.relu(x)
        
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
            name='fc_2'
        )(x)
        x = nn.LayerNorm(name='ln_2')(x)
        x = nn.relu(x)
        
        residual = nn.Dense(
            self.embedding_dim,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
            name='residual'
        )(x)
        
        return residual


class MutualInfoDiscriminator(nn.Module):
    """
    Discriminator for MINE-style mutual information estimation.
    
    Used to estimate:
    - I(ψ; φ' | φ): Mutual information between action embedding and next state embedding
      given current state embedding
    - I(φ'; φ | ψ): Mutual information between next state and current state given action
    
    The discriminator outputs a score T(x, y) where the mutual information is:
    I(X; Y) ≈ E_joint[T(x,y)] - log(E_marginal[exp(T(x,y))])
    """
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, x1: chex.Array, x2: chex.Array, condition: chex.Array) -> chex.Array:
        """
        Args:
            x1: First variable embedding [batch, dim1]
            x2: Second variable embedding [batch, dim2]
            condition: Conditioning variable [batch, cond_dim]
        Returns:
            score: Discriminator score [batch, 1]
        """
        x = jnp.concatenate([x1, x2, condition], axis=-1)
        
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
            name='fc_1'
        )(x)
        x = nn.relu(x)
        
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
            name='fc_2'
        )(x)
        x = nn.relu(x)
        
        score = nn.Dense(
            1,
            kernel_init=xavier_uniform(),
            bias_init=zeros,
            name='score'
        )(x)
        
        return score


class EMIModel(nn.Module):
    """
    Complete EMI model combining all components.
    
    Forward dynamics: φ(s') ≈ φ(s) + ψ(a) + r(φ(s), ψ(a))
    """
    obs_dim: int
    action_dim: int
    embedding_dim: int = 32
    hidden_dim: int = 128
    use_reconciler: bool = True
    
    def setup(self):
        self.state_encoder = StateEmbedding(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        self.action_encoder = ActionEmbedding(
            embedding_dim=self.embedding_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
        if self.use_reconciler:
            self.reconciler = Reconciler(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim
            )
        
        # Mutual information discriminators
        self.mi_action_discriminator = MutualInfoDiscriminator(hidden_dim=self.hidden_dim)
        self.mi_obs_discriminator = MutualInfoDiscriminator(hidden_dim=self.hidden_dim)
    
    def __call__(
        self,
        obs: chex.Array,
        action: chex.Array,
        next_obs: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Forward pass computing all embeddings.
        
        Args:
            obs: Current observation [batch, obs_dim]
            action: Action (one-hot for discrete) [batch, action_dim]
            next_obs: Next observation [batch, obs_dim]
            
        Returns:
            state_emb: φ(s) [batch, embedding_dim]
            action_emb: ψ(a) [batch, embedding_dim]
            next_state_emb: φ(s') [batch, embedding_dim]
            predicted_next_emb: φ(s) + ψ(a) + r(φ(s), ψ(a)) [batch, embedding_dim]
        """
        state_emb = self.state_encoder(obs)
        action_emb = self.action_encoder(action)
        next_state_emb = self.state_encoder(next_obs)
        
        # Predict next state embedding
        if self.use_reconciler:
            residual = self.reconciler(state_emb, action_emb)
            predicted_next_emb = state_emb + action_emb + residual
        else:
            predicted_next_emb = state_emb + action_emb
        
        return state_emb, action_emb, next_state_emb, predicted_next_emb
    
    def forward_with_mi(
        self,
        obs: chex.Array,
        action: chex.Array,
        next_obs: chex.Array,
        shuffled_action: chex.Array,
        shuffled_next_obs: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, 
               chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Forward pass with MI score computation.
        
        Args:
            obs: Current observation [batch, obs_dim]
            action: Action (one-hot for discrete) [batch, action_dim]
            next_obs: Next observation [batch, obs_dim]
            shuffled_action: Shuffled actions for marginal estimation [batch, action_dim]
            shuffled_next_obs: Shuffled next obs for marginal estimation [batch, obs_dim]
            
        Returns:
            state_emb, action_emb, next_state_emb, predicted_next_emb,
            mi_action_joint, mi_action_marginal, mi_obs_joint, mi_obs_marginal
        """
        # Get embeddings
        state_emb = self.state_encoder(obs)
        action_emb = self.action_encoder(action)
        next_state_emb = self.state_encoder(next_obs)
        
        # Shuffled embeddings for marginal distribution
        shuffled_action_emb = self.action_encoder(shuffled_action)
        shuffled_next_state_emb = self.state_encoder(shuffled_next_obs)
        
        # Predict next state embedding
        if self.use_reconciler:
            residual = self.reconciler(state_emb, action_emb)
            predicted_next_emb = state_emb + action_emb + residual
        else:
            predicted_next_emb = state_emb + action_emb
        
        # MI scores: I(ψ; φ' | φ)
        mi_action_joint = self.mi_action_discriminator(action_emb, next_state_emb, state_emb)
        mi_action_marginal = self.mi_action_discriminator(shuffled_action_emb, next_state_emb, state_emb)
        
        # MI scores: I(φ'; φ | ψ)
        mi_obs_joint = self.mi_obs_discriminator(next_state_emb, state_emb, action_emb)
        mi_obs_marginal = self.mi_obs_discriminator(shuffled_next_state_emb, state_emb, action_emb)
        
        return (state_emb, action_emb, next_state_emb, predicted_next_emb,
                mi_action_joint, mi_action_marginal, mi_obs_joint, mi_obs_marginal)


# ============================================================================
# EMI Training State
# ============================================================================

class EMIState:
    """State container for EMI training."""
    def __init__(
        self,
        params: Dict,
        opt_state: Any,
        embedding_pool: Optional[chex.Array] = None,
        pool_size: int = 10000,
        embedding_dim: int = 32,
    ):
        self.params = params
        self.opt_state = opt_state
        self.embedding_pool = embedding_pool  # For diversity-seeking reward
        self.pool_size = pool_size
        self.embedding_dim = embedding_dim
        self.pool_idx = 0


# ============================================================================
# Loss Functions
# ============================================================================

def compute_dynamics_loss(
    predicted_next_emb: chex.Array,
    next_state_emb: chex.Array
) -> chex.Array:
    """
    Forward dynamics loss: ||φ(s') - (φ(s) + ψ(a) + r)||²
    
    Args:
        predicted_next_emb: φ(s) + ψ(a) + r(φ(s), ψ(a)) [batch, embedding_dim]
        next_state_emb: φ(s') [batch, embedding_dim]
    Returns:
        loss: Mean squared error loss
    """
    error = predicted_next_emb - next_state_emb
    loss = jnp.mean(jnp.sum(error ** 2, axis=-1))
    return loss


def compute_mine_loss(
    joint_scores: chex.Array,
    marginal_scores: chex.Array,
    ma_rate: float = 0.01
) -> Tuple[chex.Array, chex.Array]:
    """
    MINE (Mutual Information Neural Estimation) loss.
    
    I(X;Y) ≈ E_joint[T(x,y)] - log(E_marginal[exp(T(x,y))])
    
    The loss is the negative of this estimate (we want to maximize MI).
    
    Args:
        joint_scores: T(x,y) from joint distribution [batch, 1]
        marginal_scores: T(x',y) from marginal [batch, 1]
        ma_rate: Moving average rate for bias correction
    Returns:
        loss: Negative MI estimate
        mi_estimate: MI estimate (for logging)
    """
    # E_joint[T]
    joint_mean = jnp.mean(joint_scores)
    
    # log E_marginal[exp(T)] with numerical stability
    # Use log-sum-exp trick
    max_marginal = jnp.max(marginal_scores)
    log_mean_exp = max_marginal + jnp.log(jnp.mean(jnp.exp(marginal_scores - max_marginal)))
    
    mi_estimate = joint_mean - log_mean_exp
    
    # Loss is negative MI (we want to maximize)
    loss = -mi_estimate
    
    return loss, mi_estimate


# ============================================================================
# Intrinsic Reward Computation
# ============================================================================

def compute_diversity_reward(
    next_state_emb: chex.Array,
    embedding_pool: chex.Array,
    bandwidth: float = 1.0
) -> chex.Array:
    """
    Diversity-seeking intrinsic reward using RBF kernel similarity.
    
    reward = 1 - mean(exp(-||φ(s') - φ(pool)||² / (2σ²)))
    
    States that are far from the pool get higher rewards (closer to 1).
    States similar to pool get lower rewards (closer to 0).
    
    Args:
        next_state_emb: φ(s') [batch, embedding_dim]
        embedding_pool: Pool of past embeddings [pool_size, embedding_dim]
        bandwidth: RBF kernel bandwidth σ
    Returns:
        reward: Diversity reward [batch] in range [0, 1]
    """
    # If pool is empty or too small, return high diversity reward
    # (encourage exploration when we haven't seen many states)
    valid_pool_size = jnp.sum(jnp.any(embedding_pool != 0, axis=-1))
    
    def compute_reward(next_emb, pool):
        # Compute pairwise distances: [batch, pool_size]
        diff = next_emb[:, None, :] - pool[None, :, :]  # [batch, pool, dim]
        sq_dist = jnp.sum(diff ** 2, axis=-1)  # [batch, pool]
        
        # RBF kernel similarity
        similarity = jnp.exp(-sq_dist / (2 * bandwidth ** 2))
        
        # Average similarity to pool (higher = more similar = less novel)
        avg_similarity = jnp.mean(similarity, axis=-1)
        
        # Reward is (1 - similarity): higher when less similar to pool
        # This gives positive rewards in [0, 1]
        reward = 1.0 - avg_similarity
        return reward
    
    # Use where to handle empty pool case - return 1.0 (max diversity) when pool is empty
    reward = jax.lax.cond(
        valid_pool_size > 10,
        lambda: compute_reward(next_state_emb, embedding_pool),
        lambda: jnp.ones(next_state_emb.shape[0])  # Max diversity when pool is empty
    )
    
    return reward


def compute_residual_reward(
    predicted_next_emb: chex.Array,
    next_state_emb: chex.Array
) -> chex.Array:
    """
    Residual error intrinsic reward.
    
    reward = mean(||φ(s') - predicted||²)
    
    States with high prediction error are novel/surprising.
    Using mean instead of sum to make reward independent of embedding dimension.
    
    Args:
        predicted_next_emb: Model's prediction [batch, embedding_dim]
        next_state_emb: Actual next state embedding [batch, embedding_dim]
    Returns:
        reward: Residual reward [batch] (typically in range [0, 1] for normalized embeddings)
    """
    error = predicted_next_emb - next_state_emb
    # Use mean instead of sum to normalize by embedding dimension
    # This gives values in similar range to diversity reward
    reward = jnp.mean(error ** 2, axis=-1)
    return reward


def update_embedding_pool(
    embedding_pool: chex.Array,
    pool_idx: int,
    new_embeddings: chex.Array,
    pool_size: int
) -> Tuple[chex.Array, int]:
    """
    Update the embedding pool with new state embeddings using a circular buffer.
    
    Args:
        embedding_pool: Current pool [pool_size, embedding_dim]
        pool_idx: Current insertion index
        new_embeddings: New embeddings to add [batch, embedding_dim]
        pool_size: Maximum pool size
        
    Returns:
        updated_pool: Updated embedding pool
        new_pool_idx: New insertion index
    """
    batch_size = new_embeddings.shape[0]
    
    # Calculate indices for insertion (circular buffer)
    indices = (jnp.arange(batch_size) + pool_idx) % pool_size
    
    # Update pool at those indices
    updated_pool = embedding_pool.at[indices].set(new_embeddings)
    
    # Update pool index
    new_pool_idx = (pool_idx + batch_size) % pool_size
    
    return updated_pool, new_pool_idx


# ============================================================================
# EMI System Creation and Training
# ============================================================================

def create_emi_system(
    key: chex.PRNGKey,
    obs_dim: int,
    action_dim: int,
    embedding_dim: int = 32,
    hidden_dim: int = 128,
    use_reconciler: bool = True,
    learning_rate: float = 3e-4,
    pool_size: int = 10000,
) -> Tuple[Dict, Callable, optax.GradientTransformation]:
    """
    Create the EMI intrinsic reward system.
    
    Args:
        key: Random key for initialization
        obs_dim: Observation dimension
        action_dim: Action dimension (number of actions for discrete)
        embedding_dim: Embedding space dimension
        hidden_dim: Hidden layer dimension
        use_reconciler: Whether to use reconciler network for residuals
        learning_rate: Learning rate for optimizer
        pool_size: Size of embedding pool for diversity reward
        
    Returns:
        params: Dictionary containing model parameters and hyperparameters
        apply_fn: Function to compute intrinsic rewards
        optimizer: Optax optimizer
    """
    # Create model
    model = EMIModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        use_reconciler=use_reconciler
    )
    
    # Initialize parameters - use forward_with_mi to include MI discriminator params
    key, init_key = jax.random.split(key)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_action = jnp.zeros((1, action_dim))
    # Initialize with forward_with_mi to include all sub-modules (including MI discriminators)
    params = model.init(
        init_key,
        dummy_obs, dummy_action, dummy_obs, dummy_action, dummy_obs,
        method=model.forward_with_mi
    )
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Initialize embedding pool
    embedding_pool = jnp.zeros((pool_size, embedding_dim))
    
    # Package everything
    emi_params = {
        'params': params,
        'opt_state': opt_state,
        'embedding_pool': embedding_pool,
        'pool_idx': 0,
        'config': {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'use_reconciler': use_reconciler,
            'pool_size': pool_size,
            'learning_rate': learning_rate,
        }
    }
    
    return emi_params, model.apply, optimizer


def compute_emi_intrinsic_rewards(
    emi_params: Dict,
    obs: chex.Array,
    action: chex.Array,
    next_obs: chex.Array,
    diversity_coeff: float = 0.1,
    residual_coeff: float = 0.1,
    diversity_bandwidth: float = 1.0,
    return_components: bool = False,
    model_apply_fn: Optional[Callable] = None,
) -> Tuple[chex.Array, Dict]:
    """
    Compute EMI intrinsic rewards for a batch of transitions.
    
    Args:
        emi_params: EMI system parameters
        obs: Current observations [batch, obs_dim]
        action: Actions (one-hot for discrete) [batch, action_dim]
        next_obs: Next observations [batch, obs_dim]
        diversity_coeff: Weight for diversity-seeking reward
        residual_coeff: Weight for residual error reward
        diversity_bandwidth: RBF kernel bandwidth for diversity reward
        return_components: Whether to return individual reward components
        model_apply_fn: Model apply function (if not provided, creates new model)
        
    Returns:
        intrinsic_rewards: Total intrinsic rewards [batch]
        info: Dictionary with components and metrics
    """
    config = emi_params['config']
    params = emi_params['params']
    embedding_pool = emi_params['embedding_pool']
    
    # Create model if apply_fn not provided
    if model_apply_fn is None:
        model = EMIModel(
            obs_dim=config['obs_dim'],
            action_dim=config['action_dim'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            use_reconciler=config['use_reconciler']
        )
        model_apply_fn = model.apply
    
    # Forward pass
    state_emb, action_emb, next_state_emb, predicted_next_emb = model_apply_fn(
        params, obs, action, next_obs
    )
    
    # Compute reward components
    diversity_reward = compute_diversity_reward(
        next_state_emb, embedding_pool, diversity_bandwidth
    )
    residual_reward = compute_residual_reward(predicted_next_emb, next_state_emb)
    
    # Total intrinsic reward
    total_reward = diversity_coeff * diversity_reward + residual_coeff * residual_reward
    
    # Compute dynamics loss for info
    dynamics_loss = compute_dynamics_loss(predicted_next_emb, next_state_emb)
    
    info = {
        'diversity_reward': diversity_reward,
        'residual_reward': residual_reward,
        'dynamics_loss': dynamics_loss,
        'state_emb_norm': jnp.mean(jnp.sum(state_emb ** 2, axis=-1)),
        'action_emb_norm': jnp.mean(jnp.sum(action_emb ** 2, axis=-1)),
        'prediction_error': jnp.mean(jnp.sum((predicted_next_emb - next_state_emb) ** 2, axis=-1)),
    }
    
    if return_components:
        info['total'] = total_reward
    
    return total_reward, info


def train_emi_step(
    emi_params: Dict,
    obs: chex.Array,
    action: chex.Array,
    next_obs: chex.Array,
    optimizer: optax.GradientTransformation,
    key: chex.PRNGKey,
    mi_action_weight: float = 0.05,
    mi_obs_weight: float = 0.05,
    dynamics_weight: float = 1.0,
) -> Tuple[Dict, Dict]:
    """
    Perform one training step for EMI.
    
    Args:
        emi_params: EMI system parameters
        obs: Current observations [batch, obs_dim]
        action: Actions [batch, action_dim]
        next_obs: Next observations [batch, obs_dim]
        optimizer: Optax optimizer
        key: Random key for shuffling
        mi_action_weight: Weight for action MI loss
        mi_obs_weight: Weight for observation MI loss
        dynamics_weight: Weight for dynamics prediction loss
        
    Returns:
        updated_emi_params: Updated parameters
        metrics: Training metrics
    """
    config = emi_params['config']
    params = emi_params['params']
    opt_state = emi_params['opt_state']
    embedding_pool = emi_params['embedding_pool']
    pool_idx = emi_params['pool_idx']
    
    # Create model
    model = EMIModel(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        use_reconciler=config['use_reconciler']
    )
    
    def loss_fn(params):
        # Create shuffled inputs for MI estimation
        batch_size = obs.shape[0]
        shuffle_key1, shuffle_key2 = jax.random.split(key)
        
        perm1 = jax.random.permutation(shuffle_key1, batch_size)
        perm2 = jax.random.permutation(shuffle_key2, batch_size)
        
        shuffled_action = action[perm1]
        shuffled_next_obs = next_obs[perm2]
        
        # Forward pass with MI computation
        (state_emb, action_emb, next_state_emb, predicted_next_emb,
         mi_action_joint, mi_action_marginal, mi_obs_joint, mi_obs_marginal) = model.apply(
            params, obs, action, next_obs, shuffled_action, shuffled_next_obs,
            method=model.forward_with_mi
        )
        
        # Dynamics loss
        dynamics_loss = compute_dynamics_loss(predicted_next_emb, next_state_emb)
        
        # MI losses (negative MI since we want to maximize)
        mi_action_loss, mi_action_est = compute_mine_loss(mi_action_joint, mi_action_marginal)
        mi_obs_loss, mi_obs_est = compute_mine_loss(mi_obs_joint, mi_obs_marginal)
        
        # Total loss
        total_loss = (
            dynamics_weight * dynamics_loss +
            mi_action_weight * mi_action_loss +
            mi_obs_weight * mi_obs_loss
        )
        
        metrics = {
            'total_loss': total_loss,
            'dynamics_loss': dynamics_loss,
            'mi_action_loss': mi_action_loss,
            'mi_obs_loss': mi_obs_loss,
            'mi_action_estimate': mi_action_est,
            'mi_obs_estimate': mi_obs_est,
        }
        
        return total_loss, (metrics, next_state_emb)
    
    # Compute gradients
    (loss, (metrics, next_state_emb)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    # Update embedding pool (circular buffer)
    batch_size = obs.shape[0]
    pool_size = config['pool_size']
    
    # Calculate indices for updating pool
    indices = (pool_idx + jnp.arange(batch_size)) % pool_size
    new_pool = embedding_pool.at[indices].set(next_state_emb)
    new_pool_idx = (pool_idx + batch_size) % pool_size
    
    # Package updated parameters
    updated_emi_params = {
        'params': new_params,
        'opt_state': new_opt_state,
        'embedding_pool': new_pool,
        'pool_idx': new_pool_idx,
        'config': config,
    }
    
    return updated_emi_params, metrics


# ============================================================================
# Vectorized/JIT-compiled versions for efficient training
# ============================================================================

def create_emi_train_step_jit(
    optimizer: optax.GradientTransformation,
    config: Dict,
) -> Callable:
    """
    Create a JIT-compiled training step function.
    
    Args:
        optimizer: Optax optimizer
        config: EMI configuration dict
        
    Returns:
        jitted_train_step: JIT-compiled training function
            Signature: (params, opt_state, embedding_pool, pool_idx, obs, action, next_obs, key)
                       -> (new_params, new_opt_state, new_pool, new_pool_idx, metrics)
    """
    model = EMIModel(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        use_reconciler=config['use_reconciler']
    )
    
    # Bake in the weight values from config
    mi_action_weight = config.get('mi_action_weight', 0.05)
    mi_obs_weight = config.get('mi_obs_weight', 0.05)
    dynamics_weight = config.get('dynamics_weight', 1.0)
    pool_size = config['pool_size']
    
    @jax.jit
    def train_step(params, opt_state, embedding_pool, pool_idx, 
                   obs, action, next_obs, key):
        
        def loss_fn(params):
            # Create shuffled inputs for MI estimation
            batch_size = obs.shape[0]
            shuffle_key1, shuffle_key2 = jax.random.split(key)
            
            perm1 = jax.random.permutation(shuffle_key1, batch_size)
            perm2 = jax.random.permutation(shuffle_key2, batch_size)
            
            shuffled_action = action[perm1]
            shuffled_next_obs = next_obs[perm2]
            
            # Forward pass with MI computation
            (state_emb, action_emb, next_state_emb, predicted_next_emb,
             mi_action_joint, mi_action_marginal, mi_obs_joint, mi_obs_marginal) = model.apply(
                params, obs, action, next_obs, shuffled_action, shuffled_next_obs,
                method=model.forward_with_mi
            )
            
            # Dynamics loss
            dynamics_loss = compute_dynamics_loss(predicted_next_emb, next_state_emb)
            
            # MI losses
            mi_action_loss, mi_action_est = compute_mine_loss(mi_action_joint, mi_action_marginal)
            mi_obs_loss, mi_obs_est = compute_mine_loss(mi_obs_joint, mi_obs_marginal)
            
            # Total loss
            total_loss = (
                dynamics_weight * dynamics_loss +
                mi_action_weight * mi_action_loss +
                mi_obs_weight * mi_obs_loss
            )
            
            return total_loss, (dynamics_loss, mi_action_loss, mi_obs_loss, 
                              mi_action_est, mi_obs_est, next_state_emb)
        
        # Compute gradients
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        dynamics_loss, mi_action_loss, mi_obs_loss, mi_action_est, mi_obs_est, next_state_emb = aux
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Update embedding pool (pool_size is captured from outer scope)
        batch_size = obs.shape[0]
        indices = (pool_idx + jnp.arange(batch_size)) % pool_size
        new_pool = embedding_pool.at[indices].set(next_state_emb)
        new_pool_idx = (pool_idx + batch_size) % pool_size
        
        metrics = {
            'total_loss': loss,
            'dynamics_loss': dynamics_loss,
            'mi_action_loss': mi_action_loss,
            'mi_obs_loss': mi_obs_loss,
            'mi_action_estimate': mi_action_est,
            'mi_obs_estimate': mi_obs_est,
        }
        
        return new_params, new_opt_state, new_pool, new_pool_idx, metrics
    
    return train_step


def create_emi_reward_fn_jit(config: Dict) -> Callable:
    """
    Create a JIT-compiled reward computation function that also updates the embedding pool.
    
    Args:
        config: EMI configuration dict
        
    Returns:
        jitted_reward_fn: JIT-compiled reward function
    """
    model = EMIModel(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        use_reconciler=config['use_reconciler']
    )
    pool_size = config['pool_size']
    
    @jax.jit
    def compute_rewards(params, embedding_pool, pool_idx, obs, action, next_obs,
                       diversity_coeff, residual_coeff, diversity_bandwidth):
        # Forward pass
        state_emb, action_emb, next_state_emb, predicted_next_emb = model.apply(
            params, obs, action, next_obs
        )
        
        # Compute reward components
        diversity_reward = compute_diversity_reward(
            next_state_emb, embedding_pool, diversity_bandwidth
        )
        residual_reward = compute_residual_reward(predicted_next_emb, next_state_emb)
        
        # Total intrinsic reward
        total_reward = diversity_coeff * diversity_reward + residual_coeff * residual_reward
        
        # Update embedding pool with new state embeddings
        new_pool, new_pool_idx = update_embedding_pool(
            embedding_pool, pool_idx, next_state_emb, pool_size
        )
        
        info = {
            'diversity_reward': diversity_reward,
            'residual_reward': residual_reward,
            'total': total_reward,
            'new_pool': new_pool,
            'new_pool_idx': new_pool_idx,
        }
        
        return total_reward, info
    
    return compute_rewards


# ============================================================================
# Factory function for integration with existing codebase
# ============================================================================

def create_optimized_emi_system(
    key: chex.PRNGKey,
    obs_dim: int,
    action_dim: int,
    embedding_dim: int = 32,
    hidden_dim: int = 128,
    use_reconciler: bool = True,
    learning_rate: float = 3e-4,
    pool_size: int = 10000,
    diversity_coeff: float = 0.1,
    residual_coeff: float = 0.1,
    diversity_bandwidth: float = 1.0,
    mi_action_weight: float = 0.05,
    mi_obs_weight: float = 0.05,
    dynamics_weight: float = 1.0,
) -> Tuple[Dict, Callable]:
    """
    Create an optimized EMI system with JIT-compiled functions.
    
    This is the main entry point for integrating EMI into the training pipeline.
    
    Args:
        key: Random key
        obs_dim: Observation dimension
        action_dim: Action dimension
        embedding_dim: Embedding space dimension
        hidden_dim: Hidden layer dimension
        use_reconciler: Whether to use reconciler network
        learning_rate: Learning rate
        pool_size: Embedding pool size for diversity reward
        diversity_coeff: Weight for diversity reward
        residual_coeff: Weight for residual error reward
        diversity_bandwidth: RBF kernel bandwidth
        mi_action_weight: Weight for action MI loss
        mi_obs_weight: Weight for observation MI loss
        dynamics_weight: Weight for dynamics loss
        
    Returns:
        emi_params: System parameters
        emi_apply_fn: Function to compute rewards (params, obs, action, next_obs) -> (rewards, info)
    """
    # Create base system
    emi_params, model_apply_fn, optimizer = create_emi_system(
        key, obs_dim, action_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        use_reconciler=use_reconciler,
        learning_rate=learning_rate,
        pool_size=pool_size
    )
    
    # Add additional config
    emi_params['config'].update({
        'diversity_coeff': diversity_coeff,
        'residual_coeff': residual_coeff,
        'diversity_bandwidth': diversity_bandwidth,
        'mi_action_weight': mi_action_weight,
        'mi_obs_weight': mi_obs_weight,
        'dynamics_weight': dynamics_weight,
    })
    
    # Create JIT-compiled functions - these are captured in closures, NOT stored in emi_params
    # This is critical because JAX functions are not valid JAX types for tracing
    config = emi_params['config']
    jit_reward_fn = create_emi_reward_fn_jit(config)
    jit_train_fn = create_emi_train_step_jit(optimizer, config)
    
    # NOTE: Do NOT store optimizer, jit_reward_fn, jit_train_fn in emi_params!
    # These are Python objects that cannot be traced by JAX and will cause errors
    # when emi_params is passed through jax.lax.scan.
    # Instead, we capture them in the emi_apply_fn closure below.
    
    def emi_apply_fn(
        params_dict,
        obs: chex.Array,
        action: chex.Array,
        next_obs: chex.Array,
        diversity_coeff: float = None,
        residual_coeff: float = None,
        return_components: bool = False,
        update_pool: bool = True,
        **kwargs
    ) -> Tuple[chex.Array, Dict]:
        """
        Compute EMI intrinsic rewards.
        
        Note: jit_reward_fn is captured in closure, not from params_dict.
        
        IMPORTANT: This function does NOT mutate params_dict to avoid JAX tracer leaks.
        The updated pool state is returned in info['new_pool'] and info['new_pool_idx'].
        The caller is responsible for updating the params_dict outside of traced contexts.
        
        Args:
            params_dict: EMI parameters dictionary (NOT mutated)
            obs: Observations [batch, obs_dim]
            action: Actions [batch, action_dim]
            next_obs: Next observations [batch, obs_dim]
            diversity_coeff: Override diversity coefficient
            residual_coeff: Override residual coefficient
            return_components: Whether to return components
            update_pool: Ignored (kept for API compatibility)
            
        Returns:
            rewards: Intrinsic rewards [batch]
            info: Component breakdown with new_pool and new_pool_idx for pool updates
        """
        cfg = params_dict['config']
        d_coeff = diversity_coeff if diversity_coeff is not None else cfg['diversity_coeff']
        r_coeff = residual_coeff if residual_coeff is not None else cfg['residual_coeff']
        
        # Use the JIT function captured in closure (not from params_dict)
        rewards, info = jit_reward_fn(
            params_dict['params'],
            params_dict['embedding_pool'],
            params_dict['pool_idx'],
            obs, action, next_obs,
            d_coeff, r_coeff, cfg['diversity_bandwidth']
        )
        
        # NOTE: We do NOT mutate params_dict here to avoid JAX tracer leaks.
        # The new pool state is in info['new_pool'] and info['new_pool_idx'].
        # The caller should update params_dict outside of traced contexts if needed.
        
        return rewards, info
    
    def emi_train_fn(
        params_dict,
        obs: chex.Array,
        action: chex.Array,
        next_obs: chex.Array,
        key: chex.PRNGKey,
    ) -> Tuple[Dict, Dict]:
        """
        Train EMI model for one step.
        
        Note: jit_train_fn and optimizer are captured in closure.
        
        Args:
            params_dict: EMI parameters dictionary
            obs: Observations [batch, obs_dim]
            action: Actions [batch, action_dim]
            next_obs: Next observations [batch, obs_dim]
            key: Random key for MI estimation
            
        Returns:
            updated_params_dict: Updated EMI parameters
            metrics: Training metrics
        """
        new_params, new_opt_state, new_pool, new_pool_idx, metrics = jit_train_fn(
            params_dict['params'],
            params_dict['opt_state'],
            params_dict['embedding_pool'],
            params_dict['pool_idx'],
            obs, action, next_obs, key
        )
        
        # Create updated params dict
        updated_params_dict = {
            'params': new_params,
            'opt_state': new_opt_state,
            'embedding_pool': new_pool,
            'pool_idx': new_pool_idx,
            'config': params_dict['config'],
        }
        
        return updated_params_dict, metrics
    
    # Return both apply and train functions
    # The emi_params dict contains only JAX-traceable data
    return emi_params, emi_apply_fn, emi_train_fn


# ============================================================================
# Wrapper class for compatibility with existing intrinsic reward interface
# ============================================================================

class EMIIntrinsicReward:
    """
    Wrapper class for EMI intrinsic rewards.
    
    Provides compatibility with the existing intrinsic reward interface.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        key: chex.PRNGKey = None,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        use_reconciler: bool = True,
        learning_rate: float = 3e-4,
        pool_size: int = 10000,
        diversity_coeff: float = 0.1,
        residual_coeff: float = 0.1,
        diversity_bandwidth: float = 1.0,
        mi_action_weight: float = 0.05,
        mi_obs_weight: float = 0.05,
        dynamics_weight: float = 1.0,
        mode: str = 'train',
        **kwargs
    ):
        """
        Initialize EMI intrinsic reward module.
        """
        import numpy as np
        import inspect
        
        # Resolve spaces
        def resolve_space(space, name):
            if callable(space):
                sig = inspect.signature(space)
                if len(sig.parameters) == 0:
                    return space()
                if hasattr(space, '__self__') and hasattr(space.__self__, name):
                    prop = getattr(space.__self__, name)
                    if not callable(prop):
                        return prop
                try:
                    return space()
                except Exception:
                    pass
                raise ValueError(f"Cannot resolve {name}")
            return space
        
        observation_space = resolve_space(observation_space, 'observation_space')
        action_space = resolve_space(action_space, 'action_space')
        
        # Extract dimensions
        if hasattr(observation_space, 'shape'):
            obs_dim = int(np.prod(observation_space.shape))
        elif hasattr(observation_space, 'n'):
            obs_dim = int(observation_space.n)
        else:
            raise ValueError(f"Cannot determine obs_dim")
            
        if hasattr(action_space, 'n'):
            action_dim = int(action_space.n)
        elif hasattr(action_space, 'shape'):
            action_dim = int(np.prod(action_space.shape))
        else:
            action_dim = 1
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mode = mode
        self.key = key if key is not None else jax.random.PRNGKey(0)
        
        # Store hyperparameters
        self.diversity_coeff = diversity_coeff
        self.residual_coeff = residual_coeff
        
        # Create EMI system
        self.emi_params, self.emi_apply_fn = create_optimized_emi_system(
            self.key, obs_dim, action_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            use_reconciler=use_reconciler,
            learning_rate=learning_rate,
            pool_size=pool_size,
            diversity_coeff=diversity_coeff,
            residual_coeff=residual_coeff,
            diversity_bandwidth=diversity_bandwidth,
            mi_action_weight=mi_action_weight,
            mi_obs_weight=mi_obs_weight,
            dynamics_weight=dynamics_weight
        )
        
        # Warm-up tracking
        self.count = 0
        self.warm_up_size = 100
        self.is_warmed_up = (mode == 'test')
    
    def compute_intrinsic_reward(
        self,
        obs: chex.Array,
        action: chex.Array,
        next_obs: chex.Array,
        return_components: bool = False
    ) -> Any:
        """
        Compute intrinsic reward for a transition.
        """
        # Ensure proper shapes
        obs_flat = jnp.reshape(jnp.asarray(obs), (-1, self.obs_dim))
        next_obs_flat = jnp.reshape(jnp.asarray(next_obs), (-1, self.obs_dim))
        
        # Handle action
        if jnp.isscalar(action) or (hasattr(action, 'shape') and action.shape == ()):
            action_flat = jax.nn.one_hot(int(action), self.action_dim)[None, :]
        elif len(jnp.asarray(action).shape) == 1 and action.shape[0] != self.action_dim:
            action_flat = jax.nn.one_hot(jnp.asarray(action), self.action_dim)
        else:
            action_flat = jnp.reshape(jnp.asarray(action), (-1, self.action_dim))
        
        # Check warm-up
        batch_size = obs_flat.shape[0]
        self.count += batch_size
        if self.count < self.warm_up_size and self.mode != 'test':
            if return_components:
                return 0.0, {'total': 0.0, 'diversity_reward': 0.0, 'residual_reward': 0.0}
            return 0.0
        
        # Compute rewards
        rewards, info = self.emi_apply_fn(
            self.emi_params, obs_flat, action_flat, next_obs_flat,
            return_components=return_components
        )
        
        if return_components:
            return float(jnp.mean(rewards)), info
        return float(jnp.mean(rewards))
    
    def train_on_batch(
        self,
        observations: chex.Array,
        actions: chex.Array,
        next_observations: chex.Array
    ) -> Dict[str, Any]:
        """
        Train EMI on a batch of transitions.
        """
        # Ensure proper shapes
        obs_flat = jnp.reshape(jnp.asarray(observations), (-1, self.obs_dim))
        next_obs_flat = jnp.reshape(jnp.asarray(next_observations), (-1, self.obs_dim))
        
        # Handle actions
        actions_arr = jnp.asarray(actions)
        if len(actions_arr.shape) == 1:
            action_flat = jax.nn.one_hot(actions_arr, self.action_dim)
        else:
            action_flat = actions_arr
        
        # Get training function and optimizer
        jit_train_fn = self.emi_params['jit_train_fn']
        config = self.emi_params['config']
        
        # Split key
        self.key, train_key = jax.random.split(self.key)
        
        # Train step
        new_params, new_opt_state, new_pool, new_pool_idx, metrics = jit_train_fn(
            self.emi_params['params'],
            self.emi_params['opt_state'],
            self.emi_params['embedding_pool'],
            self.emi_params['pool_idx'],
            obs_flat, action_flat, next_obs_flat,
            train_key,
            config['mi_action_weight'],
            config['mi_obs_weight'],
            config['dynamics_weight']
        )
        
        # Update params
        self.emi_params['params'] = new_params
        self.emi_params['opt_state'] = new_opt_state
        self.emi_params['embedding_pool'] = new_pool
        self.emi_params['pool_idx'] = new_pool_idx
        
        return {k: float(v) for k, v in metrics.items()}
    
    def reset(self):
        """Reset warm-up counter."""
        self.count = 0


# Export main components
__all__ = [
    'EMIModel',
    'StateEmbedding',
    'ActionEmbedding',
    'Reconciler',
    'MutualInfoDiscriminator',
    'create_emi_system',
    'create_optimized_emi_system',
    'compute_emi_intrinsic_rewards',
    'train_emi_step',
    'EMIIntrinsicReward',
]
