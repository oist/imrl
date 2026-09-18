"""
Count-Based Intrinsic Rewards (JAX)

Simple and effective exploration bonuses based on state visitation counts.

Core Principle:
- Novelty = 1 / sqrt(visit_count)
- Encourages visiting new states
- Much simpler than VAE-based approaches
- Proven effective in MiniGrid, Atari, etc.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
import chex


class CountBasedNovelty:
    """
    Simple count-based novelty bonus.
    
    Rewards = k / sqrt(count(state))
    
    Advantages:
    - Simple: no neural network training
    - Fast: hash table lookups
    - Effective: proven in many environments
    - Interpretable: directly rewards new states
    """
    
    def __init__(self, bonus_coef: float = 0.01, hash_bits: int = 32):
        """
        Args:
            bonus_coef: Scaling coefficient for bonus
            hash_bits: Number of bits for state hashing (controls granularity)
        """
        self.bonus_coef = bonus_coef
        self.hash_bits = hash_bits
        self.visit_counts = {}
    
    def hash_state(self, obs: chex.Array) -> int:
        """
        Hash observation to state ID.
        
        NOTE: This is NOT JAX-compatible (uses Python hash()).
        For JAX compatibility, use simple_hash_jax() instead.
        """
        # Simple hash: quantize and hash
        quantized = jnp.round(obs * 100).astype(jnp.int32)
        # Use Python's hash (NOT compatible with JAX tracing)
        # This will only work outside of jit/scan/vmap
        hash_val = hash(tuple(quantized.tolist()))
        return hash_val % (2 ** self.hash_bits)
    
    def simple_hash_jax(self, obs: chex.Array) -> chex.Array:
        """
        JAX-compatible simple hash using iterative rolling hash.
        
        This is a deterministic hash that works inside jit/scan.
        Uses iterative approach to avoid integer overflow with large observation vectors.
        Returns a scalar hash value as a JAX array.
        """
        # Quantize observation to integers (more aggressive for symbolic data)
        quantized = jnp.round(obs * 100).astype(jnp.int32)
        
        # Iterative rolling hash to avoid overflow: hash = ((hash * prime) + val) mod large_prime
        prime = jnp.int32(31)
        large_prime = jnp.int32(1000000007)  # Large prime to reduce collisions
        
        # Rolling hash with modulo at each step to prevent overflow
        def hash_step(carry, val):
            h = carry
            h = (h * prime + val) % large_prime
            return h, None
        
        hash_val, _ = jax.lax.scan(hash_step, jnp.int32(0), quantized)
        
        return jnp.abs(hash_val)
    
    def compute_bonus(self, obs: chex.Array) -> float:
        """
        Compute exploration bonus for observation (non-JAX version).
        
        This uses Python hash and dictionary, so it won't work inside jit.
        Use compute_bonus_simple() for JAX-compatible version.
        """
        state_id = self.hash_state(obs)
        
        # Get count (default to 0 for new states)
        count = self.visit_counts.get(state_id, 0)
        
        # Update count
        self.visit_counts[state_id] = count + 1
        
        # Bonus = k / sqrt(count + 1)
        # +1 ensures first visit gets finite bonus
        bonus = self.bonus_coef / jnp.sqrt(count + 1)
        
        return bonus
    
    def compute_bonus_with_counts(self, obs: chex.Array, count_table: Dict[int, int]) -> float:
        """
        Compute exploration bonus based on visit count (non-JAX version).
        
        Args:
            obs: Observation vector
            count_table: Dictionary tracking visit counts by hash
            
        Returns:
            bonus: Exploration bonus inversely proportional to sqrt(count)
        """
        obs_hash = self.simple_hash(obs)
        count = count_table.get(obs_hash, 0)
        
        # Classic count-based bonus: 1/sqrt(n+1)
        # This gives high bonus to novel states, diminishing returns for revisited states
        bonus = self.bonus_coef / jnp.sqrt(count + 1)
        return float(bonus)
    
    def compute_bonus_simple(self, obs: chex.Array) -> chex.Array:
        """
        JAX-compatible simple bonus: constant for all states.
        
        This is a simplified version that gives uniform exploration bonus.
        For true count-based exploration, you need external state tracking.
        """
        # Simple uniform bonus (no counting)
        # This encourages exploration uniformly
        return jnp.array(self.bonus_coef)
    
    def compute_batch_bonus(self, obs_batch: chex.Array) -> chex.Array:
        """
        Compute bonuses for batch of observations (non-JAX version).
        
        This won't work inside jit. Use compute_batch_bonus_simple() instead.
        """
        bonuses = []
        for obs in obs_batch:
            bonuses.append(self.compute_bonus(obs))
        return jnp.array(bonuses)
    
    def compute_batch_bonus_with_counts(self, obs_batch: chex.Array, count_table: Dict[int, int]) -> chex.Array:
        """
        Compute bonuses for batch of observations using count table (non-JAX).
        
        Args:
            obs_batch: Batch of observations [batch_size, obs_dim]
            count_table: Dictionary tracking visit counts by hash
            
        Returns:
            bonuses: Array of exploration bonuses [batch_size]
        """
        bonuses = []
        for obs in obs_batch:
            bonuses.append(self.compute_bonus_with_counts(obs, count_table))
        return jnp.array(bonuses)
    
    def compute_batch_bonus_simple(self, obs_batch: chex.Array) -> chex.Array:
        """
        JAX-compatible simple batch bonus: uniform exploration.
        
        Returns constant bonus for all observations in batch.
        For true counting, you need external state management.
        """
        # Uniform bonus for all states
        return jnp.full(obs_batch.shape[0], self.bonus_coef)
    
    def compute_batch_bonus_jax(self, obs_batch: chex.Array, count_table: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        JAX-compatible counting using fixed-size hash table.
        
        Args:
            obs_batch: Batch of observations [batch_size, obs_dim]
            count_table: Hash table of visit counts [table_size]
            
        Returns:
            bonuses: Exploration bonuses [batch_size]
            updated_count_table: Updated count table [table_size]
        """
        table_size = count_table.shape[0]
        
        def process_obs(carry, obs):
            count_table_state = carry
            # Hash the observation to table index
            hash_val = self.simple_hash_jax(obs)
            idx = hash_val % table_size
            
            # Get current count (safe indexing)
            count = count_table_state[idx]
            
            # Compute bonus: 1/sqrt(n+1)
            bonus = self.bonus_coef / jnp.sqrt(count + 1.0)
            
            # Update count table
            updated_table = count_table_state.at[idx].add(1.0)
            
            return updated_table, bonus
        
        # Scan over batch
        final_table, bonuses = jax.lax.scan(process_obs, count_table, obs_batch)
        
        return bonuses, final_table


def compute_count_based_bonus_jax(obs: chex.Array, count_table: chex.Array, 
                                   bonus_coef: float, table_size: int) -> Tuple[float, chex.Array]:
    """
    Standalone JAX function for computing count-based bonus.
    Can be used inside jit/scan without class instance.
    
    Args:
        obs: Single observation [obs_dim]
        count_table: Hash table [table_size]
        bonus_coef: Bonus coefficient
        table_size: Size of hash table
        
    Returns:
        bonus: Exploration bonus
        updated_table: Updated count table
    """
    # Iterative rolling hash (JAX-compatible using scan)
    quantized = jnp.round(obs * 100).astype(jnp.int32)
    prime = jnp.int32(31)
    large_prime = jnp.int32(1000000007)
    
    # Rolling hash with modulo at each step
    def hash_step(carry, val):
        h = carry
        h = (h * prime + val) % large_prime
        return h, None
    
    hash_val, _ = jax.lax.scan(hash_step, jnp.int32(0), quantized)
    
    # Map to table index
    idx = jnp.abs(hash_val) % table_size
    
    # Get count and compute bonus
    count = count_table[idx]
    bonus = bonus_coef / jnp.sqrt(count + 1.0)
    
    # Update table
    updated_table = count_table.at[idx].add(1.0)
    
    return bonus, updated_table


def compute_batch_bonus_jax(obs_batch: chex.Array, count_table: chex.Array,
                            bonus_coef: float, table_size: int) -> Tuple[chex.Array, chex.Array]:
    """
    Batch version using vmap for efficiency.
    
    Args:
        obs_batch: Batch of observations [batch_size, obs_dim]
        count_table: Hash table [table_size]
        bonus_coef: Bonus coefficient
        table_size: Size of hash table
        
    Returns:
        bonuses: Bonuses [batch_size]
        updated_table: Updated count table
    """
    def process_single(carry, obs):
        table = carry
        bonus, new_table = compute_count_based_bonus_jax(obs, table, bonus_coef, table_size)
        return new_table, bonus
    
    final_table, bonuses = jax.lax.scan(process_single, count_table, obs_batch)
    return bonuses, final_table


def create_count_based_system(bonus_coef: float = 0.01):
    """
    Create simple count-based exploration system.
    
    Args:
        bonus_coef: Coefficient for exploration bonus
        
    Returns:
        count_based_novelty: CountBasedNovelty instance
    """
    return CountBasedNovelty(bonus_coef=bonus_coef)
