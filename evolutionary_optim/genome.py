"""
Genome representation for evolutionary optimization of intrinsic reward mixture
weights (novelty, surprise, empowerment).

Ported from the original per-environment genome files (minigrid/craftax), which
were identical except for the valid weight range. Here that range is resolved
per environment via `default_weight_bounds`, keeping a single implementation.
"""

from typing import Dict, Optional, Tuple

import numpy as np

INT_REW_COEF_FIXED = 1.0  # Fixed intrinsic reward coefficient


def default_weight_bounds(env_name: str) -> Tuple[float, float]:
    """Environment-specific (weight_min, weight_max) bounds.

    MiniGrid's sparse reward structure requires conservative intrinsic reward
    scaling to avoid reward-shaping dominance; Craftax's richer reward signal
    tolerates much larger intrinsic contributions.
    """
    if "MiniGrid" in env_name:
        return 0.0, 0.01
    return 0.0, 1.0


class IntrinsicRewardGenome:
    """Represents a genome encoding intrinsic reward mixture weights."""

    def __init__(
        self,
        genes: Optional[Dict[str, float]] = None,
        weight_min: float = 0.0,
        weight_max: float = 0.01,
    ):
        self.weight_min = weight_min
        self.weight_max = weight_max

        if genes is None:
            self.genes = self._initialize_random_genes()
        else:
            self.genes = genes.copy()

        self.fitness = -np.inf
        self.objectives = {}
        self.generation = 0
        self.evaluation_time = 0.0
        self.diversity_score = 0.0

        # Caching for fast lookups
        self._hash = None
        self._vector_cache = None

    def __hash__(self) -> int:
        if self._hash is None:
            gene_tuple = tuple(sorted(self.genes.items()))
            self._hash = hash(gene_tuple)
        return self._hash

    def __eq__(self, other) -> bool:
        return isinstance(other, IntrinsicRewardGenome) and hash(self) == hash(other)

    def to_vector(self) -> np.ndarray:
        if self._vector_cache is None:
            self._vector_cache = np.array([
                self.genes['novelty_weight'],
                self.genes['surprise_weight'],
                self.genes['empowerment_weight'],
            ])
        return self._vector_cache.copy()

    def _invalidate_cache(self):
        self._hash = None
        self._vector_cache = None

    def _initialize_random_genes(self) -> Dict[str, float]:
        """Uniform random sampling within bounds for unbiased search coverage."""
        weights = [
            np.random.uniform(self.weight_min, self.weight_max) for _ in range(3)
        ]
        return {
            'novelty_weight': float(weights[0]),
            'surprise_weight': float(weights[1]),
            'empowerment_weight': float(weights[2]),
            'int_rew_coef': INT_REW_COEF_FIXED,
        }

    def copy(self):
        new_genome = IntrinsicRewardGenome(
            self.genes, weight_min=self.weight_min, weight_max=self.weight_max
        )
        new_genome.fitness = self.fitness
        new_genome.objectives = self.objectives.copy()
        new_genome.generation = self.generation
        new_genome.evaluation_time = self.evaluation_time
        new_genome.diversity_score = self.diversity_score
        return new_genome

    def mutate(self, mutation_rate: float = 0.3, mutation_strength: float = 0.05):
        """Mutate genome with given rate and strength.

        Uses absolute mutation strength (scaled by range) rather than relative
        mutation, so weights can escape from boundary values (0.0 or max).
        """
        self._invalidate_cache()

        weight_range = self.weight_max - self.weight_min
        for key in ('novelty_weight', 'surprise_weight', 'empowerment_weight'):
            if np.random.random() < mutation_rate:
                noise = np.random.normal(0, mutation_strength * weight_range)
                self.genes[key] = np.clip(
                    self.genes[key] + noise, self.weight_min, self.weight_max
                )

    def repair_constraints(self):
        """Hard-clip constraint violations back into the valid range."""
        self._invalidate_cache()

        for key in ('novelty_weight', 'surprise_weight', 'empowerment_weight'):
            self.genes[key] = float(np.clip(self.genes[key], self.weight_min, self.weight_max))
        self.genes['int_rew_coef'] = INT_REW_COEF_FIXED

    def __repr__(self):
        return (
            f"IntrinsicRewardGenome(fitness={self.fitness:.3f}, "
            f"novelty={self.genes['novelty_weight']:.5f}, "
            f"surprise={self.genes['surprise_weight']:.5f}, "
            f"empowerment={self.genes['empowerment_weight']:.5f})"
        )
