#!/usr/bin/env python3
"""
Local evolutionary optimizer coordinator (no SLURM required).

Evolves intrinsic reward mixture weights (novelty, surprise, empowerment) for
any of imrl's supported environments using a generational genetic algorithm:

1. Initialize a population of genomes (weight triples), seeded with a few
   intentional counterexamples plus random individuals.
2. For each generation:
   a. Evaluate every genome in parallel (ProcessPoolExecutor), each genome
      trained across several seeds via `algos.ppo_jax.train_agent.train_agent`.
   b. Log progress (CSV + JSON summaries) and save a resumable checkpoint.
   c. Preserve the fittest 10% as elites; refill the rest via tournament
      selection + adaptive (1/5th success rule) Gaussian mutation.
3. Report the best genome found.

Usage:
    python -m evolutionary_optim.local_evolutionary_coordinator \\
        --env-id MiniGrid-FourRooms-TwoGoals-RandKey-ViewSize-3x3-v0 \\
        --population-size 20 \\
        --max-generations 20 \\
        --n-seeds 3 \\
        --n-timesteps 1000000 \\
        --n-workers 4 \\
        --results-dir local_evo_results
"""

import sys
import os
import json
import csv
import argparse
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set XLA flags for deterministic compilation so genome evaluations are reproducible
os.environ.setdefault('XLA_FLAGS',
    '--xla_gpu_deterministic_ops=true '
    '--xla_gpu_autotune_level=0 '
    '--xla_gpu_force_compilation_parallelism=1')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Help migrate from Gym -> Gymnasium
try:
    import gymnasium as _gymnasium
    sys.modules['gym'] = _gymnasium
except Exception:
    pass

from evolutionary_optim.genome import IntrinsicRewardGenome, default_weight_bounds, INT_REW_COEF_FIXED

BASE_SEED = 23


def generate_eval_seeds(n_seeds: int, base_seed: int = BASE_SEED) -> list:
    """Generate reproducible evaluation seeds from a base seed."""
    rng = np.random.RandomState(base_seed)
    return rng.randint(0, 2**32 - 1, n_seeds).tolist()


def setup_logging(results_dir: Path):
    log_file = results_dir / f"coordinator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def initialize_population(population_size: int, weight_min: float, weight_max: float, logger) -> List[IntrinsicRewardGenome]:
    """Initialize population with a few intentional counterexamples plus random individuals.

    Starting from edge cases (all-zero, all-max, single-component dominance) forces
    the optimizer to discover good configurations through evolution, rather than
    starting near a heuristically "balanced" local optimum.
    """
    logger.info(f"Initializing population of {population_size} genomes")
    logger.info("Seeding population with intentionally extreme configurations (counterexamples)")

    def genome(genes):
        g = IntrinsicRewardGenome({**genes, 'int_rew_coef': INT_REW_COEF_FIXED}, weight_min=weight_min, weight_max=weight_max)
        g.repair_constraints()
        return g

    seeded_configs = [
        {'novelty_weight': weight_min, 'surprise_weight': weight_min, 'empowerment_weight': weight_min},
        {'novelty_weight': weight_max, 'surprise_weight': weight_max, 'empowerment_weight': weight_max},
        {'novelty_weight': weight_max, 'surprise_weight': weight_min, 'empowerment_weight': weight_min},
        {'novelty_weight': weight_min, 'surprise_weight': weight_max, 'empowerment_weight': weight_min},
        {'novelty_weight': weight_min, 'surprise_weight': weight_min, 'empowerment_weight': weight_max},
        {'novelty_weight': weight_max * 0.9, 'surprise_weight': weight_max * 0.5, 'empowerment_weight': weight_max * 0.1},
        {'novelty_weight': weight_min * 1.1, 'surprise_weight': weight_min * 1.1, 'empowerment_weight': weight_min * 2.0},
        {'novelty_weight': weight_max * 0.8, 'surprise_weight': weight_min * 1.2, 'empowerment_weight': weight_max * 0.2},
    ]

    population = []
    for genes in seeded_configs:
        if len(population) >= population_size // 3:
            break
        population.append(genome(genes))
    n_seeded = len(population)

    while len(population) < population_size:
        population.append(IntrinsicRewardGenome(weight_min=weight_min, weight_max=weight_max))

    logger.info(f"Population initialized with {len(population)} individuals "
                f"({n_seeded} seeded, {len(population) - n_seeded} random)")
    return population


def evaluate_genome_worker(genome_data: dict, config: dict) -> dict:
    """Evaluate a single genome across multiple seeds. Runs in a separate process."""
    genome_id = genome_data['genome_id']
    generation = genome_data['generation']
    genes = genome_data['genes']
    worker_id = genome_data.get('worker_id', 0)

    # Set GPU device for this worker to avoid memory conflicts between workers
    n_gpus = config.get('n_gpus', 1)
    if n_gpus > 0:
        gpu_id = worker_id % n_gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        os.environ['JAX_PLATFORMS'] = 'cpu'

    eval_seeds = config['eval_seeds']
    n_timesteps = config['n_timesteps']
    env_id = config['env_id']
    n_envs = config.get('n_envs', 128)
    fitness_metric = config.get('fitness_metric', 'mean_reward')
    vae_kwargs = config.get('vae_kwargs', {})

    # Import training code inside the worker process, after device env vars are set
    sys.path.insert(0, str(PROJECT_ROOT))
    import tempfile
    import shutil
    from algos.ppo_jax.train_agent import train_agent

    seed_rewards = []
    seed_large_goal_counts = []
    episode_lengths = []

    for seed in eval_seeds:
        temp_dir = tempfile.mkdtemp(prefix=f'evo_gen{generation}_genome{genome_id}_seed{seed}_')
        try:
            result = train_agent(
                env_name=env_id,
                reward_type='combined',
                seed=seed,
                total_timesteps=n_timesteps,
                novelty_weight=genes['novelty_weight'],
                surprise_weight=genes['surprise_weight'],
                empowerment_weight=genes['empowerment_weight'],
                int_rew_coef=genes['int_rew_coef'],
                NUM_ENVS=n_envs,
                n_eval_episodes=0,
                output_dir=temp_dir,
                USE_WANDB=False,
                save_policy=False,
                save_best_policy=False,
                record_episodes=False,
                DEBUG=False,
                USE_DUAL_VALUE_HEADS=False,
                INTRINSIC_WARMUP_RATIO=0.0,
                **vae_kwargs,
            )

            mean_reward = None
            if result:
                if result.get('reported_reward') is not None:
                    mean_reward = float(result['reported_reward'])
                elif result.get('final_mean_reward') is not None:
                    mean_reward = float(result['final_mean_reward'])
                elif result.get('eval_mean_reward') is not None:
                    mean_reward = float(result['eval_mean_reward'])

            if mean_reward is not None:
                seed_rewards.append(mean_reward)
            else:
                print(f"Warning: No valid reward metric for genome {genome_id}, seed {seed}")

            large_goal_count = 0.0
            metrics_file = os.path.join(temp_dir, 'metrics.jsonl')
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        for line in f:
                            data = json.loads(line)
                            if 'large_reward_goal_rate' in data:
                                # Sum of per-batch rates approximates total times the goal was reached
                                large_goal_count += data['large_reward_goal_rate']
                            if 'rollout/ep_len_mean' in data:
                                episode_lengths.append(data['rollout/ep_len_mean'])
                            elif 'episode_length' in data:
                                episode_lengths.append(data['episode_length'])
                except Exception as e:
                    print(f"Warning: could not parse metrics for genome {genome_id}, seed {seed}: {e}")
            seed_large_goal_counts.append(large_goal_count)

        except Exception as e:
            print(f"Error evaluating genome {genome_id}, seed {seed}: {e}")
            traceback.print_exc()
            seed_large_goal_counts.append(0.0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if len(seed_rewards) == 0:
        fitness = -np.inf
        objectives = {
            'error': 'No valid seeds',
            'n_seeds': 0,
            'mean_reward': 0.0,
            'large_goal_count': 0.0,
            'fitness_metric': fitness_metric,
        }
    else:
        mean_reward = float(np.mean(seed_rewards))
        std_reward = float(np.std(seed_rewards))
        total_large_goal_count = float(np.sum(seed_large_goal_counts))
        fitness = total_large_goal_count if fitness_metric == 'large_goal_count' else mean_reward
        objectives = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'large_goal_count': total_large_goal_count,
            'fitness_metric': fitness_metric,
            'n_seeds': len(seed_rewards),
            'seed_rewards': seed_rewards,
            'seed_large_goal_counts': seed_large_goal_counts,
            'episode_lengths': episode_lengths,
        }

    return {
        'genome_id': genome_id,
        'generation': generation,
        'fitness': fitness,
        'objectives': objectives,
        'genes': genes,
    }


def evaluate_population(population: List[IntrinsicRewardGenome], generation: int,
                         config: dict, n_workers: int, logger) -> Tuple[List[float], List[Dict]]:
    """Evaluate entire population in parallel using local worker processes."""
    logger.info(f"Evaluating population of {len(population)} genomes with {n_workers} workers")

    genome_data_list = [
        {'genome_id': i, 'generation': generation, 'genes': genome.genes, 'worker_id': i % n_workers}
        for i, genome in enumerate(population)
    ]

    fitnesses = [None] * len(population)
    objectives_list = [None] * len(population)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_genome = {
            executor.submit(evaluate_genome_worker, gd, config): gd['genome_id']
            for gd in genome_data_list
        }
        for future in as_completed(future_to_genome):
            genome_id = future_to_genome[future]
            try:
                result = future.result()
                population[result['genome_id']].fitness = result['fitness']
                population[result['genome_id']].objectives = result['objectives']
                fitnesses[genome_id] = result['fitness']
                objectives_list[genome_id] = result['objectives']
                logger.info(f"Genome {genome_id}: fitness={result['fitness']:.4f}")
            except Exception as e:
                logger.error(f"Genome {genome_id} evaluation failed: {e}")
                fitnesses[genome_id] = -np.inf
                objectives_list[genome_id] = {'error': str(e), 'n_seeds': 0}
                population[genome_id].fitness = -np.inf

    logger.info("Population evaluation complete")
    valid_fitnesses = [f for f in fitnesses if not np.isinf(f)]
    if valid_fitnesses:
        logger.info(f"Fitness range: [{min(valid_fitnesses):.4f}, {max(valid_fitnesses):.4f}]")
        logger.info(f"Mean fitness: {np.mean(valid_fitnesses):.4f}")

    return fitnesses, objectives_list


def tournament_selection(population: List[IntrinsicRewardGenome], tournament_size: int = 4) -> IntrinsicRewardGenome:
    """Select an individual via tournament selection (sampled without replacement)."""
    tournament = np.random.choice(population, size=tournament_size, replace=False)
    return max(tournament, key=lambda x: x.fitness if not np.isinf(x.fitness) else -1e10)


def save_checkpoint(checkpoint_path: Path, generation: int, population: List[IntrinsicRewardGenome],
                     weight_min: float, weight_max: float, logger):
    """Save checkpoint with current state for resume capability."""
    try:
        checkpoint = {
            'generation': generation,
            'weight_min': weight_min,
            'weight_max': weight_max,
            'population': [
                {
                    'genes': genome.genes,
                    'fitness': float(genome.fitness) if genome.fitness is not None and not np.isinf(genome.fitness) else None,
                    'objectives': genome.objectives if hasattr(genome, 'objectives') else {},
                } for genome in population
            ],
            'timestamp': datetime.now().isoformat(),
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_path: Path, logger) -> Tuple[int, List[IntrinsicRewardGenome], float, float]:
    """Load checkpoint and return (generation, population, weight_min, weight_max)."""
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    generation = checkpoint['generation']
    weight_min = checkpoint.get('weight_min', 0.0)
    weight_max = checkpoint.get('weight_max', 0.01)
    population = []
    for genome_data in checkpoint['population']:
        genome = IntrinsicRewardGenome(genome_data['genes'], weight_min=weight_min, weight_max=weight_max)
        if genome_data['fitness'] is not None:
            genome.fitness = genome_data['fitness']
        if genome_data.get('objectives'):
            genome.objectives = genome_data['objectives']
        population.append(genome)

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"  Generation: {generation}")
    logger.info(f"  Population size: {len(population)}")
    evaluated = sum(1 for g in population if g.fitness is not None and not np.isinf(g.fitness))
    logger.info(f"  Evaluated individuals: {evaluated}/{len(population)}")

    return generation, population, weight_min, weight_max


def create_next_generation(population: List[IntrinsicRewardGenome], population_size: int,
                            mutation_strength: float, logger) -> List[IntrinsicRewardGenome]:
    """Elitism (top 10%) + tournament selection & adaptive mutation for the rest."""
    logger.info(f"Creating next generation with mutation_strength={mutation_strength:.4f}")

    sorted_pop = sorted(population, key=lambda x: x.fitness if not np.isinf(x.fitness) else -1e10, reverse=True)

    n_elite = max(1, population_size // 10)
    next_generation = [genome.copy() for genome in sorted_pop[:n_elite]]
    logger.info(f"Preserved {n_elite} elite individuals")

    while len(next_generation) < population_size:
        parent = tournament_selection(population)
        offspring = parent.copy()
        offspring.mutate(mutation_rate=0.3, mutation_strength=mutation_strength)
        offspring.repair_constraints()
        next_generation.append(offspring)

    logger.info(f"Created generation with {len(next_generation)} individuals")
    return next_generation


def save_generation_summary(generation: int, population: List[IntrinsicRewardGenome], base_dir: Path, logger):
    """Save a JSON summary of a generation's results."""
    summary_dir = base_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    fitnesses = [g.fitness for g in population if not np.isinf(g.fitness)]
    if not fitnesses:
        logger.warning(f"Generation {generation}: No valid fitness values")
        return

    best_genome = max(population, key=lambda x: x.fitness if not np.isinf(x.fitness) else -1e10)
    summary = {
        'generation': generation,
        'timestamp': datetime.now().isoformat(),
        'statistics': {
            'best_fitness': float(max(fitnesses)),
            'mean_fitness': float(np.mean(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'worst_fitness': float(min(fitnesses)),
            'n_valid': len(fitnesses),
            'n_total': len(population),
        },
        'best_genome': {
            'genes': best_genome.genes,
            'fitness': float(best_genome.fitness),
            'objectives': best_genome.objectives,
        },
    }

    summary_file = summary_dir / f"gen_{generation}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved generation summary: {summary_file}")
    logger.info(f"Generation {generation}: Best={summary['statistics']['best_fitness']:.3f}, "
                f"Mean={summary['statistics']['mean_fitness']:.3f}, "
                f"Std={summary['statistics']['std_fitness']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Local evolutionary optimizer for intrinsic reward mixture weights (no SLURM)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--population-size', type=int, default=20)
    parser.add_argument('--max-generations', type=int, default=20)
    parser.add_argument('--n-seeds', type=int, default=3, help='Number of evaluation seeds per genome')
    parser.add_argument('--n-timesteps', type=int, default=1000000, help='Training timesteps per seed')
    parser.add_argument('--env-id', type=str,
                         default='MiniGrid-FourRooms-TwoGoals-RandKey-ViewSize-3x3-v0',
                         help='Environment ID (any imrl-supported MiniGrid or Craftax env)')
    parser.add_argument('--fitness-metric', type=str, default='mean_reward',
                         choices=['mean_reward', 'large_goal_count'],
                         help='mean_reward (episode return) or large_goal_count (times the large reward goal was reached)')
    parser.add_argument('--target-fitness', type=float, default=9.5,
                         help='Early-stop once best mean_reward fitness reaches this value (env-reward-scale dependent)')
    parser.add_argument('--weight-min', type=float, default=None,
                         help='Override the minimum intrinsic reward weight (default: environment-specific)')
    parser.add_argument('--weight-max', type=float, default=None,
                         help='Override the maximum intrinsic reward weight (default: environment-specific)')

    # Local execution parameters
    parser.add_argument('--n-workers', type=int, default=4, help='Number of parallel worker processes')
    parser.add_argument('--n-envs', type=int, default=128, help='Number of parallel environments per JAX training run')
    parser.add_argument('--n-gpus', type=int, default=1, help='Number of GPUs available (workers distributed across them)')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU-only execution')

    # Intrinsic reward network hyperparameters (kept small for evaluation speed)
    parser.add_argument('--vae-hidden-dim', type=int, default=128)
    parser.add_argument('--vae-latent-dim', type=int, default=32)
    parser.add_argument('--vae-planning-horizon', type=int, default=3)
    parser.add_argument('--vae-num-action-sequences', type=int, default=10)

    parser.add_argument('--results-dir', type=str, default='local_evo_results')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                         help='Path to checkpoint file (default: results_dir/checkpoint.json)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(results_dir)

    logger.info("=" * 80)
    logger.info("Local Evolutionary Optimization Starting")
    logger.info("=" * 80)
    logger.info(f"Environment: {args.env_id}")
    logger.info(f"Population size: {args.population_size}")
    logger.info(f"Generations: {args.max_generations}")
    logger.info(f"Seeds per genome: {args.n_seeds}")
    logger.info(f"Workers: {args.n_workers}")
    logger.info(f"Timesteps per seed: {args.n_timesteps}")
    logger.info(f"Parallel environments: {args.n_envs}")
    logger.info(f"Fitness metric: {args.fitness_metric}")
    logger.info(f"Results directory: {results_dir}")
    logger.info("=" * 80)

    env_weight_min, env_weight_max = default_weight_bounds(args.env_id)
    weight_min = args.weight_min if args.weight_min is not None else env_weight_min
    weight_max = args.weight_max if args.weight_max is not None else env_weight_max
    logger.info(f"Intrinsic reward weight bounds: [{weight_min}, {weight_max}]")

    eval_seeds = generate_eval_seeds(args.n_seeds, BASE_SEED)
    config = {
        'eval_seeds': eval_seeds,
        'n_timesteps': args.n_timesteps,
        'env_id': args.env_id,
        'n_envs': args.n_envs,
        'n_gpus': 0 if args.cpu_only else args.n_gpus,
        'fitness_metric': args.fitness_metric,
        'vae_kwargs': {
            'hidden_dim': args.vae_hidden_dim,
            'latent_dim': args.vae_latent_dim,
            'planning_horizon': args.vae_planning_horizon,
            'num_action_sequences': args.vae_num_action_sequences,
        },
    }
    logger.info(f"Using BASE_SEED={BASE_SEED} to generate {args.n_seeds} reproducible seeds: {eval_seeds}")

    if args.cpu_only:
        logger.info("CPU-only mode enabled (no GPU)")
    elif args.n_gpus < args.n_workers:
        logger.warning(f"More workers ({args.n_workers}) than GPUs ({args.n_gpus}); "
                        f"workers will share GPUs, which may cause memory issues.")

    # Initialize progress CSV
    csv_file = results_dir / 'progress.csv'
    if not csv_file.exists():
        with open(csv_file, 'w') as f:
            headers = [
                'timestamp', 'generation', 'individual_id',
                'novelty_weight', 'surprise_weight', 'empowerment_weight', 'int_rew_coef',
                'fitness', 'mean_reward', 'large_goal_count', 'fitness_metric', 'std_reward',
                'is_elite', 'is_best_individual',
                'generation_best_fitness', 'generation_mean_fitness', 'generation_std_fitness',
                'population_diversity',
                'weight_sum', 'normalized_novelty_weight', 'normalized_surprise_weight', 'normalized_empowerment_weight',
                'mutation_strength',
                'total_evaluations_accumulated',
            ]
            f.write(','.join(headers) + '\n')
        logger.info(f"Initialized progress CSV: {csv_file}")

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else (results_dir / 'checkpoint.json')

    start_generation = 0
    if args.resume and checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        start_generation, population, weight_min, weight_max = load_checkpoint(checkpoint_path, logger)
        evaluated = sum(1 for g in population if g.fitness is not None and not np.isinf(g.fitness))
        if evaluated >= len(population):
            start_generation += 1
        logger.info(f"Resumed with weight bounds: [{weight_min}, {weight_max}]")
    else:
        population = initialize_population(args.population_size, weight_min, weight_max, logger)

    # Adaptive mutation (1/5th success rule; Rechenberg, 1973)
    mutation_strength = 0.05
    success_history = []
    history_window = 5
    target_success_rate = 0.2
    adaptation_factor = 1.2
    min_mutation_strength = 0.005
    max_mutation_strength = 0.2

    logger.info("Adaptive Mutation (1/5th Success Rule) Initialized: "
                f"sigma={mutation_strength}, target_success_rate={target_success_rate}, "
                f"adaptation_factor={adaptation_factor}, history_window={history_window}")

    for generation in range(start_generation, args.max_generations):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"GENERATION {generation} (of {args.max_generations - 1})")
        logger.info("=" * 80)

        fitnesses, objectives = evaluate_population(population, generation, config, args.n_workers, logger)

        current_time = datetime.now().isoformat()
        valid_fitnesses = [f for f in fitnesses if f != -np.inf]
        generation_best_fitness = max(valid_fitnesses) if valid_fitnesses else -np.inf
        generation_mean_fitness = np.mean(valid_fitnesses) if valid_fitnesses else 0.0
        generation_std_fitness = np.std(valid_fitnesses) if len(valid_fitnesses) > 1 else 0.0

        weight_matrix = np.array([
            [g.genes['novelty_weight'], g.genes['surprise_weight'], g.genes['empowerment_weight']]
            for g in population
        ])
        population_diversity = np.mean(np.std(weight_matrix, axis=0))

        sorted_pop = sorted(population, key=lambda x: x.fitness if x.fitness != -np.inf else -1e10, reverse=True)
        elite_size = max(1, int(len(population) * 0.2))
        best_genome = sorted_pop[0]

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, (genome, fitness, obj) in enumerate(zip(population, fitnesses, objectives)):
                is_elite = genome in sorted_pop[:elite_size]
                is_best = (genome.fitness == best_genome.fitness)

                weight_sum = (genome.genes['novelty_weight'] + genome.genes['surprise_weight']
                              + genome.genes['empowerment_weight'])
                norm_novelty = genome.genes['novelty_weight'] / weight_sum if weight_sum > 0 else 0
                norm_surprise = genome.genes['surprise_weight'] / weight_sum if weight_sum > 0 else 0
                norm_empowerment = genome.genes['empowerment_weight'] / weight_sum if weight_sum > 0 else 0

                mean_reward = obj.get('mean_reward', fitness if fitness != -np.inf else 0.0)
                large_goal_count = obj.get('large_goal_count', 0.0)
                fitness_metric_used = obj.get('fitness_metric', args.fitness_metric)
                std_reward = obj.get('std_reward', 0.0)

                writer.writerow([
                    current_time, generation, i,
                    genome.genes['novelty_weight'], genome.genes['surprise_weight'],
                    genome.genes['empowerment_weight'], genome.genes['int_rew_coef'],
                    fitness, mean_reward, large_goal_count, fitness_metric_used, std_reward,
                    is_elite, is_best,
                    generation_best_fitness, generation_mean_fitness, generation_std_fitness,
                    population_diversity,
                    weight_sum, norm_novelty, norm_surprise, norm_empowerment,
                    mutation_strength,
                    generation * len(population) + i,
                ])

        logger.info(f"Progress logged to CSV: {csv_file}")
        save_generation_summary(generation, population, results_dir, logger)
        save_checkpoint(checkpoint_path, generation, population, weight_min, weight_max, logger)

        # Adaptive mutation update (1/5th success rule)
        if generation > 0:
            previous_best_fitness = success_history[-1]['best_fitness'] if success_history else -np.inf
            success = 1 if generation_best_fitness > previous_best_fitness else 0
            success_history.append({'generation': generation, 'success': success, 'best_fitness': generation_best_fitness})
            if len(success_history) > history_window:
                success_history = success_history[-history_window:]

            if len(success_history) >= min(3, history_window):
                success_rate = sum(h['success'] for h in success_history) / len(success_history)
                old_strength = mutation_strength
                if success_rate > target_success_rate:
                    mutation_strength = min(mutation_strength * adaptation_factor, max_mutation_strength)
                    adjustment = "INCREASED (exploring more)"
                elif success_rate < target_success_rate:
                    mutation_strength = max(mutation_strength / adaptation_factor, min_mutation_strength)
                    adjustment = "DECREASED (exploiting more)"
                else:
                    adjustment = "UNCHANGED (optimal)"
                logger.info(f"Adaptive Mutation Update: success_rate={success_rate:.3f} "
                            f"(target={target_success_rate}), sigma {old_strength:.4f} -> {mutation_strength:.4f} ({adjustment})")
        else:
            success_history.append({'generation': generation, 'success': 0, 'best_fitness': generation_best_fitness})

        # Early stopping
        best_fitness = max(fitnesses) if fitnesses else -np.inf
        if args.fitness_metric == 'large_goal_count':
            all_episode_lengths = [l for obj in objectives for l in obj.get('episode_lengths', [])]
            avg_episode_length = float(np.mean(all_episode_lengths)) if all_episode_lengths else 200.0
            expected_episodes_per_seed = max(100, int(args.n_timesteps / avg_episode_length))
            optimal_threshold = args.n_seeds * expected_episodes_per_seed * 0.8
            if best_fitness >= optimal_threshold:
                logger.info(f"Optimal solution found! Best fitness (large_goal_count): "
                            f"{best_fitness:.1f} >= {optimal_threshold:.1f}")
                break
        else:
            if best_fitness >= args.target_fitness:
                logger.info(f"Optimal solution found! Best fitness (mean_reward): {best_fitness:.3f}")
                break

        if generation < args.max_generations - 1:
            population = create_next_generation(population, args.population_size, mutation_strength, logger)

    logger.info("")
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETED")
    logger.info("=" * 80)

    best_genome = max(population, key=lambda x: x.fitness if not np.isinf(x.fitness) else -1e10)
    logger.info(f"Best genome: fitness={best_genome.fitness:.3f}, genes={best_genome.genes}")

    final_results = {
        'best_genome': {
            'genes': best_genome.genes,
            'fitness': float(best_genome.fitness),
            'objectives': best_genome.objectives,
        },
        'configuration': {
            'env_id': args.env_id,
            'population_size': args.population_size,
            'max_generations': args.max_generations,
            'n_seeds': args.n_seeds,
            'n_timesteps': args.n_timesteps,
            'n_workers': args.n_workers,
            'weight_min': weight_min,
            'weight_max': weight_max,
        },
        'timestamp': datetime.now().isoformat(),
    }
    final_results_file = results_dir / "final_results.json"
    with open(final_results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Final results saved to: {final_results_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
