import argparse
import logging
import os
import random
import numpy as np
# NOTE: jax import moved to after device configuration in main()

def set_global_seeds(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        logging.info(f"Global random seed set to {seed}")

def parse_args():
    def int_or_scientific(value):
        """Convert string to int, handling scientific notation."""
        try:
            return int(float(value))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid number: {value}")
    
    parser = argparse.ArgumentParser(
        description='Train PPO agent with intrinsic rewards (JAX/Flax)'
    )
    parser.add_argument('--env', type=str, default='Craftax-Classic-Symbolic-v1',
                        help='Environment ID (e.g., Craftax-Classic-Symbolic-v1)')
    parser.add_argument('--reward-type', type=str, default='novelty',
                        choices=['novelty', 'surprise', 'empowerment', 'combined', 'rnd', 'icm', 'emi', 'vime'],
                        help='Type of intrinsic reward to use')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--timesteps', type=int_or_scientific, default=100000,
                        help='Total timesteps for training (supports scientific notation, e.g., 1e6)')
    parser.add_argument('--save-model', type=str, default=None,
                        help='Path to save the trained model')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for JAX (auto, cpu, cuda)')
    parser.add_argument('--int-rew-coef', type=float, default=1.0,
                        help='Global coefficient for intrinsic reward scaling (default: 1.0)')
    parser.add_argument('--novelty-weight', type=float, default=0.01,
                        help='Weight for novelty intrinsic reward (default: 0.01). '
                             'With min-max normalization to [0,1], intrinsic rewards should be ~10-50%% of extrinsic. '
                             'Craftax extrinsic reward mean is ~0.02-0.04, so total intrinsic ~0.01-0.02 is good.')
    parser.add_argument('--surprise-weight', type=float, default=0.01,
                        help='Weight for surprise intrinsic reward (default: 0.01). '
                             'See --novelty-weight for scaling guidance.')
    parser.add_argument('--empowerment-weight', type=float, default=0.01,
                        help='Weight for empowerment intrinsic reward (default: 0.01). '
                             'See --novelty-weight for scaling guidance.')
    
    # Information-Theoretic Intrinsic Reward System Parameters
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension for neural networks (default: 256)')
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Latent dimension for VAE (default: 64)')
    parser.add_argument('--planning-horizon', type=int, default=5,
                        help='Planning horizon for empowerment (default: 5)')
    parser.add_argument('--num-action-sequences', type=int, default=16,
                        help='Number of action sequences for empowerment (default: 16)')
    parser.add_argument('--vae-learning-rate', type=float, default=5e-4,
                        help='Learning rate for VAE training (default: 5e-4)')
    
    # Count-Based Exploration Parameters
    parser.add_argument('--use-count-based-novelty', action='store_true',
                        help='Use count-based exploration instead of VAE for NOVELTY ONLY. '
                             'Automatically disables surprise/empowerment (which require VAE training). '
                             'Good for symbolic/discrete environments like Craftax.')
    parser.add_argument('--count-based-bonus-coef', type=float, default=None,
                        help='Scaling coefficient for count-based bonus. Auto-matches --novelty-weight if not specified.')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save logs and outputs')
    parser.add_argument('--save-policy', action='store_true',
                        help='If set, save the policy at the end of training (passed to PPO as SAVE_POLICY)')
    parser.add_argument('--save-best-policy', action='store_true',
                        help='If set, save the best policy during training (passed to PPO as SAVE_BEST_POLICY)')
    parser.add_argument('--terminate-on-reward', action='store_true',
                        help='Terminate episodes when a small or large achievement reward is given. '
                             'Default: True for MiniGrid (goal-based), False for Craftax (open-ended exploration)')
    parser.add_argument('--fixed-landscape', action='store_true',
                        help='If set, use a fixed procedurally-generated landscape for all episodes')
    parser.add_argument('--record-episodes', action='store_true',
                        help='If set, record videos of best episodes (full map + FOV) and log to wandb')
    
    # PPO Hyperparameters
    parser.add_argument('--num-repeats', type=int, default=1,
                        help='Number of training runs to repeat (default: 1)')
    parser.add_argument('--num-steps', type=int, default=64,
                        help='Number of environment steps per rollout (default: 64)')
    parser.add_argument('--num-envs', type=int, default=1024,
                        help='Number of parallel environments (default: 1024)')
    parser.add_argument('--num-minibatches', type=int, default=8,
                        help='Number of minibatches for PPO updates (default: 8)')
    parser.add_argument('--use-optimistic-resets', action='store_true', default=True,
                        help='Use optimistic resets for faster environment resets (default: True)')
    parser.add_argument('--no-optimistic-resets', dest='use_optimistic_resets', action='store_false',
                        help='Disable optimistic resets')
    parser.add_argument('--optimistic-reset-ratio', type=int, default=16,
                        help='Ratio for optimistic resets (default: 16)')
    parser.add_argument('--layer-size', type=int, default=512,
                        help='Size of hidden layers in the network (default: 512)')
    parser.add_argument('--anneal-lr', action='store_true', default=True,
                        help='Anneal learning rate during training (default: True)')
    parser.add_argument('--no-anneal-lr', dest='anneal_lr', action='store_false',
                        help='Disable learning rate annealing')
    parser.add_argument('--activation', type=str, default='tanh',
                        help='Activation function for policy networks (default: tanh)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping (default: 1.0)')
    parser.add_argument('--train-icm', action='store_true',
                        help='Train Intrinsic Curiosity Module (ICM) (default: False)')
    parser.add_argument('--icm-latent-size', type=int, default=32,
                        help='ICM latent size (default: 32)')
    parser.add_argument('--icm-layer-size', type=int, default=256,
                        help='Hidden layer size for ICM networks (default: 256)')
    parser.add_argument('--icm-lr', type=float, default=3e-4,
                        help='ICM optimizer learning rate (default: 3e-4)')
    parser.add_argument('--icm-forward-loss-coef', type=float, default=1.0,
                        help='ICM forward loss coefficient (default: 1.0)')
    parser.add_argument('--icm-inverse-loss-coef', type=float, default=1.0,
                        help='ICM inverse loss coefficient (default: 1.0)')
    parser.add_argument('--icm-reward-coeff', type=float, default=1.0,
                        help='ICM reward coefficient applied to intrinsic reward (default: 1.0)')
    parser.add_argument('--train-rnd', action='store_true',
                        help='Train Random Network Distillation (RND) exploration module (default: False)')
    parser.add_argument('--rnd-reward-coeff', type=float, default=1.0,
                        help='Intrinsic reward coefficient for RND (default: 1.0)')
    parser.add_argument('--rnd-layer-size', type=int, default=256,
                        help='RND network layer size (default: 256)')
    parser.add_argument('--rnd-output-size', type=int, default=512,
                        help='RND output embedding size (default: 512)')
    parser.add_argument('--rnd-lr', type=float, default=3e-4,
                        help='RND optimizer learning rate (default: 3e-4)')
    parser.add_argument('--rnd-loss-coeff', type=float, default=0.01,
                        help='RND distillation loss coefficient (default: 0.01)')
    parser.add_argument('--train-vime', action='store_true',
                        help='Train VIME (Variational Information Maximizing Exploration) exploration module (default: False)')
    parser.add_argument('--vime-hidden-dim', type=int, default=64,
                        help='Hidden layer size for the VIME Bayesian dynamics network (default: 64)')
    parser.add_argument('--vime-prior-std', type=float, default=0.5,
                        help='Std of the Gaussian prior over VIME dynamics network weights (default: 0.5)')
    parser.add_argument('--vime-step-size', type=float, default=0.01,
                        help='Gradient step size used to probe the information gain of a transition (default: 0.01)')
    parser.add_argument('--vime-out-sigma', type=float, default=1.0,
                        help='Std of the Gaussian likelihood for predicted next-observation deltas (default: 1.0)')
    parser.add_argument('--vime-lr', type=float, default=1e-3,
                        help='Learning rate for training the VIME dynamics network (default: 1e-3)')
    parser.add_argument('--vime-reward-coeff', type=float, default=1.0,
                        help='Intrinsic reward coefficient for VIME (default: 1.0)')
    parser.add_argument('--vime-kl-buffer-size', type=int, default=10000,
                        help='Size of the running buffer used to median-normalize VIME KL rewards (default: 10000)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor gamma (default: 0.99)')
    parser.add_argument('--gae-lambda', type=float, default=0.8,
                        help='GAE lambda parameter (default: 0.8)')
    parser.add_argument('--update-epochs', type=int, default=4,
                        help='Number of epochs for PPO updates (default: 4)')
    parser.add_argument('--clip-eps', type=float, default=0.2,
                        help='PPO clipping epsilon (default: 0.2)')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='Value function loss coefficient (default: 0.5)')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Entropy bonus coefficient (default: 0.01)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate (default: 2e-4)')
    parser.add_argument('--debug', action='store_true', default=True,
                        help='Enable debug mode (default: True)')
    parser.add_argument('--no-debug', dest='debug', action='store_false',
                        help='Disable debug mode')
    
    # Weights & Biases (wandb) Configuration
    parser.add_argument('--use-wandb', action='store_true', default=True,
                        help='Use Weights & Biases for logging (default: True)')
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='pure_jax_experiments',
                        help='Weights & Biases project name (default: pure_jax_experiments)')
    parser.add_argument('--wandb-entity', type=str, default='tojo',
                        help='Weights & Biases entity/username (default: tojo)')
    # Run name override for logging systems (wandb)
    parser.add_argument('--run-name', type=str, default=None,
                        help='Optional run name override for wandb runs (default: auto)')
    
    # Dual value heads for intrinsic/extrinsic separation
    parser.add_argument('--use-dual-value-heads', action='store_true', default=True,
                        help='Use separate value heads for intrinsic and extrinsic rewards (default: True). '
                             'This prevents value function confusion and helps unlock hidden achievements.')
    parser.add_argument('--no-dual-value-heads', dest='use_dual_value_heads', action='store_false',
                        help='Disable dual value heads (use single combined value head)')
    
    # EMI (Exploration with Mutual Information) Parameters
    parser.add_argument('--train-emi', action='store_true', default=False,
                        help='Train with EMI (Exploration with Mutual Information) intrinsic rewards')
    parser.add_argument('--emi-embedding-dim', type=int, default=32,
                        help='EMI embedding dimension for state and action embeddings (default: 32)')
    parser.add_argument('--emi-hidden-dim', type=int, default=128,
                        help='EMI hidden layer dimension (default: 128)')
    parser.add_argument('--emi-use-reconciler', action='store_true', default=True,
                        help='Use reconciler network for residual correction in EMI (default: True)')
    parser.add_argument('--emi-no-reconciler', dest='emi_use_reconciler', action='store_false',
                        help='Disable reconciler network in EMI')
    parser.add_argument('--emi-learning-rate', type=float, default=3e-4,
                        help='EMI optimizer learning rate (default: 3e-4)')
    parser.add_argument('--emi-pool-size', type=int, default=10000,
                        help='EMI embedding pool size for diversity reward (default: 10000)')
    parser.add_argument('--emi-diversity-coeff', type=float, default=0.1,
                        help='EMI diversity-seeking reward coefficient (default: 0.1)')
    parser.add_argument('--emi-residual-coeff', type=float, default=0.1,
                        help='EMI residual error reward coefficient (default: 0.1)')
    parser.add_argument('--emi-diversity-bandwidth', type=float, default=1.0,
                        help='EMI RBF kernel bandwidth for diversity reward (default: 1.0)')
    parser.add_argument('--emi-mi-action-weight', type=float, default=0.05,
                        help='EMI mutual information weight for action embedding (default: 0.05)')
    parser.add_argument('--emi-mi-obs-weight', type=float, default=0.05,
                        help='EMI mutual information weight for observation embedding (default: 0.05)')
    parser.add_argument('--emi-dynamics-weight', type=float, default=1.0,
                        help='EMI dynamics prediction loss weight (default: 1.0)')
    
    # Intrinsic reward warmup
    parser.add_argument('--intrinsic-warmup-ratio', type=float, default=0.1,
                        help='Ratio of training to use for intrinsic reward warmup (0-1). '
                             'During warmup, intrinsic rewards are scaled linearly from 0 to 1. (default: 0.1)')
    
    # Checkpointing and Resume Training
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Path to checkpoint directory for resuming training (default: auto-detect from wandb run)')
    parser.add_argument('--checkpoint-freq', type=int, default=1000,
                        help='Save checkpoint every N update steps (default: 1000). Set to 0 to disable periodic checkpoints. '
                             'A final checkpoint is always saved at end of training.')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save checkpoints (default: wandb run dir or output-dir/checkpoints)')
    
    return parser.parse_args()


def apply_reward_type_mapping(args):
    """Map the `--reward-type` string to CLI booleans/weights.

    This returns a mutated `args` namespace with the mapped values set.
    """
    if args.reward_type == 'rnd':
        args.train_rnd = True
        # disable ICM/VIME/JAX-native intrinsics
        args.train_icm = False
        args.train_vime = False
        args.novelty_weight = 0.0
        args.surprise_weight = 0.0
        args.empowerment_weight = 0.0
    elif args.reward_type == 'icm':
        args.train_icm = True
        args.train_rnd = False
        args.train_vime = False
        # disable JAX-native intrinsic weights to avoid confusion
        args.novelty_weight = 0.0
        args.surprise_weight = 0.0
        args.empowerment_weight = 0.0
    elif args.reward_type == 'vime':
        args.train_vime = True
        args.train_icm = False
        args.train_rnd = False
        # disable JAX-native intrinsic weights to avoid confusion
        args.novelty_weight = 0.0
        args.surprise_weight = 0.0
        args.empowerment_weight = 0.0
    elif args.reward_type == 'novelty':
        args.train_icm = False
        args.train_rnd = False
        args.train_vime = False
        args.surprise_weight = 0.0
        args.empowerment_weight = 0.0
    elif args.reward_type == 'surprise':
        args.train_icm = False
        args.train_rnd = False
        args.train_vime = False
        args.novelty_weight = 0.0
        args.empowerment_weight = 0.0
    elif args.reward_type == 'empowerment':
        args.train_icm = False
        args.train_rnd = False
        args.train_vime = False
        args.novelty_weight = 0.0
        args.surprise_weight = 0.0
    elif args.reward_type == 'emi':
        # EMI (Exploration with Mutual Information) - completely independent intrinsic reward system
        # EMI uses its own diversity_coeff and residual_coeff, NOT the novelty/surprise weights
        args.train_emi = True
        args.train_icm = False
        args.train_rnd = False
        args.train_vime = False
        # Disable all other intrinsic reward systems - EMI is self-contained
        args.novelty_weight = 0.0
        args.surprise_weight = 0.0
        args.empowerment_weight = 0.0
    elif args.reward_type == 'combined':
        # Combined uses all three intrinsic rewards (novelty, surprise, empowerment)
        # User-provided weights are used directly (no override)
        args.train_icm = False
        args.train_rnd = False
        args.train_vime = False
        args.train_emi = False
        # Don't override weights - use whatever the user provided via CLI
        # If no weights specified, defaults are: novelty=1.0, surprise=1.0, empowerment=0.5
    return args

def main():
    args = parse_args()
    
    # Set terminate_on_reward default based on environment type
    # MiniGrid: True (terminate when reaching goal rewards)
    # Craftax: False (continue exploring after achievements)
    # Only apply default if user hasn't explicitly set --terminate-on-reward flag
    import sys
    if '--terminate-on-reward' not in sys.argv:
        # User did not explicitly set the flag, so apply environment-based default
        is_minigrid = 'MiniGrid' in args.env
        is_craftax = 'Craftax' in args.env
        
        if is_minigrid:
            args.terminate_on_reward = True
            logging.info("Auto-setting terminate_on_reward=True for MiniGrid environment")
        elif is_craftax:
            args.terminate_on_reward = False
            logging.info("Auto-setting terminate_on_reward=False for Craftax environment")
    
    # Configure XLA for deterministic compilation BEFORE any JAX imports
    # This ensures reproducible results with the same seed
    # --xla_gpu_deterministic_ops: Forces deterministic operations on GPU
    # --xla_gpu_autotune_level=0: Disables autotuning which can vary between runs
    # --xla_gpu_force_compilation_parallelism=1: Single-threaded compilation for consistency
    os.environ.setdefault('XLA_FLAGS', 
        '--xla_gpu_deterministic_ops=true '
        '--xla_gpu_autotune_level=0 '
        '--xla_gpu_force_compilation_parallelism=1')
    
    # Configure JAX device BEFORE importing jax
    if args.device == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'
    elif args.device == 'cuda':
        # Include both CUDA and CPU for debug callbacks
        # CPU is needed for jax.debug.callback to work
        os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    elif args.device == 'auto':
        # Set CPU as fallback if GPU fails
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    
    # Import train_agent AFTER setting JAX environment
    from algos.ppo_jax.train_agent import train_agent
    
    # Configure logging: the --debug flag overrides the log level and sets DEBUG
    effective_level = logging.DEBUG if args.debug else getattr(logging, args.log_level)
    logging.basicConfig(level=effective_level)
    set_global_seeds(args.seed)
    output_dir = args.output_dir
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Log intrinsic reward system info
    logging.info("Using SCIRE (VODM+VFDM+VEDM) intrinsic reward system")
    logging.info(f"  - Hidden dim: {args.hidden_dim}, Latent dim: {args.latent_dim}")
    logging.info(f"  - Novelty weight: {args.novelty_weight}, Surprise weight: {args.surprise_weight}, Empowerment weight: {args.empowerment_weight}")

    # Apply mapping for reward types
    args = apply_reward_type_mapping(args)
    
    # For count-based novelty, always match count_based_bonus_coef to novelty_weight
    # This ensures consistent scaling between VAE and count-based approaches
    if args.use_count_based_novelty:
        if args.count_based_bonus_coef is None or args.count_based_bonus_coef != args.novelty_weight:
            args.count_based_bonus_coef = args.novelty_weight
            logging.info(f"Setting count_based_bonus_coef={args.count_based_bonus_coef} to match novelty_weight for fair comparison")
        
        # Count-based ONLY replaces novelty! Surprise and empowerment models won't be trained.
        # Automatically disable surprise/empowerment if they're non-zero
        if args.surprise_weight != 0.0 or args.empowerment_weight != 0.0:
            logging.warning("⚠️  Count-based novelty ONLY replaces novelty component!")
            logging.warning("⚠️  Surprise and empowerment models will NOT be trained with count-based mode.")
            logging.warning(f"⚠️  Disabling: surprise_weight ({args.surprise_weight} → 0.0), empowerment_weight ({args.empowerment_weight} → 0.0)")
            args.surprise_weight = 0.0
            args.empowerment_weight = 0.0
            logging.info("✓ Count-based mode: Using ONLY novelty exploration")

        # Count-based bonus replaces scire's own VODM novelty computation, so
        # novelty_weight must be zeroed to avoid also running scire's novelty path.
        args.novelty_weight = 0.0

    # Build and call train_agent with args
    # Log the effective training choice for clarity in logs
    if args.train_rnd:
        logging.info("Training path: RND (ppo)")
    elif args.train_icm:
        logging.info("Training path: ICM (ppo)")
    elif args.train_vime:
        logging.info("Training path: VIME (ppo)")
    else:
        # JAX-native combined or single-component weights
        active = []
        if args.novelty_weight != 0:
            active.append('Novelty')
        if args.surprise_weight != 0:
            active.append('Surprise')
        if args.empowerment_weight != 0:
            active.append('Empowerment')
        if not active:
            logging.info("Training path: Vanilla PPO")
        elif len(active) == 1:
            logging.info(f"Training path: {active[0]}")
        elif len(active) == 3:
            logging.info("Training path: N+S+E (combined JAX-native)")
        else:
            logging.info(f"Training path: {'+'.join(active)} (combined JAX-native)")
    # Report selected training path and key flags
    selected = None
    if args.train_rnd:
        selected = 'RND'
    elif args.train_icm:
        selected = 'ICM'
    elif args.train_vime:
        selected = 'VIME'
    else:
        active = []
        if args.novelty_weight != 0:
            active.append('Novelty')
        if args.surprise_weight != 0:
            active.append('Surprise')
        if args.empowerment_weight != 0:
            active.append('Empowerment')
        selected = 'Vanilla PPO' if not active else '+'.join(active)

    logging.info("Selected training path: %s (train_icm=%s, train_rnd=%s, train_vime=%s, train_emi=%s, novelty=%s, surprise=%s, empowerment=%s)", selected, args.train_icm, args.train_rnd, args.train_vime, getattr(args, 'train_emi', False), args.novelty_weight, args.surprise_weight, args.empowerment_weight)

    train_agent(
        env_name=args.env,
        reward_type=args.reward_type,
        seed=args.seed,
        log_level=args.log_level,
        total_timesteps=args.timesteps,
        save_model_path=args.save_model,
        device=args.device,
        int_rew_coef=args.int_rew_coef,
        novelty_weight=args.novelty_weight,
        surprise_weight=args.surprise_weight,
        empowerment_weight=args.empowerment_weight,
        output_dir=output_dir,
        save_policy=args.save_policy,
        save_best_policy=args.save_best_policy,
        terminate_on_reward=args.terminate_on_reward,
        fixed_landscape=args.fixed_landscape,
        record_episodes=args.record_episodes,
        # SCIRE (VODM+VFDM+VEDM) system parameters
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        planning_horizon=args.planning_horizon,
        num_action_sequences=args.num_action_sequences,
        vae_learning_rate=args.vae_learning_rate,
        # Count-Based Exploration Parameters
        use_count_based_novelty=args.use_count_based_novelty,
        count_based_bonus_coef=args.count_based_bonus_coef,
        # PPO Hyperparameters
        NUM_REPEATS=args.num_repeats,
        NUM_STEPS=args.num_steps,
        NUM_ENVS=args.num_envs,
        NUM_MINIBATCHES=args.num_minibatches,
        USE_OPTIMISTIC_RESETS=args.use_optimistic_resets,
        OPTIMISTIC_RESET_RATIO=args.optimistic_reset_ratio,
        LAYER_SIZE=args.layer_size,
        ANNEAL_LR=args.anneal_lr,
        MAX_GRAD_NORM=args.max_grad_norm,
        TRAIN_ICM=args.train_icm,
        GAMMA=args.gamma,
        GAE_LAMBDA=args.gae_lambda,
        UPDATE_EPOCHS=args.update_epochs,
        CLIP_EPS=args.clip_eps,
        VF_COEF=args.vf_coef,
        ENT_COEF=args.ent_coef,
        LR=args.lr,
        DEBUG=args.debug,
        USE_WANDB=args.use_wandb,
        WANDB_PROJECT=args.wandb_project,
        WANDB_ENTITY=args.wandb_entity, 
        TRAIN_RND=args.train_rnd,
        RND_REWARD_COEFF=args.rnd_reward_coeff,
        RND_LAYER_SIZE=args.rnd_layer_size,
        RND_OUTPUT_SIZE=args.rnd_output_size,
        RND_LR=args.rnd_lr,
        RND_LOSS_COEFF=args.rnd_loss_coeff,
        TRAIN_VIME=args.train_vime,
        VIME_HIDDEN_DIM=args.vime_hidden_dim,
        VIME_PRIOR_STD=args.vime_prior_std,
        VIME_STEP_SIZE=args.vime_step_size,
        VIME_OUT_SIGMA=args.vime_out_sigma,
        VIME_LR=args.vime_lr,
        VIME_REWARD_COEFF=args.vime_reward_coeff,
        VIME_KL_BUFFER_SIZE=args.vime_kl_buffer_size,
        ICM_LATENT_SIZE=args.icm_latent_size,
        ICM_LAYER_SIZE=args.icm_layer_size,
        ICM_LR=args.icm_lr,
        ICM_FORWARD_LOSS_COEF=args.icm_forward_loss_coef,
        ICM_INVERSE_LOSS_COEF=args.icm_inverse_loss_coef,
        ICM_REWARD_COEFF=args.icm_reward_coeff,
        RUN_NAME=args.run_name,
        # EMI (Exploration with Mutual Information) parameters
        TRAIN_EMI=getattr(args, 'train_emi', False),
        EMI_EMBEDDING_DIM=args.emi_embedding_dim,
        EMI_HIDDEN_DIM=args.emi_hidden_dim,
        EMI_USE_RECONCILER=args.emi_use_reconciler,
        EMI_LEARNING_RATE=args.emi_learning_rate,
        EMI_POOL_SIZE=args.emi_pool_size,
        EMI_DIVERSITY_COEFF=args.emi_diversity_coeff,
        EMI_RESIDUAL_COEFF=args.emi_residual_coeff,
        EMI_DIVERSITY_BANDWIDTH=args.emi_diversity_bandwidth,
        EMI_MI_ACTION_WEIGHT=args.emi_mi_action_weight,
        EMI_MI_OBS_WEIGHT=args.emi_mi_obs_weight,
        EMI_DYNAMICS_WEIGHT=args.emi_dynamics_weight,
        INTRINSIC_WARMUP_RATIO=args.intrinsic_warmup_ratio,
        # Dual value heads
        USE_DUAL_VALUE_HEADS=args.use_dual_value_heads,
        # Checkpointing and Resume Training
        RESUME=args.resume,
        CHECKPOINT_PATH=args.checkpoint_path,
        CHECKPOINT_FREQ=args.checkpoint_freq,
        CHECKPOINT_DIR=args.checkpoint_dir,
        # Activation function (passed through to PPO)
        ACTIVATION=args.activation,
    )

if __name__ == "__main__":
    main()
