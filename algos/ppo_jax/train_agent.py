import logging
import numpy as np
import jax
import jax.numpy as jnp
from algos.ppo_jax.ppo import run_ppo
from envs.env_factory import make_env_from_name
from envs.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)


def train_agent(env_name='Craftax-Classic-Symbolic-v1',
                reward_type='novelty', seed=None, log_level='INFO', total_timesteps=100000,
                save_model_path=None, device='auto',
                # Increased intrinsic reward weights from 1e-3 to 1e-2 for stronger exploration
                int_rew_coef=1.0, novelty_weight=1e-2, surprise_weight=1e-2, empowerment_weight=1e-2,
                n_eval_episodes=10, output_dir=None,
                terminate_on_reward=False,
                fixed_landscape=False,
                record_episodes=False,
                # SCIRE (VODM+VFDM+VEDM) intrinsic reward system parameters
                hidden_dim=256,
                latent_dim=64,
                planning_horizon=3,
                num_action_sequences=10,
                # Lower learning rate (3e-4) for deep networks is more stable than 1e-3
                # Based on "1000 Layer Networks for Self-Supervised RL" (NeurIPS 2025)
                vae_learning_rate=3e-4,
                # Count-Based Exploration Parameters
                use_count_based_novelty=False,
                count_based_bonus_coef=1.0,
                count_table_size=100000,  # Hash table size for count-based exploration
                NUM_REPEATS=1,
                NUM_STEPS=64,
                # NUM_ENVS=128 provides good balance for 1M steps:
                # - More gradient updates (122 vs 15 with 1024 envs)
                # - Better world model adaptation
                # - Sufficient batch diversity
                NUM_ENVS=128,
                NUM_MINIBATCHES=8,
                USE_OPTIMISTIC_RESETS=True,
                OPTIMISTIC_RESET_RATIO=16,
                LAYER_SIZE=512,
                ANNEAL_LR=True,
                MAX_GRAD_NORM=1.0,
                TRAIN_ICM=False,
                GAMMA=0.99,
                GAE_LAMBDA=0.8,
                UPDATE_EPOCHS=4,
                CLIP_EPS=0.2,
                VF_COEF=0.5,
                ENT_COEF=0.01,
                LR=2e-4,
                # Network activation (default: tanh)
                ACTIVATION='tanh',
                DEBUG=True,
                save_policy=False,
                save_best_policy=False,
                USE_WANDB=True,
                WANDB_PROJECT='jax-crafter-2sg-test',
                WANDB_ENTITY='tojo',
                TRAIN_RND=False,
                RND_REWARD_COEFF=1.0,
                RND_LAYER_SIZE=256,
                RND_OUTPUT_SIZE=512,
                RND_LR=3e-4,
                RND_LOSS_COEFF=0.01,
                TRAIN_VIME=False,
                VIME_HIDDEN_DIM=64,
                VIME_PRIOR_STD=0.5,
                VIME_STEP_SIZE=0.01,
                VIME_OUT_SIGMA=1.0,
                VIME_LR=1e-3,
                VIME_REWARD_COEFF=1.0,
                VIME_KL_BUFFER_SIZE=10000,
                RUN_NAME=None,
                # ICM parameters used by ppo
                ICM_LATENT_SIZE=32,
                ICM_LAYER_SIZE=256,
                ICM_LR=3e-4,
                ICM_FORWARD_LOSS_COEF=1.0,
                ICM_INVERSE_LOSS_COEF=1.0,
                ICM_REWARD_COEFF=1.0,
                EXPLORATION_UPDATE_EPOCHS=16,
                # Checkpointing and Resume Training
                RESUME=False,
                CHECKPOINT_PATH=None,
                CHECKPOINT_FREQ=100,
                CHECKPOINT_DIR=None,
                # EMI (Exploration with Mutual Information) Parameters
                TRAIN_EMI=False,
                EMI_EMBEDDING_DIM=32,
                EMI_HIDDEN_DIM=128,
                EMI_USE_RECONCILER=True,
                EMI_LEARNING_RATE=3e-4,
                EMI_POOL_SIZE=10000,
                EMI_DIVERSITY_COEFF=0.1,
                EMI_RESIDUAL_COEFF=0.1,
                EMI_DIVERSITY_BANDWIDTH=1.0,
                EMI_MI_ACTION_WEIGHT=0.05,
                EMI_MI_OBS_WEIGHT=0.05,
                EMI_DYNAMICS_WEIGHT=1.0,
                INTRINSIC_WARMUP_RATIO=0.1,
                USE_DUAL_VALUE_HEADS=True):
    """
    Train an agent using custom PPO with the specified intrinsic reward.
    """
    # Handle device setting (must be done before any JAX operations)
    import os
    if device == 'cuda':
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    elif device == 'cpu':
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'

    if seed is not None:
        # Global seeds should already be set in imrl.py, but ensure JAX key is created
        key = jax.random.PRNGKey(seed)
        logging.info(f"JAX random key created with seed: {seed}")
    else:
        key = jax.random.PRNGKey(0)
        logging.info("JAX random key created with default seed: 0")

    # Create environment using universal env factory
    result = make_env_from_name(env_name)
    # If MiniGrid, prefer the pure-JAX vectorized implementation as default
    if "MiniGrid" in env_name:
        try:
            from envs.minigrid.jax_vector_env import make_vectorized_minigrid, VectorizedGymnaxAdapter
            # Create vectorized env with requested number of environments
            env = make_vectorized_minigrid(
                env_id=env_name, n_envs=NUM_ENVS, seed=(seed if seed is not None else 0))
            env_params = getattr(env, 'default_params', None)
            logging.info(
                "Using Vectorized JAX MiniGrid as default environment for training")
            vectorized_env_used = True
        except Exception as e:
            logging.warning(
                f"Could not use Vectorized JAX env (falling back): {e}")
            vectorized_env_used = False
            if isinstance(result, tuple):
                env, env_params = result
            else:
                env = result
                env_params = env.default_params
        # Apply MiniGrid-specific HPO defaults for PPO hyperparameters
        logging.info("Applying MiniGrid HPO defaults for PPO hyperparameters")
        # These values come from the recent HPO run (best trial parameters)
        NUM_STEPS = 64
        NUM_MINIBATCHES = 8
        GAMMA = 0.995
        LR = 0.0003199391874121294
        UPDATE_EPOCHS = 4
        GAE_LAMBDA = 0.95
        MAX_GRAD_NORM = 1.0
        ENT_COEF = 0.0008905265887071218
        CLIP_EPS = 0.2
        VF_COEF = 0.11902792352294289
        LAYER_SIZE = 64
        ANNEAL_LR = True
        ACTIVATION = 'tanh'
    else:
        vectorized_env_used = False
        if isinstance(result, tuple):
            env, env_params = result
        else:
            env = result
            env_params = env.default_params

    # Propagate terminate_on_reward flag into environment params if available
    try:
        # env_params may be a dataclass with attribute terminate_on_reward
        env_params = env_params.replace(
            terminate_on_reward=terminate_on_reward)
    except Exception:
        try:
            setattr(env_params, 'terminate_on_reward', terminate_on_reward)
        except Exception:
            # If env_params doesn't support attribute setting, ignore
            pass

    # If the environment is the vectorized adapter we created above, skip the extra batching wrappers
    env = LogWrapper(env)
    if not vectorized_env_used:
        if USE_OPTIMISTIC_RESETS:
            env = OptimisticResetVecEnvWrapper(
                env,
                num_envs=NUM_ENVS,
                reset_ratio=min(OPTIMISTIC_RESET_RATIO, NUM_ENVS),
            )
        else:
            env = AutoResetEnvWrapper(env)
            env = BatchEnvWrapper(env, num_envs=NUM_ENVS)

    # Handle fixed landscape: pre-generate a single world map and attach it to env_params
    if fixed_landscape:
        try:
            # Only generate a Craftax world if the selected environment is Craftax
            if 'Craftax' in env_name:
                # Generate a world once via the world generator if available
                from envs.craftax.craftax_classic.world_gen import generate_world
                # create a temporary RNG and generate a world state
                rng = jax.random.PRNGKey(seed if seed is not None else 0)
                single_state = generate_world(
                    rng, env_params, env.static_env_params)
            else:
                single_state = None
        except Exception:
            single_state = None

        # If we managed to create an EnvState, extract the map and attach it
        if single_state is not None:
            try:
                fixed_map = single_state.map
                env_params = env_params.replace(
                    fixed_landscape=True, fixed_landscape_map=fixed_map)
                # Also attach the full EnvState prototype to the env instance so
                # resets can reuse it without branching on tracer-valued params.
                try:
                    setattr(env, '_fixed_landscape_state', single_state)
                    setattr(env, '_use_fixed_landscape', True)
                except Exception:
                    pass
            except Exception:
                try:
                    setattr(env_params, 'fixed_landscape', True)
                    setattr(env_params, 'fixed_landscape_map', fixed_map)
                    try:
                        setattr(env, '_fixed_landscape_state', single_state)
                        setattr(env, '_use_fixed_landscape', True)
                    except Exception:
                        pass
                except Exception:
                    pass

    # Always resolve observation_space and action_space to the actual space object

    # Robustly resolve observation_space and action_space, handling methods that require arguments

    def resolve_space_with_params(space, name, env):
        import inspect
        if callable(space):
            sig = inspect.signature(space)
            if len(sig.parameters) == 0:
                return space()
            # Try with env.default_params if available
            if hasattr(env, 'default_params'):
                try:
                    return space(env.default_params)
                except Exception:
                    pass
            # If it's a bound method, try to get the property from __self__
            if hasattr(space, '__self__') and hasattr(space.__self__, name):
                prop = getattr(space.__self__, name)
                if not callable(prop):
                    return prop
                prop_sig = inspect.signature(prop)
                if len(prop_sig.parameters) == 0:
                    return prop()
            # As a last resort, try calling the method with no arguments
            try:
                return space()
            except Exception:
                pass
            raise ValueError(
                f"{name} appears to be a method requiring arguments. Could not resolve to a space object. Got: {space}")
        return space

    obs_space = resolve_space_with_params(
        env.observation_space, 'observation_space', env)
    act_space = resolve_space_with_params(
        env.action_space, 'action_space', env)

    # Check if we should use intrinsic rewards or run vanilla PPO
    # EMI is an independent intrinsic reward system and doesn't use novelty/surprise/empowerment weights
    use_intrinsic_rewards = (
        novelty_weight != 0 or surprise_weight != 0 or empowerment_weight != 0 or TRAIN_EMI)

    if not use_intrinsic_rewards:
        # When all intrinsic reward weights are 0, run vanilla PPO directly
        # This ensures identical behavior to calling ppo.py directly
        import types

        # Create a config object that matches ppo.py's expectations
        ppo_config = types.SimpleNamespace(
            ENV_NAME=env_name,
            TOTAL_TIMESTEPS=total_timesteps,
            SEED=seed if seed is not None else np.random.randint(2**31),
            NUM_REPEATS=NUM_REPEATS,
            NUM_STEPS=NUM_STEPS,
            NUM_ENVS=NUM_ENVS,
            NUM_MINIBATCHES=NUM_MINIBATCHES,
            USE_OPTIMISTIC_RESETS=USE_OPTIMISTIC_RESETS,
            OPTIMISTIC_RESET_RATIO=OPTIMISTIC_RESET_RATIO,
            LAYER_SIZE=LAYER_SIZE,
            ACTIVATION=ACTIVATION,
            ANNEAL_LR=ANNEAL_LR,
            MAX_GRAD_NORM=MAX_GRAD_NORM,
            TRAIN_ICM=TRAIN_ICM,
            GAMMA=GAMMA,
            GAE_LAMBDA=GAE_LAMBDA,
            UPDATE_EPOCHS=UPDATE_EPOCHS,
            CLIP_EPS=CLIP_EPS,
            VF_COEF=VF_COEF,
            ENT_COEF=ENT_COEF,
            LR=LR,
            DEBUG=DEBUG,
            SAVE_POLICY=save_policy,
            SAVE_BEST_POLICY=save_best_policy,
            USE_WANDB=USE_WANDB,
            WANDB_PROJECT=WANDB_PROJECT,
            WANDB_ENTITY=WANDB_ENTITY,
            RUN_NAME=RUN_NAME,
            # Add intrinsic reward weights for directory generation
            NOVELTY_WEIGHT=novelty_weight,
            SURPRISE_WEIGHT=surprise_weight,
            EMPOWERMENT_WEIGHT=empowerment_weight,
            INTRINSIC_WARMUP_RATIO=INTRINSIC_WARMUP_RATIO,
            # Dual value heads
            USE_DUAL_VALUE_HEADS=USE_DUAL_VALUE_HEADS,
            # Vanilla PPO doesn't use JAX intrinsic rewards
            USE_JAX_INTRINSIC_REWARDS=False,
            # Environment configuration
            FIXED_LANDSCAPE=fixed_landscape,
            TERMINATE_ON_REWARD=terminate_on_reward,
            # Video recording
            RECORD_EPISODES=record_episodes,
            # RND options
            TRAIN_RND=TRAIN_RND,
            RND_REWARD_COEFF=RND_REWARD_COEFF,
            RND_LAYER_SIZE=RND_LAYER_SIZE,
            RND_OUTPUT_SIZE=RND_OUTPUT_SIZE,
            RND_LR=RND_LR,
            RND_LOSS_COEFF=RND_LOSS_COEFF,
            # VIME options
            TRAIN_VIME=TRAIN_VIME,
            VIME_HIDDEN_DIM=VIME_HIDDEN_DIM,
            VIME_PRIOR_STD=VIME_PRIOR_STD,
            VIME_STEP_SIZE=VIME_STEP_SIZE,
            VIME_OUT_SIGMA=VIME_OUT_SIGMA,
            VIME_LR=VIME_LR,
            VIME_REWARD_COEFF=VIME_REWARD_COEFF,
            VIME_KL_BUFFER_SIZE=VIME_KL_BUFFER_SIZE,
            EXPLORATION_UPDATE_EPOCHS=EXPLORATION_UPDATE_EPOCHS,
            # ICM defaults expected by ppo.py
            ICM_LATENT_SIZE=ICM_LATENT_SIZE,
            ICM_LAYER_SIZE=ICM_LAYER_SIZE,
            ICM_LR=ICM_LR,
            ICM_FORWARD_LOSS_COEF=ICM_FORWARD_LOSS_COEF,
            ICM_INVERSE_LOSS_COEF=ICM_INVERSE_LOSS_COEF,
            ICM_REWARD_COEFF=ICM_REWARD_COEFF,
            # Count-based exploration (independent of scire; can run standalone here)
            USE_COUNT_BASED_NOVELTY=use_count_based_novelty,
            COUNT_TABLE_SIZE=count_table_size,
            COUNT_BASED_BONUS_COEF=count_based_bonus_coef,
            # Checkpointing and Resume Training
            RESUME=RESUME,
            CHECKPOINT_PATH=CHECKPOINT_PATH,
            CHECKPOINT_FREQ=CHECKPOINT_FREQ,
            CHECKPOINT_DIR=CHECKPOINT_DIR,
            NUM_EVAL_EPISODES=n_eval_episodes,
        )
        if output_dir is not None:
            ppo_config.RECORD_DIR = output_dir

        # RND (like ICM) is fully integrated into ppo.py's generic training loop
        result = run_ppo(ppo_config)
        logging.info("Training complete (vanilla PPO).")
        return result

    # If we reach here, we need intrinsic rewards - use JAX-native implementation
    logging.info("Using JAX-native intrinsic rewards")

    # Handle EMI training separately
    if TRAIN_EMI:
        logging.info(
            "Using EMI (Exploration with Mutual Information) intrinsic rewards")
        logging.info(
            f"  - Embedding dim: {EMI_EMBEDDING_DIM}, Hidden dim: {EMI_HIDDEN_DIM}")
        logging.info(
            f"  - Diversity coeff: {EMI_DIVERSITY_COEFF}, Residual coeff: {EMI_RESIDUAL_COEFF}")
        logging.info(
            f"  - MI weights: action={EMI_MI_ACTION_WEIGHT}, obs={EMI_MI_OBS_WEIGHT}")

        # Resolve observation and action spaces
        obs_space = resolve_space_with_params(
            env.observation_space, 'observation_space', env)
        act_space = resolve_space_with_params(
            env.action_space, 'action_space', env)

        # Get dimensions
        if hasattr(obs_space, 'shape'):
            obs_dim = int(np.prod(obs_space.shape))
        elif hasattr(obs_space, 'n'):
            obs_dim = int(obs_space.n)
        else:
            raise ValueError(
                f"Cannot determine obs_dim from observation_space: {obs_space}")

        if hasattr(act_space, 'n'):
            action_dim = int(act_space.n)
        elif hasattr(act_space, 'shape'):
            action_dim = int(np.prod(act_space.shape))
        else:
            action_dim = 1

        # Create EMI intrinsic reward system
        from intrinsic_rewards.emi_jax import create_optimized_emi_system
        import scire

        # create_optimized_emi_system returns 3 values: (params, apply_fn, train_fn)
        # The JIT functions are captured in closures, not stored in emi_params
        # This is critical because JAX functions cannot be traced through jax.lax.scan
        emi_params, emi_raw_apply_fn, emi_raw_train_fn = create_optimized_emi_system(
            key=key,
            obs_dim=obs_dim,
            action_dim=action_dim,
            embedding_dim=EMI_EMBEDDING_DIM,
            hidden_dim=EMI_HIDDEN_DIM,
            use_reconciler=EMI_USE_RECONCILER,
            learning_rate=EMI_LEARNING_RATE,
            pool_size=EMI_POOL_SIZE,
            diversity_coeff=EMI_DIVERSITY_COEFF,
            residual_coeff=EMI_RESIDUAL_COEFF,
            diversity_bandwidth=EMI_DIVERSITY_BANDWIDTH,
            mi_action_weight=EMI_MI_ACTION_WEIGHT,
            mi_obs_weight=EMI_MI_OBS_WEIGHT,
            dynamics_weight=EMI_DYNAMICS_WEIGHT
        )

        # EMI ignores the normalizer's contents (its own reward is unnormalized), but
        # ex_state still expects a scire.NormalizerState-shaped placeholder in this slot.
        normalizer = scire.NormalizerState.init()
        logging.info("Created placeholder normalizer state for EMI intrinsic rewards")

        # Create wrapper to match PPO interface - EMI is independent from novelty/surprise system
        def emi_apply_wrapper(params, obs_batch, action_batch, next_obs_batch,
                              novelty_weight=None, surprise_weight=None, empowerment_weight=None,
                              normalizer_state=None, normalizer=None, update_normalizer=False, key=None,
                              return_components=False, **kwargs):
            """
            Wrapper for EMI implementation to match PPO interface.
            
            EMI is completely independent from the VODM/VFDM novelty/surprise system.
            It uses its own diversity_coeff and residual_coeff directly.
            The novelty_weight/surprise_weight args are ignored - EMI uses EMI_DIVERSITY_COEFF/EMI_RESIDUAL_COEFF.
            
            NOTE: This wrapper does NOT mutate params to avoid JAX tracer leaks.
            Pool updates happen through the EMI training function, not the reward function.
            """
            # EMI uses its own coefficients, NOT the novelty/surprise weights
            # This keeps EMI completely independent from the VODM/VFDM system
            diversity_coeff = scaled_diversity_coeff  # Pre-scaled by int_rew_coef
            residual_coeff = scaled_residual_coeff    # Pre-scaled by int_rew_coef

            # Call EMI apply function with EMI-specific coefficients
            # NOTE: We don't update the pool here to avoid tracer leaks
            # Pool updates happen in emi_train_fn during the training step
            rewards, info = emi_raw_apply_fn(
                params,
                obs_batch,
                action_batch,
                next_obs_batch,
                diversity_coeff=diversity_coeff,
                residual_coeff=residual_coeff,
                return_components=return_components,
            )

            # EMI returns its own component names - keep them as-is for proper logging
            if return_components:
                # Add EMI-specific keys for logging (don't map to novelty/surprise)
                info['emi_diversity_reward'] = info.get(
                    'diversity_reward', jnp.zeros_like(rewards))
                info['emi_residual_reward'] = info.get(
                    'residual_reward', jnp.zeros_like(rewards))
                info['emi_total_reward'] = rewards

            # EMI doesn't use external normalizer, return unchanged
            normalizer_out = normalizer_state if normalizer_state is not None else (
                normalizer if normalizer is not None else None)

            return rewards, info if return_components else normalizer_out

        emi_apply_fn = emi_apply_wrapper

        # Apply int_rew_coef scaling to EMI coefficients
        scaled_diversity_coeff = EMI_DIVERSITY_COEFF * int_rew_coef
        scaled_residual_coeff = EMI_RESIDUAL_COEFF * int_rew_coef

        logging.info(f"Applied int_rew_coef={int_rew_coef} scaling: "
                     f"diversity_coeff={EMI_DIVERSITY_COEFF:.6f} -> {scaled_diversity_coeff:.6f}, "
                     f"residual_coeff={EMI_RESIDUAL_COEFF:.6f} -> {scaled_residual_coeff:.6f}")

        # Prepare config for PPO with EMI intrinsic rewards
        import types
        ppo_config = types.SimpleNamespace(
            ENV_NAME=env_name,
            TOTAL_TIMESTEPS=total_timesteps,
            SEED=seed if seed is not None else np.random.randint(2**31),
            NUM_REPEATS=NUM_REPEATS,
            NUM_STEPS=NUM_STEPS,
            NUM_ENVS=NUM_ENVS,
            NUM_MINIBATCHES=NUM_MINIBATCHES,
            USE_OPTIMISTIC_RESETS=USE_OPTIMISTIC_RESETS,
            OPTIMISTIC_RESET_RATIO=OPTIMISTIC_RESET_RATIO,
            LAYER_SIZE=LAYER_SIZE,
            ANNEAL_LR=ANNEAL_LR,
            MAX_GRAD_NORM=MAX_GRAD_NORM,
            TRAIN_ICM=TRAIN_ICM,
            GAMMA=GAMMA,
            GAE_LAMBDA=GAE_LAMBDA,
            UPDATE_EPOCHS=UPDATE_EPOCHS,
            CLIP_EPS=CLIP_EPS,
            VF_COEF=VF_COEF,
            ENT_COEF=ENT_COEF,
            LR=LR,
            DEBUG=DEBUG,
            SAVE_POLICY=save_policy,
            SAVE_BEST_POLICY=save_best_policy,
            USE_WANDB=USE_WANDB,
            WANDB_PROJECT=WANDB_PROJECT,
            WANDB_ENTITY=WANDB_ENTITY,
            RUN_NAME=RUN_NAME or f"EMI_div{scaled_diversity_coeff:.4f}_res{scaled_residual_coeff:.4f}",
            # EMI uses its own intrinsic reward system, independent from VODM/VFDM
            USE_JAX_INTRINSIC_REWARDS=True,
            USE_EMI=True,
            VAE_PARAMS=emi_params,
            VAE_APPLY_FN=emi_apply_fn,
            # EMI training function (closure with JIT)
            EMI_TRAIN_FN=emi_raw_train_fn,
            INTRINSIC_REWARD_NORMALIZER=normalizer,
            OBS_DIM=obs_dim,
            ACTION_DIM=action_dim,
            # EMI does NOT use novelty/surprise/empowerment weights - set to 0
            # EMI has its own coefficients: EMI_DIVERSITY_COEFF and EMI_RESIDUAL_COEFF
            NOVELTY_WEIGHT=0.0,
            SURPRISE_WEIGHT=0.0,
            EMPOWERMENT_WEIGHT=0.0,
            # EMI-specific parameters (these are what EMI actually uses)
            EMI_DIVERSITY_COEFF=scaled_diversity_coeff,
            EMI_RESIDUAL_COEFF=scaled_residual_coeff,
            EMI_EMBEDDING_DIM=EMI_EMBEDDING_DIM,
            EMI_HIDDEN_DIM=EMI_HIDDEN_DIM,
            EMI_LEARNING_RATE=EMI_LEARNING_RATE,
            EMI_DIVERSITY_BANDWIDTH=EMI_DIVERSITY_BANDWIDTH,
            EMI_MI_ACTION_WEIGHT=EMI_MI_ACTION_WEIGHT,
            EMI_MI_OBS_WEIGHT=EMI_MI_OBS_WEIGHT,
            EMI_DYNAMICS_WEIGHT=EMI_DYNAMICS_WEIGHT,
            INTRINSIC_WARMUP_RATIO=INTRINSIC_WARMUP_RATIO,
            # Dual value heads
            USE_DUAL_VALUE_HEADS=USE_DUAL_VALUE_HEADS,
            # Environment configuration
            FIXED_LANDSCAPE=fixed_landscape,
            TERMINATE_ON_REWARD=terminate_on_reward,
            RECORD_EPISODES=record_episodes,
            # ICM parameters (for compatibility)
            ICM_LATENT_SIZE=ICM_LATENT_SIZE,
            ICM_LAYER_SIZE=ICM_LAYER_SIZE,
            ICM_LR=ICM_LR,
            ICM_FORWARD_LOSS_COEF=ICM_FORWARD_LOSS_COEF,
            ICM_INVERSE_LOSS_COEF=ICM_INVERSE_LOSS_COEF,
            ICM_REWARD_COEFF=ICM_REWARD_COEFF,
            EXPLORATION_UPDATE_EPOCHS=EXPLORATION_UPDATE_EPOCHS,
            # Checkpointing and Resume Training
            RESUME=RESUME,
            CHECKPOINT_PATH=CHECKPOINT_PATH,
            CHECKPOINT_FREQ=CHECKPOINT_FREQ,
            CHECKPOINT_DIR=CHECKPOINT_DIR,
            NUM_EVAL_EPISODES=n_eval_episodes,
        )
        if output_dir is not None:
            ppo_config.RECORD_DIR = output_dir

        result = run_ppo(ppo_config)
        logging.info("Training complete (EMI).")
        return result

    if TRAIN_RND:
        # RND training is mutually exclusive with SCIRE's JAX-native intrinsic rewards
        logging.warning("TRAIN_RND flag set alongside JAX-native intrinsic rewards. RND-specific training is incompatible with SCIRE's VAE-based intrinsic rewards. TRAIN_RND will be ignored and the JAX-native intrinsic reward training will proceed.")

    if TRAIN_VIME:
        # VIME training is mutually exclusive with SCIRE's JAX-native intrinsic rewards
        logging.warning("TRAIN_VIME flag set alongside JAX-native intrinsic rewards. VIME-specific training is incompatible with SCIRE's VAE-based intrinsic rewards. TRAIN_VIME will be ignored and the JAX-native intrinsic reward training will proceed.")

    # Ensure optimistic resets are enabled for intrinsic rewards
    # This is important for efficient exploration with intrinsic rewards
    USE_OPTIMISTIC_RESETS = True
    logging.info(
        f"Enforcing optimistic resets for intrinsic rewards with ratio: {OPTIMISTIC_RESET_RATIO}")

    logging.info("Using SCIRE (VODM+VFDM+VEDM) intrinsic reward system:")
    logging.info(f"  - Hidden dim: {hidden_dim}, Latent dim: {latent_dim}")
    logging.info(
        f"  - Planning horizon: {planning_horizon}, Action sequences: {num_action_sequences}")

    # Resolve observation and action spaces
    obs_space = resolve_space_with_params(
        env.observation_space, 'observation_space', env)
    act_space = resolve_space_with_params(
        env.action_space, 'action_space', env)

    # Get dimensions
    if hasattr(obs_space, 'shape'):
        obs_dim = int(np.prod(obs_space.shape))
    elif hasattr(obs_space, 'n'):
        obs_dim = int(obs_space.n)
    else:
        raise ValueError(
            f"Cannot determine obs_dim from observation_space: {obs_space}")

    if hasattr(act_space, 'n'):
        action_dim = int(act_space.n)
    elif hasattr(act_space, 'shape'):
        action_dim = int(np.prod(act_space.shape))
    else:
        action_dim = 1

    # Create the SCIRE (VODM+VFDM+VEDM) intrinsic reward system.
    # imrl's observations (MiniGrid, Craftax Classic Symbolic) are normalized to [0, 1],
    # so use imrl's sigmoid-bounded variant of scire's model instead of the published
    # package's generalized Gaussian observation model. See intrinsic_rewards/scire_bounded.py.
    import scire
    from intrinsic_rewards.scire_bounded import (
        create_bounded_scire_system,
        LOGVAR_MIN,
        LOGVAR_MAX,
        LATENT_PRED_COEF,
        MAX_EXHAUSTIVE_SEQUENCES,
    )

    key, scire_key = jax.random.split(key)
    scire_model, scire_params, scire_action_sequences = create_bounded_scire_system(
        key=scire_key,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        planning_horizon=planning_horizon,
        num_action_sequences=num_action_sequences,
        beta_vae=1.0,
        logvar_min=LOGVAR_MIN,
        logvar_max=LOGVAR_MAX,
        latent_pred_coef=LATENT_PRED_COEF,
        max_exhaustive_sequences=MAX_EXHAUSTIVE_SEQUENCES,
    )
    logging.info("Created SCIRE (VODM+VFDM+VEDM) intrinsic reward system with sigmoid-bounded [0, 1] observation model")

    # Create normalizer for the three SCIRE reward components
    normalizer = scire.NormalizerState.init()
    logging.info("Created normalizer for SCIRE intrinsic rewards")

    # Apply int_rew_coef scaling to weights (like the factory does)
    scaled_novelty_weight = novelty_weight * int_rew_coef
    scaled_surprise_weight = surprise_weight * int_rew_coef
    scaled_empowerment_weight = empowerment_weight * int_rew_coef

    logging.info(f"Applied int_rew_coef={int_rew_coef} scaling to weights: "
                 f"novelty_weight={novelty_weight:.6f} -> {scaled_novelty_weight:.6f}, "
                 f"surprise_weight={surprise_weight:.6f} -> {scaled_surprise_weight:.6f}, "
                 f"empowerment_weight={empowerment_weight:.6f} -> {scaled_empowerment_weight:.6f}")

    # Prepare config for PPO with SCIRE intrinsic rewards
    import types
    ppo_config = types.SimpleNamespace(
        ENV_NAME=env_name,
        TOTAL_TIMESTEPS=total_timesteps,
        SEED=seed if seed is not None else np.random.randint(2**31),
        NUM_REPEATS=NUM_REPEATS,
        NUM_STEPS=NUM_STEPS,
        NUM_ENVS=NUM_ENVS,
        NUM_MINIBATCHES=NUM_MINIBATCHES,
        USE_OPTIMISTIC_RESETS=USE_OPTIMISTIC_RESETS,
        OPTIMISTIC_RESET_RATIO=OPTIMISTIC_RESET_RATIO,
        LAYER_SIZE=LAYER_SIZE,
        ACTIVATION=ACTIVATION,
        ANNEAL_LR=ANNEAL_LR,
        MAX_GRAD_NORM=MAX_GRAD_NORM,
        TRAIN_ICM=TRAIN_ICM,
        GAMMA=GAMMA,
        GAE_LAMBDA=GAE_LAMBDA,
        UPDATE_EPOCHS=UPDATE_EPOCHS,
        CLIP_EPS=CLIP_EPS,
        VF_COEF=VF_COEF,
        ENT_COEF=ENT_COEF,
        LR=LR,
        DEBUG=DEBUG,
        SAVE_POLICY=save_policy,
        SAVE_BEST_POLICY=save_best_policy,
        USE_WANDB=USE_WANDB,
        WANDB_PROJECT=WANDB_PROJECT,
        WANDB_ENTITY=WANDB_ENTITY,
        RUN_NAME=RUN_NAME,
        # SCIRE intrinsic reward parameters (with int_rew_coef applied)
        USE_JAX_INTRINSIC_REWARDS=True,
        USE_SCIRE=True,
        SCIRE_MODEL=scire_model,
        SCIRE_PARAMS=scire_params,
        SCIRE_ACTION_SEQUENCES=scire_action_sequences,
        INTRINSIC_REWARD_NORMALIZER=normalizer,
        OBS_DIM=obs_dim,
        ACTION_DIM=action_dim,
        NOVELTY_WEIGHT=scaled_novelty_weight,
        SURPRISE_WEIGHT=scaled_surprise_weight,
        EMPOWERMENT_WEIGHT=scaled_empowerment_weight,
        # SCIRE system parameters
        VAE_LEARNING_RATE=vae_learning_rate,
        PLANNING_HORIZON=planning_horizon,
        NUM_ACTION_SEQUENCES=num_action_sequences,
        # Environment configuration
        FIXED_LANDSCAPE=fixed_landscape,
        TERMINATE_ON_REWARD=terminate_on_reward,
        # Video recording
        RECORD_EPISODES=record_episodes,
        # Dual value heads
        USE_DUAL_VALUE_HEADS=USE_DUAL_VALUE_HEADS,
        # RND flags - make sure posterior code paths forward them
        TRAIN_RND=TRAIN_RND,
        RND_REWARD_COEFF=RND_REWARD_COEFF,
        # VIME flags - make sure posterior code paths forward them
        TRAIN_VIME=TRAIN_VIME,
        VIME_REWARD_COEFF=VIME_REWARD_COEFF,
        # Exploration update epochs (for intrinsic model training)
        EXPLORATION_UPDATE_EPOCHS=EXPLORATION_UPDATE_EPOCHS,
        # Count-based exploration
        USE_COUNT_BASED_NOVELTY=use_count_based_novelty,
        COUNT_TABLE_SIZE=count_table_size,
        COUNT_BASED_BONUS_COEF=count_based_bonus_coef,
        # ICM parameters (for ICM training/integration in PPO)
        ICM_LATENT_SIZE=ICM_LATENT_SIZE,
        ICM_LAYER_SIZE=ICM_LAYER_SIZE,
        ICM_LR=ICM_LR,
        ICM_FORWARD_LOSS_COEF=ICM_FORWARD_LOSS_COEF,
        ICM_INVERSE_LOSS_COEF=ICM_INVERSE_LOSS_COEF,
        ICM_REWARD_COEFF=ICM_REWARD_COEFF,
        # Checkpointing and Resume Training
        RESUME=RESUME,
        CHECKPOINT_PATH=CHECKPOINT_PATH,
        CHECKPOINT_FREQ=CHECKPOINT_FREQ,
        CHECKPOINT_DIR=CHECKPOINT_DIR,
        NUM_EVAL_EPISODES=n_eval_episodes,
    )
    if output_dir is not None:
        ppo_config.RECORD_DIR = output_dir

    result = run_ppo(ppo_config)
    logging.info("Training complete.")
    return result
