"""
Episode Monitor for Policy Saving

This module monitors training logs and saves policies when episodes achieve goals.
It works outside JAX compilation to avoid concretization errors.
"""

import os
import json
import jax.numpy as jnp
from typing import Dict, Any, Optional
from dataclasses import dataclass
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer
import numpy as np


@dataclass
class EpisodeAchievement:
    """Record of an episode that achieved a goal."""
    update_step: int
    env_id: int
    goal_type: str
    episode_return: float
    episode_length: int
    achievement_info: Dict[str, Any]
    action_sequence: Optional[list] = None  # NEW: Store the exact action sequence taken


class EpisodeMonitor:
    """
    Monitors training progress and saves policies when goals are achieved.
    
    This works by monitoring the logged metrics instead of trying to integrate
    directly into the JAX-compiled training loop.
    """
    
    def __init__(self, config: Dict, save_dir: str = None):
        self.config = config
        self.save_dir = save_dir or config.get("RECORD_DIR", "episode_policies")
        
        # Create directories
        self.episode_policies_dir = os.path.join(self.save_dir, "episode_policies")
        self.achievements_dir = os.path.join(self.save_dir, "episode_achievements")
        os.makedirs(self.episode_policies_dir, exist_ok=True)
        os.makedirs(self.achievements_dir, exist_ok=True)
        
        # Track achievements
        self.achievements = []
        self.best_achievement = None  # Track only THE BEST episode overall
        self.best_achievements = {
            'small_goal': None,
            'large_goal': None,
            'both_goals': None
        }
        
        print(f"🎯 Episode monitor initialized at: {self.save_dir}")
    
    def check_for_achievements(self, update_step: int, metrics: Dict[str, Any], 
                             train_state: TrainState, traj_batch: Any = None) -> bool:
        """
        Check if any achievements occurred in this update and save policies if so.
        
        Args:
            update_step: Current training update step
            metrics: Training metrics dictionary
            train_state: Current training state
            traj_batch: Trajectory batch containing actions and episode info (optional)
        
        Returns True if any achievements were found and saved.
        """
        achievements_found = False
        
        # Check for MiniGrid-specific achievements
        if "MiniGrid" in self.config.get("ENV_NAME", ""):
            achievements_found = self._check_minigrid_achievements(
                update_step, metrics, train_state, traj_batch
            )
        
        return achievements_found
    
    def _check_minigrid_achievements(self, update_step: int, metrics: Dict[str, Any],
                                   train_state: TrainState, traj_batch: Any = None) -> bool:
        """Check for MiniGrid goal achievements."""
        achievements_found = False
        
        # Look for goal achievement metrics
        achievement_keys = [
            'small_reward_goal_rate',
            'large_reward_goal_rate', 
            'any_goal_achievement_rate'
        ]
        
        for key in achievement_keys:
            if key in metrics and metrics[key] is not None:
                try:
                    # Handle different metric formats
                    val = metrics[key]
                    if hasattr(val, 'item'):
                        achievement_rate = float(val.item())
                    elif hasattr(val, '__len__') and len(val) > 0:
                        achievement_rate = float(jnp.mean(val))
                    else:
                        achievement_rate = float(val)
                    
                    # If achievement rate > 0, we found achievements
                    if achievement_rate > 0.0:
                        episode_return = self._extract_episode_return(metrics)
                        episode_length = self._extract_episode_length(metrics)
                        
                        # Determine goal type
                        if 'small_reward' in key:
                            goal_type = 'small_goal'
                        elif 'large_reward' in key:
                            goal_type = 'large_goal'
                        else:
                            goal_type = 'any_goal'
                        
                        # Extract action sequence if available
                        action_sequence = self._extract_action_sequence(
                            traj_batch, metrics, episode_length
                        ) if traj_batch is not None else None
                        
                        # Create achievement record
                        achievement = EpisodeAchievement(
                            update_step=update_step,
                            env_id=0,  # We don't have per-env info here
                            goal_type=goal_type,
                            episode_return=episode_return,
                            episode_length=episode_length,
                            achievement_info={
                                'achievement_rate': achievement_rate,
                                'metric_key': key
                            },
                            action_sequence=action_sequence
                        )
                        
                        # Check if this is a new best for this goal type
                        if self._is_new_best_achievement(achievement):
                            self._save_achievement_policy(achievement, train_state)
                            achievements_found = True
                        
                        # ALSO check if this is the overall best episode
                        if self._is_overall_best_achievement(achievement):
                            self._save_best_episode_policy(achievement, train_state)
                        
                        self.achievements.append(achievement)
                
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"⚠️  Error processing achievement metric {key}: {e}")
                    continue
        
        return achievements_found
    
    def _extract_episode_return(self, metrics: Dict[str, Any]) -> float:
        """Extract episode return from metrics."""
        for key in ['episode_return', 'episode_returns', 'mean_episode_return', 'returned_episode_returns']:
            if key in metrics and metrics[key] is not None:
                try:
                    val = metrics[key]
                    if hasattr(val, 'item'):
                        return float(val.item())
                    elif hasattr(val, '__len__') and len(val) > 0:
                        return float(jnp.mean(val))
                    else:
                        return float(val)
                except (ValueError, TypeError, AttributeError):
                    continue
        return 0.0
    
    def _extract_episode_length(self, metrics: Dict[str, Any]) -> int:
        """Extract episode length from metrics."""
        for key in ['episode_length', 'episode_lengths', 'mean_episode_length', 'returned_episode_lengths']:
            if key in metrics and metrics[key] is not None:
                try:
                    val = metrics[key]
                    if hasattr(val, 'item'):
                        return int(val.item())
                    elif hasattr(val, '__len__') and len(val) > 0:
                        return int(jnp.mean(val))
                    else:
                        return int(val)
                except (ValueError, TypeError, AttributeError):
                    continue
        return 0
    
    def _extract_action_sequence(self, traj_batch: Any, metrics: Dict[str, Any], 
                                episode_length: int) -> Optional[list]:
        """
        Extract the action sequence from a completed episode.
        
        This finds which environment completed an episode and extracts its full action sequence
        from the info dict where it's accumulated by LogWrapper.
        """
        try:
            # traj_batch.info should contain 'returned_episode' marking which envs completed
            if not hasattr(traj_batch, 'info') or 'returned_episode' not in traj_batch.info:
                return None
            
            returned_episodes = traj_batch.info['returned_episode']  # Shape: (num_steps, num_envs)
            
            # Find which environment(s) completed an episode
            # Sum over time steps to find envs that completed
            completed_envs = jnp.sum(returned_episodes, axis=0) > 0  # Shape: (num_envs,)
            
            # Get the first environment that completed (if any)
            env_indices = jnp.where(completed_envs)[0]
            if len(env_indices) == 0:
                return None
            
            env_id = int(env_indices[0])
            
            # NEW: Extract full action sequence from info dict
            # LogWrapper now accumulates actions and stores them in 'episode_actions'
            if 'episode_actions' in traj_batch.info and 'episode_action_count' in traj_batch.info:
                # Find where the episode completed
                episode_dones = returned_episodes[:, env_id]  # Shape: (num_steps,)
                done_indices = jnp.where(episode_dones)[0]
                
                if len(done_indices) > 0:
                    # Get the last done index (most recent episode completion)
                    done_idx = int(done_indices[-1])
                    
                    # Get the action buffer and count from that timestep
                    action_buffer = traj_batch.info['episode_actions'][done_idx, env_id]  # Full action buffer
                    action_count = int(traj_batch.info['episode_action_count'][done_idx, env_id])  # Valid length
                    
                    # Extract only the valid actions (up to action_count)
                    episode_actions = action_buffer[:action_count]
                    
                    # Convert to list of Python ints
                    action_list = [int(a) for a in episode_actions]
                    
                    actions_captured = len(action_list)
                    # Only log if not MiniGrid (reduce verbosity for MiniGrid)
                    if 'MiniGrid' not in self.config.get('ENV_NAME', ''):
                        if actions_captured == episode_length:
                            print(f"   📝 Extracted FULL action sequence: {actions_captured} actions from env {env_id}")
                        elif actions_captured < episode_length:
                            print(f"   📝 Extracted action sequence: {actions_captured}/{episode_length} actions from env {env_id}")
                        else:
                            print(f"   📝 Extracted action sequence: {actions_captured} actions from env {env_id} (more than expected {episode_length})")
                    
                    return action_list
            
            # FALLBACK: Old method if new fields not available
            # This shouldn't happen with updated LogWrapper, but keep for backwards compatibility
            if hasattr(traj_batch, 'action'):
                actions = traj_batch.action[:, env_id]  # Shape: (num_steps,)
                episode_dones = returned_episodes[:, env_id]  # Shape: (num_steps,)
                done_indices = jnp.where(episode_dones)[0]
                
                if len(done_indices) > 0:
                    done_idx = int(done_indices[-1])
                    start_idx = max(0, done_idx - episode_length + 1)
                    episode_actions = actions[start_idx:done_idx+1]
                    action_list = [int(a) for a in episode_actions]
                    # Only log if not MiniGrid (reduce verbosity for MiniGrid)
                    if 'MiniGrid' not in self.config.get('ENV_NAME', ''):
                        print(f"   📝 Extracted partial sequence (fallback): {len(action_list)}/{episode_length} actions from env {env_id}")
                    return action_list
            
            return None
            
        except Exception as e:
            print(f"⚠️  Failed to extract action sequence: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _is_new_best_achievement(self, achievement: EpisodeAchievement) -> bool:
        """Check if this achievement is better than the current best for its goal type."""
        current_best = self.best_achievements.get(achievement.goal_type)
        
        if current_best is None:
            return True
        
        # Use episode return as the primary metric for "best"
        return achievement.episode_return > current_best.episode_return
    
    def _is_overall_best_achievement(self, achievement: EpisodeAchievement) -> bool:
        """Check if this achievement is the overall best episode across all goal types."""
        if self.best_achievement is None:
            return True
        
        # Use episode return as the primary metric for "best"
        return achievement.episode_return > self.best_achievement.episode_return
    
    def _save_best_episode_policy(self, achievement: EpisodeAchievement, train_state: TrainState):
        """Save the single best episode policy (replaces any previous best)."""
        # Use a fixed directory name for the best episode
        policy_dir = os.path.join(
            self.episode_policies_dir,
            "best_episode"
        )
        
        # Remove old best episode if it exists
        if os.path.exists(policy_dir):
            import shutil
            shutil.rmtree(policy_dir)
        
        os.makedirs(policy_dir, exist_ok=True)
        
        try:
            # Save policy parameters with device-agnostic approach
            checkpointer = PyTreeCheckpointer()
            
            # Convert train_state to device-agnostic format before saving
            import jax.tree_util as tree_util
            import numpy as np
            
            def to_numpy_for_saving(x):
                """Convert JAX arrays to numpy for device-agnostic saving."""
                if hasattr(x, 'shape') and hasattr(x, 'dtype'):
                    return np.array(x)
                return x
            
            # Create device-agnostic copy of train_state
            numpy_train_state = tree_util.tree_map(to_numpy_for_saving, train_state)
            
            # Debug: Verify we're saving the OLD policy (before update)
            try:
                import hashlib
                first_layer = list(train_state.params['params'].values())[0]
                first_params = list(first_layer.values())[0]
                params_signature = hashlib.md5(str(first_params.flatten()[:10]).encode()).hexdigest()[:8]
                print(f"   💾 Saving policy with signature: {params_signature}")
            except:
                pass
            
            # Save the numpy version
            checkpointer.save(
                os.path.join(policy_dir, "policy"),
                numpy_train_state,
                force=True
            )
            
            # Save achievement metadata with seed and env_id for deterministic replay
            metadata = {
                'update_step': achievement.update_step,
                'env_id': achievement.env_id,
                'goal_type': achievement.goal_type,
                'episode_return': achievement.episode_return,
                'episode_length': achievement.episode_length,
                'achievement_info': achievement.achievement_info,
                'original_seed': self.config.get('SEED', 42),  # For deterministic replay
                'original_env_id': achievement.env_id,  # Which parallel env achieved this
                'action_sequence': achievement.action_sequence,  # NEW: Exact actions taken
                'config': self.config
            }
            
            with open(os.path.join(policy_dir, "achievement_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2, default=self._json_serializer)
            
            # Update overall best
            self.best_achievement = achievement
            
            print(f"🏆 BEST episode policy saved at update {achievement.update_step}")
            print(f"   Goal: {achievement.goal_type}")
            print(f"   Return: {achievement.episode_return:.3f}, Length: {achievement.episode_length}")
            print(f"   Saved to: {policy_dir}")
            
        except Exception as e:
            print(f"❌ Failed to save best episode policy: {e}")
    
    def _save_achievement_policy(self, achievement: EpisodeAchievement, train_state: TrainState):
        """Save policy for a significant achievement."""
        # Create unique directory name
        policy_dir = os.path.join(
            self.episode_policies_dir,
            f"achievement_{achievement.update_step}_{achievement.goal_type}"
        )
        os.makedirs(policy_dir, exist_ok=True)
        
        try:
            # Save policy parameters with device-agnostic approach
            checkpointer = PyTreeCheckpointer()
            
            # Convert train_state to device-agnostic format before saving
            import jax.tree_util as tree_util
            import numpy as np
            
            def to_numpy_for_saving(x):
                """Convert JAX arrays to numpy for device-agnostic saving."""
                if hasattr(x, 'shape') and hasattr(x, 'dtype'):
                    return np.array(x)
                return x
            
            # Create device-agnostic copy of train_state
            numpy_train_state = tree_util.tree_map(to_numpy_for_saving, train_state)
            
            # Save the numpy version
            checkpointer.save(
                os.path.join(policy_dir, "policy"),
                numpy_train_state,
                force=True
            )
            
            # Save achievement metadata
            metadata = {
                'update_step': achievement.update_step,
                'env_id': achievement.env_id,
                'goal_type': achievement.goal_type,
                'episode_return': achievement.episode_return,
                'episode_length': achievement.episode_length,
                'achievement_info': achievement.achievement_info,
                'action_sequence': achievement.action_sequence,  # NEW: Exact actions taken
                'config': self.config
            }
            
            with open(os.path.join(policy_dir, "achievement_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2, default=self._json_serializer)
            
            # Update best achievements
            self.best_achievements[achievement.goal_type] = achievement
            
            print(f"✅ Achievement policy saved: {achievement.goal_type} at update {achievement.update_step}")
            print(f"   Return: {achievement.episode_return:.3f}, Length: {achievement.episode_length}")
            print(f"   Saved to: {policy_dir}")
            
        except Exception as e:
            print(f"❌ Failed to save achievement policy: {e}")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy/jax types."""
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return str(obj)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all achievements."""
        return {
            'total_achievements': len(self.achievements),
            'best_achievements': {
                goal_type: {
                    'update_step': achievement.update_step,
                    'episode_return': achievement.episode_return,
                    'episode_length': achievement.episode_length
                } if achievement else None
                for goal_type, achievement in self.best_achievements.items()
            },
            'achievement_history': [
                {
                    'update_step': a.update_step,
                    'goal_type': a.goal_type,
                    'episode_return': a.episode_return,
                    'episode_length': a.episode_length
                }
                for a in self.achievements
            ]
        }
    
    def save_summary(self):
        """Save achievement summary to file."""
        summary = self.get_summary()
        summary_file = os.path.join(self.achievements_dir, "achievements_summary.json")
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Achievement summary saved to: {summary_file}")


def create_episode_monitor(config: Dict) -> Optional[EpisodeMonitor]:
    """Create episode monitor if appropriate for the environment."""
    if "MiniGrid" not in config.get("ENV_NAME", ""):
        return None
    
    if not config.get("USE_WANDB", False):
        return None
    
    save_dir = config.get("RECORD_DIR")
    if save_dir is None:
        # Use wandb directory if available
        import wandb
        if wandb.run is not None:
            save_dir = wandb.run.dir
        else:
            save_dir = "episode_policies"
    
    return EpisodeMonitor(config, save_dir)