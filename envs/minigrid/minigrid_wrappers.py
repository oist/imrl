import gymnasium as gym
from gymnasium import spaces
import numpy as np
from minigrid.wrappers import OneHotPartialObsWrapper


class DictToArrayObsWrapper(gym.ObservationWrapper):
    """Convert dictionary observations to array observations."""

    def __init__(self, env):
        super().__init__(env)
        # Get sample observation to determine space
        sample_obs = env.observation_space.sample()

        # Calculate total dimension by handling different types
        flat_dim = 0
        for k, v in sample_obs.items():
            if isinstance(v, (np.ndarray, list)):
                flat_dim += np.array(v).flatten().shape[0]
            elif isinstance(v, (int, float, bool, str, np.integer, np.floating)):
                flat_dim += 1
            else:
                raise ValueError(
                    f"Unsupported observation type for key {k}: {type(v)}")

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(flat_dim,),
            dtype=np.float32
        )

    def observation(self, obs):
        # Convert each observation component to a flat array
        arrays = []
        for k, v in obs.items():
            if isinstance(v, (np.ndarray, list)):
                arrays.append(np.array(v).flatten())
            elif isinstance(v, (int, float, bool, np.integer, np.floating)):
                arrays.append(np.array([float(v)]))
            elif isinstance(v, str):
                # Convert string to one-hot or other numerical representation if needed
                arrays.append(np.array([0.0]))
            else:
                raise ValueError(
                    f"Unsupported observation type for key {k}: {type(v)}")

        return np.concatenate(arrays).astype(np.float32)


class CustomOneHotPartialObsWrapper(OneHotPartialObsWrapper):
    """
    Extends OneHotPartialObsWrapper to include the agent's position in the observation.
    This wrapper concatenates the agent position (x,y) to the one-hot encoding 
    of the partially observable view.
    """
    def __init__(self, env):
        # First initialize with the original environment
        gym.ObservationWrapper.__init__(self, env)
        
        # Initialize the observation space calculation
        obs_shape = env.observation_space["image"].shape
        # Calculate actual flattened size
        one_hot_size = obs_shape[0] * obs_shape[1] * obs_shape[2]
        
        # Set the observation space with correct shape: one_hot_size + 2 for agent position
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(one_hot_size + 2,),  # +2 for agent position only
            dtype=np.float32,
        )

    def observation(self, obs):
        # Process the observation like the parent class but without calling super()
        env = self.env
        
        obs_grid = obs["image"]
        one_hot = []
        for x in range(obs_grid.shape[0]):
            for y in range(obs_grid.shape[1]):
                for i in range(3):  # Loop over the 3 channels
                    one_hot.append(obs_grid[x, y, i])
        one_hot_obs = np.array(one_hot, dtype=np.float32)
        
        # Get agent position and normalize to [0,1] range if needed
        agent_pos = np.array([
            self.env.unwrapped.agent_pos[0],
            self.env.unwrapped.agent_pos[1],
            # self.env.unwrapped.agent_pos[0] / self.env.unwrapped.width,
            # self.env.unwrapped.agent_pos[1] / self.env.unwrapped.height
        ], dtype=np.float32)
        
        # Concatenate one-hot encoding and agent position
        return np.concatenate([one_hot_obs, agent_pos])


class SimpleOneHotPartialObsWrapper(gym.ObservationWrapper):
    """
    One-hot wrapper that encodes each cell as a one-hot vector without including agent position.
    This provides a clean observation containing just what the agent can see.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Initialize the observation space calculation
        obs_shape = env.observation_space["image"].shape
        # Calculate flattened size of one-hot encoding
        one_hot_size = obs_shape[0] * obs_shape[1] * obs_shape[2]
        
        # Set the observation space (without agent position)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(one_hot_size,),
            dtype=np.float32,
        )

    def observation(self, obs):
        # Process the observation to create one-hot encoding
        obs_grid = obs["image"]
        one_hot = []
        for x in range(obs_grid.shape[0]):
            for y in range(obs_grid.shape[1]):
                for i in range(3):  # Loop over the 3 channels
                    one_hot.append(obs_grid[x, y, i])
        
        # Return only the one-hot encoding without agent position
        return np.array(one_hot, dtype=np.float32)
