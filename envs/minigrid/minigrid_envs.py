import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, make
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, COLORS, STATE_TO_IDX, DIR_TO_VEC
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Key, Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_rect

import numpy as np

from minigrid.core.constants import COLORS, COLOR_TO_IDX

# Ensure the colors defined in minigrid_envs.py are used consistently
COLORS['emeraldgreen'] = np.array([80, 200, 120])
COLOR_TO_IDX['emeraldgreen'] = len(COLOR_TO_IDX)

COLORS['neongreen'] = np.array([15, 255, 80])
COLOR_TO_IDX['neongreen'] = len(COLOR_TO_IDX) + 1
# print(f"len(OBJECT_TO_IDX) = {len(OBJECT_TO_IDX)}")
# print(f"len(COLOR_TO_IDX) = {len(COLOR_TO_IDX)}")
# print(f"len(STATE_TO_IDX) = {len(STATE_TO_IDX)}")
COLOR_NAMES = sorted(list(COLORS.keys()))
# print(COLOR_NAMES)


# Update the Goal and Goal2 classes to use these colors consistently
class Goal(WorldObj):
    def __init__(self):
        super().__init__('goal', 'emeraldgreen')

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Goal2(Goal):
    def __init__(self):
        super().__init__()
        self.color = 'neongreen'


class CustomFourRoomsTwoGoalsFixedViewSize3x3(MiniGridEnv):
    # Define class-level metadata
    metadata = {
        'render_modes': ['rgb_array', 'human'],  # Supported render modes
        'render_fps': 10,  # Add render_fps which is required by MiniGrid
        'track_specific_features': True,
        'has_left_right_gates': True,
        'has_left_right_goals': True,
        'left_gate_positions': [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6)],
        'right_gate_positions': [(7, 6), (8, 6), (9, 6), (10, 6), (11, 6)],
        'left_goal_position': (1, 11),
        'right_goal_position': (11, 11),
    }

    def __init__(self, agent_pos=(1, 1), agent_dir=0, goal_pos=(1, 11), max_steps=1000, render_mode="rgb_array", **kwargs):
        self._agent_default_pos = agent_pos
        self._agent_start_dir = agent_dir
        self._goal_default_pos = goal_pos
        self._goal2_default_pos = (11, 11)

        self.width = 13
        self.height = 13
        self.agent_view_size = 3

        mission_space = MissionSpace(mission_func=lambda: "Reach the goal")

        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=self.agent_view_size,
            render_mode=render_mode,
            **kwargs
        )

    def generate_random_position(self, x_range=(-1, -3), y_range=(-1, -6)):
        x = self._rand_int(*x_range)
        y = self._rand_int(*y_range)
        return (x, y)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w     # Left x-coordinate
                yT = j * room_h     # Top y-coordinate
                xR = xL + room_w    # Right x-coordinate
                yB = yT + room_h    # Bottom y-coordinate

                # Vertical wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)

                    if i == 0 and j == 0:
                        self.grid.set(
                            xR,
                            yB - room_h // 2,
                            Door(COLOR_NAMES[7],
                                 is_open=False, is_locked=False)
                        )

                # Horizontal wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)

                    if i == 0 and j == 0:
                        self.grid.set(xR - room_w // 2, yB, None)
                    if i == 1 and j == 0:
                        self.grid.set(xR - room_w // 2, yB, None)

        # agent start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)

            # assuming fixed start direction
            self.agent_dir = self._agent_start_dir
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        if self._goal2_default_pos is not None:
            goal2 = Goal2()
            self.put_obj(goal2, *self._goal2_default_pos)
            goal2.init_pos, goal2.cur_pos = self._goal2_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.step_count = 0
        obs = self.gen_obs()
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if np.array_equal(self.agent_pos, self._goal2_default_pos):
            reward *= 10

        return obs, reward, terminated, truncated, info

class CustomFourRoomsTwoGoalsRandViewSize3x3(MiniGridEnv):
    # Define class-level metadata
    metadata = {
        'render_modes': ['rgb_array', 'human'],  # Supported render modes
        'render_fps': 10,  # Add render_fps which is required by MiniGrid
        'track_specific_features': True,
        'has_left_right_gates': True,
        'has_left_right_goals': True,
        'left_gate_positions': [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6)],
        'right_gate_positions': [(7, 6), (8, 6), (9, 6), (10, 6), (11, 6)],
        'left_goal_position': (1, 11),
        'right_goal_position': (11, 11),
    }

    def __init__(self, agent_pos=(1, 1), agent_dir=0, goal_pos=(1, 11), max_steps=1024, render_mode="rgb_array", **kwargs):
        self._agent_default_pos = agent_pos
        self._agent_start_dir = agent_dir
        self._goal_default_pos = goal_pos
        self._goal2_default_pos = (11, 11)

        self.width = 13
        self.height = 13

        self.agent_view_size = 3

        mission_space = MissionSpace(mission_func=lambda: "Reach the goal")

        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=self.agent_view_size,
            render_mode=render_mode,
            **kwargs
        )

    def generate_random_position(self, x_range=(-1, -3), y_range=(-1, -6)):
        x = self._rand_int(*x_range)
        y = self._rand_int(*y_range)
        return (x, y)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w     # Left x-coordinate
                yT = j * room_h     # Top y-coordinate
                xR = xL + room_w    # Right x-coordinate
                yB = yT + room_h    # Bottom y-coordinate

                # Vertical wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)

                    if i == 0 and j == 0:
                        upper_half_start = yT + 1
                        upper_half_end = yB
                        pos_u = (xR, self._rand_int(
                            upper_half_start, upper_half_end))
                        self.grid.set(
                            *pos_u,
                            Door(COLOR_NAMES[7],
                                 is_open=False, is_locked=False)
                        )

                # Horizontal wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)

                    if i == 0 and j == 0:
                        left_half_start = xL + 1
                        left_half_end = xR
                        pos_l = (self._rand_int(
                            left_half_start, left_half_end), yB)
                        self.grid.set(*pos_l, None)
                    if i == 1 and j == 0:
                        right_half_start = xL + 1
                        right_half_end = xR
                        pos_r = (self._rand_int(
                            right_half_start, right_half_end), yB)
                        self.grid.set(*pos_r, None)

        # agent start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)

            # assuming fixed start direction
            self.agent_dir = self._agent_start_dir
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        if self._goal2_default_pos is not None:
            goal2 = Goal2()
            self.put_obj(goal2, *self._goal2_default_pos)
            goal2.init_pos, goal2.cur_pos = self._goal2_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.step_count = 0
        obs = self.gen_obs()
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if np.array_equal(self.agent_pos, self._goal2_default_pos):
            reward *= 10

        return obs, reward, terminated, truncated, info

class CustomFourRoomsTwoGoalsRandKeyViewSize3x3(MiniGridEnv):
    # Define class-level metadata
    metadata = {
        'render_modes': ['rgb_array', 'human'],  # Supported render modes
        'render_fps': 10,  # Add render_fps which is required by MiniGrid
        'track_specific_features': True,
        'has_left_right_gates': True,
        'has_left_right_goals': True,
        'left_gate_positions': [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6)],
        'right_gate_positions': [(7, 6), (8, 6), (9, 6), (10, 6), (11, 6)],
        'left_goal_position': (1, 11),
        'right_goal_position': (11, 11),
    }

    def __init__(self, agent_pos=(1, 1), agent_dir=0, goal_pos=(1, 11), goal2_pos=(11, 11), max_steps=1000, render_mode='rgb_array', **kwargs):
        self._agent_default_pos = agent_pos
        self._agent_start_dir = agent_dir
        self._goal_default_pos = goal_pos
        self._goal2_default_pos = goal2_pos

        self.width = 13
        self.height = 13

        self.agent_view_size = 3

        mission_space = MissionSpace(mission_func=lambda: "Reach the goal")

        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=self.agent_view_size,
            render_mode=render_mode,
            **kwargs
        )

    def generate_random_position(self, x_range=(1, 6), y_range=(1, 6)):
        x = self._rand_int(*x_range)
        y = self._rand_int(*y_range)
        return (x, y)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w     # Left x-coordinate
                yT = j * room_h     # Top y-coordinate
                xR = xL + room_w    # Right x-coordinate
                yB = yT + room_h    # Bottom y-coordinate

                # Vertical wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)

                    if i == 0 and j == 0:
                        self.grid.set(
                            xR,
                            yB - room_h // 2,
                            Door(COLOR_NAMES[7], is_open=False, is_locked=True)
                        )

                # Horizontal wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)

                    if i == 0 and j == 0:
                        self.grid.set(xR - room_w // 2, yB, None)
                    if i == 1 and j == 0:
                        self.grid.set(xR - room_w // 2, yB, None)

        pos_key = self.generate_random_position((1, 6), (1, 6))
        while pos_key == self._agent_default_pos:
            pos_key = self.generate_random_position((1, 6), (1, 6))

        self.grid.set(*pos_key, Key(COLOR_NAMES[7]))

        # agent start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)

            # assuming fixed start direction
            self.agent_dir = self._agent_start_dir
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        if self._goal2_default_pos is not None:
            goal2 = Goal2()
            self.put_obj(goal2, *self._goal2_default_pos)
            goal2.init_pos, goal2.cur_pos = self._goal2_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.step_count = 0
        obs = self.gen_obs()
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if np.array_equal(self.agent_pos, self._goal2_default_pos):
            reward *= 10

        return obs, reward, terminated, truncated, info


class CustomFourRoomsDebugNoGoalFixedKeyViewSize3x3(MiniGridEnv):
    # Define class-level metadata
    metadata = {
        'render_modes': ['rgb_array', 'human'],  # Supported render modes
        'render_fps': 10,  # Add render_fps which is required by MiniGrid
        'track_specific_features': True,  # Enable specific feature tracking
        'has_key': True,
        'has_door': True
    }

    def __init__(self, agent_pos=(1, 1), agent_dir=0, max_steps=500, render_mode='rgb_array', **kwargs):
        self._agent_default_pos = agent_pos
        self._agent_start_dir = agent_dir

        self.width = 13
        self.height = 13

        self.agent_view_size = 3
        
        # Episode tracking for key appearance
        self.episodes_completed = 0
        self.episodes_threshold = 5  # Key appears after 5 episodes
        self.key_visible = False

        mission_space = MissionSpace(mission_func=lambda: "Reach the goal")

        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=self.agent_view_size,
            render_mode=render_mode,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w     # Left x-coordinate
                yT = j * room_h     # Top y-coordinate
                xR = xL + room_w    # Right x-coordinate
                yB = yT + room_h    # Bottom y-coordinate

                # Vertical wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)

                    if i == 0 and j == 0:
                        self.grid.set(
                            xR,
                            yB - room_h // 2,
                            Door(COLOR_NAMES[7], is_open=False, is_locked=True)
                        )
                    if i == 0 and j == 1:
                        self.grid.set(xR, yB - room_h // 2, None)

                # Horizontal wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xR, yB)

        # agent start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)

            # assuming fixed start direction
            self.agent_dir = self._agent_start_dir
        else:
            self.place_agent()

        # Only add the key if enough episodes have been completed
        key_pos = (9, 9)
        if self.key_visible:
            self.grid.set(*key_pos, Key(COLOR_NAMES[7]))

        self.mission = 'Reach the goal'

    def reset(self, seed=None, options=None):
        # Check if this reset is due to episode completion (not just initial reset)
        if hasattr(self, 'step_count') and self.step_count > 0:
            self.episodes_completed += 1
            
        # Update key visibility status based on episodes completed
        self.key_visible = self.episodes_completed >= self.episodes_threshold
        
        obs, info = super().reset(seed=seed, options=options)
        self.step_count = 0
        
        # Add episode info to info dict
        info['episodes_completed'] = self.episodes_completed
        info['key_visible'] = self.key_visible
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Add episode tracking info to info dict
        info['episodes_completed'] = self.episodes_completed
        info['key_visible'] = self.key_visible
        
        return obs, reward, terminated, truncated, info


class CustomFourRoomsDebugNoGoalRandKeyViewSize3x3(MiniGridEnv):
    # Define class-level metadata
    metadata = {
        'render_modes': ['rgb_array', 'human'],  # Supported render modes
        'render_fps': 10,  # Add render_fps which is required by MiniGrid
        'track_specific_features': False,  # Use dynamic tracking for this environment
    }

    def __init__(self, agent_pos=(1, 1), agent_dir=0, max_steps=200, render_mode='rgb_array', **kwargs):
        self._agent_default_pos = agent_pos
        self._agent_start_dir = agent_dir
        self.key_generated = False
        self.key_delay_steps = 100   # Number of steps to wait before spawning the key

        self.width = 13
        self.height = 13

        self.agent_view_size = 3

        mission_space = MissionSpace(mission_func=lambda: "Reach the goal")

        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=self.agent_view_size,
            render_mode=render_mode,
            **kwargs
        )

    def generate_random_position(self, x_range=(1, 6), y_range=(1, 6)):
        x = self._rand_int(*x_range)
        y = self._rand_int(*y_range)
        return (x, y)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w     # Left x-coordinate
                yT = j * room_h     # Top y-coordinate
                xR = xL + room_w    # Right x-coordinate
                yB = yT + room_h    # Bottom y-coordinate

                # Vertical wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)

                    if i == 0 and j == 0:
                        self.grid.set(
                            xR,
                            yB - room_h // 2,
                            Door(COLOR_NAMES[7], is_open=False, is_locked=True)
                        )
                    if i == 0 and j == 1:
                        self.grid.set(xR, yB - room_h // 2, None)

                # Horizontal wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xR, yB)

        # agent start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)

            # assuming fixed start direction
            self.agent_dir = self._agent_start_dir
        else:
            self.place_agent()

        self.mission = 'Reach the goal'

    def reset(self, seed=None, options=None):
        self.key_generated = False  # Reset key generation state
        super().reset(seed=seed, options=options)
        self.step_count = 0
        obs = self.gen_obs()
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Generate key after key_delay_steps have passed
        if not self.key_generated and self.step_count >= self.key_delay_steps:
            pos_key = self.generate_random_position((1, 6), (1, 6))
            while pos_key == self.agent_pos:  # Make sure key doesn't spawn on agent
                pos_key = self.generate_random_position((1, 6), (1, 6))
            
            self.grid.set(*pos_key, Key(COLOR_NAMES[7]))
            self.key_generated = True
        
        return obs, reward, terminated, truncated, info





from gymnasium.envs.registration import register

"""Calculates the optimal reward threshold based on the environment layout and reward structure.

Given:
- Agent starts at (1,1)
- Goal2 is at (11,11)
- Minimum path length to goal2 ≈ 20 steps (through the doors)
- Reward = (1 - 0.9 * (step_count/max_steps)) * 10 for goal2

For a 1000-step episode limit:
With optimal path (20 steps):
- Base reward = 1 - 0.9 * (20/1000) = 0.982
- Final reward = 0.982 * 10 ≈ 9.82 

"""

register(
    id="MiniGrid-FourRooms-TwoGoals-Fixed-ViewSize-3x3-v0",
    entry_point="envs.minigrid.minigrid_envs:CustomFourRoomsTwoGoalsFixedViewSize3x3",
    max_episode_steps=1000,
    reward_threshold=9.5,  # Allowing some suboptimality from the theoretical maximum of 9.82
    kwargs={
        'render_mode': 'rgb_array'
    }
)

register(
    id="MiniGrid-FourRooms-TwoGoals-Rand-ViewSize-3x3-v0",
    entry_point="envs.minigrid.minigrid_envs:CustomFourRoomsTwoGoalsRandViewSize3x3",
    max_episode_steps=1024,
    reward_threshold=9.5,  # Same threshold despite slightly different max_steps
    kwargs={
        'render_mode': 'rgb_array'
    }
)

register(
    id="MiniGrid-FourRooms-TwoGoals-RandKey-ViewSize-3x3-v0",
    entry_point="envs.minigrid.minigrid_envs:CustomFourRoomsTwoGoalsRandKeyViewSize3x3",
    max_episode_steps=1000,
    reward_threshold=9.0,  # Slightly lower due to additional key-handling complexity
    kwargs={
        'render_mode': 'rgb_array'
    }
)


register(
    id="MiniGrid-FourRooms-Debug-NoGoal-FixedKey-ViewSize-3x3-v0",
    entry_point="envs.minigrid.minigrid_envs:CustomFourRoomsDebugNoGoalFixedKeyViewSize3x3",
    max_episode_steps=500,
    # reward_threshold=9.0,  # Slightly lower due to additional key-handling complexity
    kwargs={
        'render_mode': 'rgb_array'
    }
)

register(
    id="MiniGrid-FourRooms-Debug-NoGoal-RandKey-ViewSize-3x3-v0",
    entry_point="envs.minigrid.minigrid_envs:CustomFourRoomsDebugNoGoalRandKeyViewSize3x3",
    max_episode_steps=200,
    # reward_threshold=9.0,  # Slightly lower due to additional key-handling complexity
    kwargs={
        'render_mode': 'rgb_array'
    }
)

# Environments are registered above and will be available when the module is imported
