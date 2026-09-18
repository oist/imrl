"""
MiniGrid Episode Video Recorder

Lightweight recorder that renders the MiniGrid grid to RGB frames and saves
mp4 videos. Designed to be minimal and not depend on Craftax textures.
"""

import os
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw


class MiniGridEpisodeVideoRecorder:
    """Simple video recorder for MiniGrid-like environments."""

    def __init__(self,
                 env,
                 env_params,
                 env_name: str,
                 output_dir: str,
                 frame_size: Tuple[int, int] = (512, 512),
                 record_full_map: bool = True,
                 record_fov: bool = True,
                 fov_alpha: int = 60,
                 draw_agent_marker: bool = True):
        self.env = env
        self.env_params = env_params
        self.env_name = env_name
        self.output_dir = output_dir
        self.frame_size = frame_size
        self.record_full_map = record_full_map
        self.record_fov = record_fov
        # Opacity used for FOV overlays (0-255). Lower values reduce the
        # visibility of the highlight which can help avoid green tinting
        # after video encoding; set to 0 to disable highlight composition.
        self.fov_alpha = int(fov_alpha)

        os.makedirs(output_dir, exist_ok=True)

        self.frames = []
        self.recording = False
        # Whether to overlay a POV-relative agent marker on frames. This
        # can be disabled for final recording artifacts to preserve the
        # canonical render produced by MiniGrid or to avoid adding a
        # synthetic blue marker to videos.
        self.draw_agent_marker = draw_agent_marker
        # Initialize palette and helpers
        self._init_palette()

    def _init_palette(self):
        """Initialize a canonical MiniGrid color palette.

        Per your request we avoid importing `gym_minigrid` and instead use the
        official MiniGrid colors as used in the project (taken from the
        MiniGrid website and repository). These are the canonical color ids
        (red, green, blue, purple, yellow, grey, emeraldgreen, neongreen).
        """
        # Canonical MiniGrid palette (color_id -> RGB)
        self.fallback_palette = {
            0: (255, 0, 0),       # red
            1: (0, 160, 0),       # green
            2: (0, 0, 255),       # blue
            3: (160, 32, 240),    # purple
            4: (255, 215, 0),     # yellow
            5: (120, 120, 120),   # grey
            6: (80, 200, 120),   # emeraldgreen
            # Use colors that match MiniGrid's rendered goal colors (preserve vivid neon)
            7: (15, 255, 80),     # neongreen (match MG render)
            8: (15, 255, 80),     # neongreen (alternate index)
        }
        self.use_gym_palette = False

        # Try to bind to minigrid world object renderers (preferred source of truth)
        try:
            from minigrid.core.world_object import (
                COLORS as MG_COLORS,
                IDX_TO_COLOR as MG_IDX_TO_COLOR,
                IDX_TO_OBJECT as MG_IDX_TO_OBJECT,
            )
            from minigrid.core.world_object import (
                Goal as MG_Goal,
                Door as MG_Door,
                Key as MG_Key,
                Ball as MG_Ball,
                Box as MG_Box,
                Wall as MG_Wall,
                Floor as MG_Floor,
                Lava as MG_Lava,
            )

            self._mg_colors = MG_COLORS
            self._mg_idx_to_color = MG_IDX_TO_COLOR
            self._mg_idx_to_object = MG_IDX_TO_OBJECT
            self._mg_classes = {
                'goal': MG_Goal,
                'door': MG_Door,
                'key': MG_Key,
                'ball': MG_Ball,
                'box': MG_Box,
                'wall': MG_Wall,
                'floor': MG_Floor,
                'lava': MG_Lava,
            }
            self._use_minigrid_renderers = True
        except Exception:
            self._use_minigrid_renderers = False
        
    def _construct_grid_from_minigrid_state(self, state):
        """Construct a (H,W,3) integer grid from a compact MiniGridEnvState.

        The returned grid uses the same indexing as MiniGrid's Grid.render
        pipeline: each cell is (obj_idx, color_idx, state_idx).
        """
        try:
            from minigrid.core.world_object import COLOR_TO_IDX, OBJECT_TO_IDX
        except Exception:
            # If minigrid isn't available, fall back to simple numeric constants
            COLOR_TO_IDX = {'yellow': 4, 'green': 1, 'grey': 5}
            OBJECT_TO_IDX = {'floor': 3, 'wall': 2, 'door': 4, 'key': 5, 'goal': 8}

        size = None
        try:
            size = int(getattr(self.env_params, 'size'))
        except Exception:
            try:
                size = int(getattr(self.env, 'params').size)
            except Exception:
                size = 13

        grid = np.zeros((size, size, 3), dtype=int)

        FLOOR = OBJECT_TO_IDX.get('floor', 3)
        WALL = OBJECT_TO_IDX.get('wall', 2)
        DOOR = OBJECT_TO_IDX.get('door', 4)
        KEY = OBJECT_TO_IDX.get('key', 5)
        GOAL = OBJECT_TO_IDX.get('goal', 8)

        floor_color = COLOR_TO_IDX.get('grey', 5)
        wall_color = COLOR_TO_IDX.get('grey', 5)
        door_color = COLOR_TO_IDX.get('yellow', 4)
        key_color = COLOR_TO_IDX.get('yellow', 4)
        goal_color = COLOR_TO_IDX.get('green', 1)

        # Fill base floor
        grid[:, :, 0] = FLOOR
        grid[:, :, 1] = floor_color
        grid[:, :, 2] = 0
        grid[:, 0, 1] = wall_color
        grid[:, -1, 1] = wall_color

        # FourRooms internal walls (matches MiniGrid FourRooms layout)
        # Vertical wall at x=6 except door at (6,3)
        for y in range(size):
            if y == 3:
                continue
            grid[y, 6, 0] = WALL
            grid[y, 6, 1] = wall_color

        # Horizontal wall at y=6 except openings at x=3 and x=9
        for x in range(size):
            if x in (3, 9):
                continue
            grid[6, x, 0] = WALL
            grid[6, x, 1] = wall_color

        # Door at (6,3)
        grid[3, 6, 0] = DOOR
        grid[3, 6, 1] = door_color
        door_state = 0 if getattr(state, 'door_unlocked', False) else 1
        grid[3, 6, 2] = int(door_state)

        # Goals: use env params if available and assign different colors for small/large goals
        try:
            g1 = tuple(getattr(self.env_params, 'goal1_pos'))
        except Exception:
            g1 = (1, 11)
        try:
            g2 = tuple(getattr(self.env_params, 'goal2_pos'))
        except Exception:
            g2 = (11, 11)

        # Prefer to query the wrapped gym env for the actual colors used by
        # the environment (Goal and Goal2 may use 'emeraldgreen' and 'neongreen').
        goal1_color = COLOR_TO_IDX.get('emeraldgreen', 1)
        goal2_color = COLOR_TO_IDX.get('neongreen', COLOR_TO_IDX.get('purple', 3))
        if hasattr(self, 'env') and hasattr(self.env, '_create_wrapped_env'):
            wenv = self.env._create_wrapped_env()
            # Ensure grid is initialized
            try:
                wenv.reset()
            except Exception:
                pass
            try:
                obj1 = wenv.grid.get(g1[0], g1[1])
                if obj1 is not None and hasattr(obj1, 'color'):
                    goal1_color = COLOR_TO_IDX.get(obj1.color, goal1_color)
            except Exception:
                pass
            try:
                obj2 = wenv.grid.get(g2[0], g2[1])
                if obj2 is not None and hasattr(obj2, 'color'):
                    goal2_color = COLOR_TO_IDX.get(obj2.color, goal2_color)
            except Exception:
                pass
            try:
                # Door position commonly at (6,3) in FourRooms; try to read actual door color
                d = wenv.grid.get(6, 3)
                if d is not None and hasattr(d, 'color'):
                    door_color = COLOR_TO_IDX.get(d.color, door_color)
            except Exception:
                pass

        # If we couldn't query via attached env, try to instantiate a local wrapped env
        if (goal1_color == COLOR_TO_IDX.get('emeraldgreen', 1) or goal2_color == COLOR_TO_IDX.get('neongreen', COLOR_TO_IDX.get('purple', 3))) and not getattr(self, 'env', None):
            try:
                from gymnasium.envs.registration import make as gym_make
                try:
                    local_env = gym_make(self.env_name)
                except Exception:
                    local_env = None
                # If gym.make failed, try instantiating the local custom env class
                if local_env is None:
                    try:
                        from envs.minigrid.minigrid_envs import CustomFourRoomsTwoGoalsRandKeyViewSize3x3
                        local_env = CustomFourRoomsTwoGoalsRandKeyViewSize3x3()
                    except Exception:
                        local_env = None
                # attach the instantiated env to recorder for future queries
                try:
                    if local_env is not None:
                        self.env = local_env
                except Exception:
                    pass
            except Exception:
                local_env = None
            # Ensure the instantiated env is initialized (reset) so its grid is populated
            try:
                if local_env is not None:
                    try:
                        local_env.reset()
                    except Exception:
                        # Some custom envs / wrappers may require different reset signatures
                        try:
                            local_env.reset(seed=None)
                        except Exception:
                            pass
            except Exception:
                pass
            # apply wrappers like in setup_minigrid_env
            try:
                if local_env is not None:
                    from minigrid.wrappers import OneHotPartialObsWrapper
                    from envs.minigrid.minigrid_wrappers import DictToArrayObsWrapper
                    local_env = OneHotPartialObsWrapper(local_env)
                    local_env = DictToArrayObsWrapper(local_env)
                    # Reset again after wrapping to ensure wrapper-initialized state
                    try:
                        local_env.reset()
                    except Exception:
                        try:
                            local_env.reset(seed=None)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                # inspect grid objects; the active grid may live on the
                # original env instance (self.env) rather than the wrapper
                inspect_env = getattr(self, 'env', None) if getattr(self, 'env', None) is not None else local_env
                if inspect_env is not None and getattr(inspect_env, 'grid', None) is not None:
                    obj1 = inspect_env.grid.get(g1[0], g1[1])
                    if obj1 is not None and hasattr(obj1, 'color'):
                        goal1_color = COLOR_TO_IDX.get(obj1.color, goal1_color)
            except Exception:
                pass
            try:
                inspect_env = getattr(self, 'env', None) if getattr(self, 'env', None) is not None else local_env
                if inspect_env is not None and getattr(inspect_env, 'grid', None) is not None:
                    obj2 = inspect_env.grid.get(g2[0], g2[1])
                    if obj2 is not None and hasattr(obj2, 'color'):
                        goal2_color = COLOR_TO_IDX.get(obj2.color, goal2_color)
            except Exception:
                pass
            try:
                if local_env is not None:
                    local_env.close()
            except Exception:
                pass

        grid[g1[1], g1[0], 0] = GOAL
        grid[g1[1], g1[0], 1] = goal1_color
        grid[g2[1], g2[0], 0] = GOAL
        grid[g2[1], g2[0], 1] = goal2_color

        # Key
        try:
            kp = np.array(state.key_pos)
            if kp.shape and kp[0] >= 0:
                kx, ky = int(kp[0]), int(kp[1])
                grid[ky, kx, 0] = KEY
        except Exception:
            pass

        return grid

    def start_recording(self):
        self.frames = []
        self.recording = True
        # Start recording: frames list is cleared and recording flag set.

    def stop_recording(self):
        self.recording = False

    def add_frame(self, env_state, observation):
        """Add a rendered frame for the given env state.

        env_state may be wrapped or batched; we handle common cases by
        unwrapping and taking the first element when necessary.
        """
        if not self.recording:
            return

        try:
            state = env_state.env_state if hasattr(env_state, 'env_state') else env_state
            # If this is a compact MiniGridEnvState (JAX wrapper), it does not
            # contain a full `grid`. Synthesize a full grid representation so
            # the renderer can use the canonical MiniGrid tile pipeline.
            try:
                from envs.minigrid.minigrid_jax_env import MiniGridEnvState
                is_minigrid_state = isinstance(state, MiniGridEnvState)
            except Exception:
                is_minigrid_state = False
            if is_minigrid_state and not hasattr(state, 'grid'):
                try:
                    # Synthesize a grid and wrap it together with agent info
                    grid = self._construct_grid_from_minigrid_state(state)
                    from types import SimpleNamespace
                    # Convert potential JAX arrays to numpy scalars
                    try:
                        agent_pos = np.array(state.agent_pos)
                        if agent_pos.ndim > 1:
                            agent_pos = agent_pos[0]
                        agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
                    except Exception:
                        agent_pos = (-1, -1)
                    try:
                        agent_dir = int(state.agent_dir)
                    except Exception:
                        agent_dir = 0
                    try:
                        has_key = bool(state.has_key)
                    except Exception:
                        has_key = False
                    try:
                        door_unlocked = bool(state.door_unlocked)
                    except Exception:
                        door_unlocked = False

                    state = SimpleNamespace(grid=grid, agent_pos=agent_pos, agent_dir=agent_dir, has_key=has_key, door_unlocked=door_unlocked)
                except Exception:
                    # If synthesis fails, fall back to attempting to render the
                    # raw state (error will be logged once)
                    pass
            # If batched, take first
            try:
                # JAX arrays expose shape attribute
                if hasattr(state.agent_pos, 'shape') and len(state.agent_pos.shape) > 1:
                    # take index 0
                    def _maybe_take(x):
                        try:
                            return np.array(x)[0]
                        except Exception:
                            return x
                    state = type(state)(**{k: _maybe_take(getattr(state, k)) for k in state._fields}) if hasattr(state, '_fields') else state
            except Exception:
                pass

            frame = self._render_state_to_frame(state)
            if frame is not None:
                self.frames.append(frame)
        except Exception as e:
            if not hasattr(self, '_render_error_shown'):
                print(f"⚠️  MiniGrid recorder failed to render frame: {e}")
                self._render_error_shown = True

    def _render_state_to_frame(self, state) -> np.ndarray:
        # Expect state.grid with shape (H, W, 3): (obj_id, color_id, state_id)
        # If `state` is already a numpy grid (synthesized earlier), accept that
        if hasattr(state, 'grid'):
            grid = np.array(state.grid)
        else:
            grid = np.array(state)
        H, W, _ = grid.shape

        # Tile size to scale up to frame_size
        tile_h = max(1, self.frame_size[1] // H)
        tile_w = max(1, self.frame_size[0] // W)

        # Use a black background to match MiniGrid's default floor color
        img = Image.new('RGB', (W * tile_w, H * tile_h), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        def palette_lookup(color_id: int):
            if self.use_gym_palette and self.gym_palette is not None:
                # gym palette maps names -> rgb; jax env uses indices; best-effort: try index->name mapping
                try:
                    # Map numeric idx to gym color names via known ordering
                    # Known ordering (COLOR_TO_IDX): red, green, blue, purple, yellow, grey, emeraldgreen, neongreen
                    name_order = ['red', 'green', 'blue', 'purple', 'yellow', 'grey', 'emeraldgreen', 'neongreen']
                    name = name_order[int(color_id) % len(name_order)]
                    return tuple(int(v * 255) if isinstance(v, float) and v <= 1.0 else v for v in self.gym_palette[name])
                except Exception:
                    return self.fallback_palette.get(color_id, (200, 200, 200))
            else:
                return self.fallback_palette.get(color_id, (200, 200, 200))

        def base_color_for_object(obj: int):
            # Default base colors matching MiniGrid's COLORS for consistency
            base = {
                0: (0, 0, 0),         # unseen (black)
                1: (100, 100, 100),   # empty / floor (grey)
                2: (100, 100, 100),   # wall (grey)
                3: (100, 100, 100),   # floor (grey)
                4: (100, 100, 100),   # door (grey)
                5: (255, 255, 0),     # key (yellow)
                6: (255, 0, 0),       # ball (red)
                7: (112, 39, 160),    # box (purple)
                8: (0, 255, 0),       # goal (green)
                9: (255, 0, 0),       # lava (red)
                10: (0, 0, 255),      # agent (blue)
            }
            return base.get(obj, (200, 200, 200))

        for y in range(H):
            for x in range(W):
                obj = int(grid[y, x, 0])
                col = int(grid[y, x, 1]) if grid.shape[2] > 1 else 0

                # Choose color: for colorable objects prefer palette color, otherwise base color
                if obj in (4, 5, 6, 7, 8):
                    color = palette_lookup(col)
                else:
                    color = base_color_for_object(obj)

                x0 = x * tile_w
                y0 = y * tile_h
                x1 = x0 + tile_w
                y1 = y0 + tile_h
                # If minigrid renderers are available, use official object renderers
                if getattr(self, '_use_minigrid_renderers', False):
                    # Use official Grid.tile rendering for highest fidelity (includes
                    # supersampling, grid lines, and highlight blending). This should
                    # reproduce the canonical MiniGrid visuals for doors, keys, etc.
                    try:
                        from minigrid.core.grid import Grid as MG_Grid
                        obj_name = self._mg_idx_to_object.get(obj, None) if hasattr(self, '_mg_idx_to_object') else None
                        color_name = self._mg_idx_to_color.get(col, None) if hasattr(self, '_mg_idx_to_color') else None

                        if obj_name is None:
                            # Nothing to render with official renderer; fall back
                            raise RuntimeError("no object mapping")

                        # Construct object instance for the cell (or None for empty)
                        if obj_name in ('empty', 'unseen'):
                            cell_obj = None
                        else:
                            cls = self._mg_classes.get(obj_name, None)
                            if cls is None:
                                cell_obj = None
                            else:
                                # Prefer the actual object's color from the wrapped
                                # env grid if available to respect custom env colors
                                try:
                                    if hasattr(self, 'env') and hasattr(self.env, '_create_wrapped_env'):
                                        wenv_col = None
                                        try:
                                            wenv = self.env._create_wrapped_env()
                                            wobj = wenv.grid.get(x, y)
                                            if wobj is not None and hasattr(wobj, 'color'):
                                                wenv_col = wobj.color
                                        except Exception:
                                            wenv_col = None
                                        if wenv_col is not None:
                                            color_name = wenv_col
                                except Exception:
                                    pass

                                if obj_name == 'door':
                                    o = cls(color_name if color_name is not None else 'grey')
                                    state_val = int(grid[y, x, 2]) if grid.shape[2] > 2 else 1
                                    o.is_open = (state_val == 0)
                                    o.is_locked = (state_val == 2)
                                    cell_obj = o
                                else:
                                    # If the wrapped env didn't provide a color name,
                                    # choose sensible defaults per-object so floors
                                    # and walls appear neutral instead of vivid
                                    # colors (e.g., 'blue') which can dominate the
                                    # scene.
                                    default_color = None
                                    if obj_name in ('floor', 'wall', 'goal'):
                                        default_color = 'grey'
                                    else:
                                        default_color = 'blue'
                                    cell_obj = cls(color_name if color_name is not None else default_color)

                        # Determine whether this cell should be highlighted (in-agent FOV)
                        # Use MiniGrid's visibility mask when possible to get the
                        # triangular, orientation-aware FOV instead of a rectangular box.
                        highlight = False
                        view_size = getattr(self.env_params, 'agent_view_size', None) or getattr(self.env, 'params', None) and getattr(self.env.params, 'agent_view_size', None) or 3
                        try:
                            agent_pos = np.array(state.agent_pos)
                            if agent_pos.ndim == 2:
                                agent_pos = agent_pos[0]
                            ax = int(agent_pos[0])
                            ay = int(agent_pos[1])
                        except Exception:
                            ax = ay = -999

                        # Try to use wrapped MiniGrid env's gen_obs_grid to compute
                        # a precise visibility mask for the agent's current pose.
                        try:
                            if hasattr(self, 'env') and hasattr(self.env, '_create_wrapped_env'):
                                wenv = self.env._create_wrapped_env()
                                # set agent pose
                                try:
                                    wenv.agent_pos = (int(ax), int(ay))
                                    wenv.agent_dir = int(getattr(state, 'agent_dir', 0))
                                except Exception:
                                    pass
                                # get vis mask and compute top-left world coords
                                try:
                                    _, vis_mask = wenv.gen_obs_grid(view_size)
                                    topX, topY, _, _ = wenv.get_view_exts(view_size)
                                    # vis_mask indices: vis_i (x), vis_j (y)
                                    # Map vis_mask True cells back to world coords
                                    if vis_mask is not None and vis_mask.shape[0] == view_size:
                                        for vis_j in range(view_size):
                                            for vis_i in range(view_size):
                                                if not vis_mask[vis_i, vis_j]:
                                                    continue
                                                # compute absolute coords using same mapping as MiniGrid.get_full_render
                                                f_vec = wenv.dir_vec
                                                r_vec = wenv.right_vec
                                                top_left = (np.array(wenv.agent_pos) + f_vec * (view_size - 1) - r_vec * (view_size // 2)).astype(int)
                                                abs_pos = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                                                abs_x = int(abs_pos[0])
                                                abs_y = int(abs_pos[1])
                                                if abs_x == x and abs_y == y:
                                                    highlight = True
                                                    break
                                            if highlight:
                                                break
                                except Exception:
                                    highlight = (abs(x - ax) <= view_size // 2 and abs(y - ay) <= view_size // 2)
                        except Exception:
                            # Fallback to a rectangular FOV if anything fails
                            half = int(view_size) // 2
                            highlight = (x >= ax - half and x <= ax + half and y >= ay - half and y <= ay + half)

                        # Disable highlight if FOV recording is disabled
                        if not getattr(self, 'record_fov', True):
                            highlight = False

                        # Render a canonical tile using MiniGrid's Grid.render_tile
                        # Prefer letting MiniGrid draw the agent on its world-tile
                        # when using the official renderers; for other tiles and
                        # when not using MG renderers pass agent_dir=None so the
                        # agent won't be drawn prematurely (we draw a POV marker
                        # later for consistency).
                        agent_dir_param = None
                        try:
                            # Let MiniGrid draw the canonical agent triangle when
                            # using the official renderers. The `draw_agent_marker`
                            # flag only controls whether we add a synthetic
                            # overlay marker (which we avoid for recordings), but
                            # the MG renderer's triangle is the canonical visual
                            # and should still be drawn to reproduce the env.
                            if getattr(self, '_use_minigrid_renderers', False) and ax == x and ay == y:
                                agent_dir_param = int(getattr(state, 'agent_dir', 0))
                        except Exception:
                            agent_dir_param = None
                        tile = MG_Grid.render_tile(cell_obj, agent_dir=agent_dir_param, highlight=highlight, tile_size=max(tile_h, tile_w))
                        # When MiniGrid's renderer applies a highlight we can
                        # scale the effective highlight via `fov_alpha`. Instead
                        # of changing MG internals we render a second tile
                        # without highlight and blend the highlight component
                        # down using `fov_alpha` so recorded frames don't get
                        # an overwhelming green tint after encoding.
                        try:
                            if highlight and getattr(self, 'fov_alpha', 60) > 0 and getattr(self, 'fov_alpha', 60) < 255:
                                alpha_frac = float(self.fov_alpha) / 255.0
                                tile_no_high = MG_Grid.render_tile(cell_obj, agent_dir=agent_dir_param, highlight=False, tile_size=max(tile_h, tile_w))
                                # Compute the highlight overlay and scale it
                                overlay = tile.astype('float32') - tile_no_high.astype('float32')
                                tile = (tile_no_high.astype('float32') + overlay * alpha_frac).clip(0, 255).astype(tile.dtype)
                        except Exception:
                            pass
                        # If this is a goal, ensure the rendered object pixels use
                        # the environment's intended palette color (some custom
                        # envs use names not recognized by the local MiniGrid
                        # package). We recolor non-floor pixels to the
                        # palette_lookup color for the given color index. This
                        # is applied unconditionally so that goals are colored
                        # correctly even when the local MG build doesn't know
                        # about custom color names (e.g., 'emeraldgreen').
                        try:
                            if obj_name == 'goal':
                                # Prefer to render a Goal object with the env's
                                # exact color name (if available) to reproduce the
                                # canonical tile; this avoids aggressive recoloring
                                # that can bias channels. Fall back to shading-
                                # preserving recolor when MG cannot render with the
                                # requested color name.
                                color_name = None
                                # Try to get color name from wrapped env grid
                                try:
                                    if hasattr(self, 'env') and hasattr(self.env, '_create_wrapped_env'):
                                        wenv = self.env._create_wrapped_env()
                                        wobj = wenv.grid.get(x, y)
                                        if wobj is not None and hasattr(wobj, 'color'):
                                            color_name = wobj.color
                                except Exception:
                                    color_name = None

                                # If no name yet, try mapping from color index
                                if color_name is None and hasattr(self, '_mg_idx_to_color'):
                                    try:
                                        color_name = self._mg_idx_to_color.get(col, None)
                                    except Exception:
                                        color_name = None

                                # Try to render using MG Goal with the resolved name
                                try:
                                    from minigrid.core.world_object import Goal as MG_Goal
                                    if color_name is not None:
                                        try:
                                            goal_obj = MG_Goal(color_name)
                                            goal_tile = MG_Grid.render_tile(goal_obj, agent_dir=None, highlight=highlight, tile_size=max(tile_h, tile_w))
                                            # Use rendered goal tile directly (it preserves
                                            # MG's shading and palette). However, some
                                            # local MiniGrid builds don't recognize custom
                                            # color names (e.g., 'emeraldgreen') and will
                                            # render to a default color (often blue). If
                                            # that happens, detect the mismatch and
                                            # instead recolor the non-floor pixels to the
                                            # intended canonical RGB.
                                            tile = goal_tile
                                            try:
                                                floor_rgb = base_color_for_object(3)
                                                floor_tile = np.zeros_like(tile) + np.array(floor_rgb, dtype=tile.dtype)
                                                mask = np.any(tile != floor_tile, axis=2)
                                                if np.any(mask):
                                                    rendered_mean = tile[mask].astype(np.float32).mean(axis=0)
                                                    desired_rgb = palette_lookup(col)
                                                    desired_rgb = np.array(desired_rgb, dtype=np.float32)
                                                    if np.linalg.norm(rendered_mean - desired_rgb) > 30:
                                                        # Replace non-floor pixels with the canonical RGB
                                                        tile = tile.copy()
                                                        tile[mask] = tuple(int(min(255, max(0, round(v)))) for v in desired_rgb)
                                            except Exception:
                                                pass
                                        except Exception:
                                            # Fall back to shading-preserving recolor below
                                            raise
                                except Exception:
                                    # Shading-preserving recolor fallback will be
                                    # applied below if necessary.
                                    pass
                        except Exception:
                            # If any unexpected error occurs while handling
                            # goal recoloring, fall back silently. We
                            # don't want a single recolor failure to
                            # prevent frame rendering.
                            pass

                        # Ensure goal tiles use the intended palette color
                        # even if the MG renderer used a default (unknown)
                        # color name. Recolor non-floor pixels to the
                        # canonical RGB for the given color index.
                        try:
                            if obj_name in ('goal', 'key'):
                                floor_rgb = base_color_for_object(3)
                                floor_tile = np.zeros_like(tile) + np.array(floor_rgb, dtype=tile.dtype)
                                mask = np.any(tile != floor_tile, axis=2)
                                if np.any(mask):
                                    desired_rgb = palette_lookup(col)
                                    recol_uint8 = np.array(tuple(int(min(255, max(0, round(v)))) for v in desired_rgb), dtype=np.uint8)
                                    tile = tile.copy()
                                    tile[mask] = recol_uint8
                        except Exception:
                            pass
                        # If the object is a door, composite it onto a floor tile so
                        # the background matches the environment floor (transparent
                        # door icon over floor) rather than painting the door as a
                        # full-tile background.
                        try:
                            if obj_name == 'door':
                                # Composite the door icon over a pure floor-colored
                                # background (black by default) so door tiles show
                                # floor pixels underneath (transparent background
                                # effect). This ensures door backgrounds match the
                                # floor & key backgrounds.
                                try:
                                    floor_rgb = base_color_for_object(3)
                                    floor_tile = np.zeros_like(tile) + np.array(floor_rgb, dtype=tile.dtype)
                                    mask = np.any(tile != floor_tile, axis=2)
                                    comp = floor_tile.copy()
                                    comp[mask] = tile[mask]
                                    tile = comp
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # tile is a numpy array (H, W, 3); paste into image
                        try:
                            img.paste(Image.fromarray(tile.astype('uint8')), (x0, y0))
                        except Exception:
                            arr = np.array(img)
                            # ensure shapes match
                            h = min(arr.shape[0] - y0, tile.shape[0])
                            w = min(arr.shape[1] - x0, tile.shape[1])
                            arr[y0:y0 + h, x0:x0 + w] = tile[:h, :w]
                            img = Image.fromarray(arr)
                        continue
                    except Exception:
                        # Fall back to local rendering on any failure
                        pass
                # Textured rendering fallback (non-minigrid or failure)
                if obj == 2:  # wall (brick-like)
                    draw.rectangle([x0, y0, x1, y1], fill=color)
                    brick = tuple(max(0, c - 30) for c in color)
                    bw = max(1, tile_w // 6)
                    bh = max(1, tile_h // 6)
                    for yy in range(y0, y1, bh):
                        for xx in range(x0 + ((yy - y0) // bh) % 2 * bw, x1, bw * 2):
                            rx0 = xx
                            ry0 = yy
                            rx1 = min(xx + bw * 2 - 1, x1)
                            ry1 = min(yy + bh - 1, y1)
                            draw.rectangle([rx0, ry0, rx1, ry1], fill=brick)
                elif obj == 3:  # floor (textured)
                    draw.rectangle([x0, y0, x1, y1], fill=color)
                    # subtle grain
                    grain = tuple(min(255, int(c * 1.04)) for c in color)
                    if (x + y) % 2 == 0:
                        draw.rectangle([x0 + tile_w // 8, y0 + tile_h // 8, x1 - tile_w // 8, y1 - tile_h // 8], outline=grain)
                else:
                    # For objects, draw floor background first and then an icon
                    if obj in (4, 5, 6, 7, 8):
                        floor_col = base_color_for_object(3)
                        draw.rectangle([x0, y0, x1, y1], fill=floor_col)
                        # Draw simplified icons for each object type
                        cx = (x0 + x1) // 2
                        cy = (y0 + y1) // 2
                        bw = max(1, tile_w // 6)
                        bh = max(1, tile_h // 6)
                        if obj == 4:  # door
                            # door frame and inset panel
                            pad = max(1, min(tile_w, tile_h) // 8)
                            draw.rectangle([x0 + pad, y0 + pad, x1 - pad, y1 - pad], fill=color)
                            inner_pad = pad + max(1, pad // 2)
                            draw.rectangle([x0 + inner_pad, y0 + inner_pad, x1 - inner_pad, y1 - inner_pad], outline=(0,0,0))
                            # handle
                            hx = x1 - pad * 2
                            hy = cy
                            hr = max(1, pad // 2)
                            draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=(0,0,0))
                        elif obj == 5:  # key
                            # Shaft
                            shaft_w = max(1, tile_w // 10)
                            shaft_h = max(1, tile_h // 4)
                            draw.rectangle([cx - shaft_w, cy - shaft_h, cx + shaft_w, cy + shaft_h], fill=color)
                            # Ring
                            rr = max(1, min(tile_w, tile_h) // 6)
                            draw.ellipse([cx - rr, cy - rr*2, cx + rr, cy], outline=color, width=1)
                        elif obj == 6:  # ball
                            rr = max(2, min(tile_w, tile_h) // 4)
                            draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=color, outline=(0,0,0))
                        elif obj == 7:  # box
                            pad = max(1, min(tile_w, tile_h) // 8)
                            draw.rectangle([x0 + pad, y0 + pad, x1 - pad, y1 - pad], fill=color, outline=(0,0,0))
                        elif obj == 8:  # goal
                            rr = max(2, min(tile_w, tile_h) // 4)
                            draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=color, outline=(0,0,0))
                    else:
                        draw.rectangle([x0, y0, x1, y1], fill=color)

        # Overlay agent marker at the agent's POV-relative tile (bottom-center
        # within the triangular FOV) so viewers can see facing direction.
        # When using the official MiniGrid renderers we still add a small
        # POV-relative marker on top of the rendered tiles for clarity.
        try:
            agent_pos = np.array(state.agent_pos)
            if agent_pos.ndim == 2:
                agent_pos = agent_pos[0]
            ax = int(agent_pos[0])
            ay = int(agent_pos[1])
            # Determine direction
            try:
                dir_val = int(state.agent_dir)
            except Exception:
                dir_val = 0
            # Compute POV-relative pixel coordinates for the agent marker
            view_size = getattr(self.env_params, 'agent_view_size', None) or (getattr(self.env, 'params', None) and getattr(self.env.params, 'agent_view_size', None)) or 3
            pov_x = view_size // 2
            pov_y = view_size - 1
            try:
                if hasattr(self, 'env') and hasattr(self.env, '_create_wrapped_env'):
                    wenv = self.env._create_wrapped_env()
                    # attempt to set agent pose for get_view_exts
                    try:
                        wenv.agent_pos = (int(ax), int(ay))
                        wenv.agent_dir = int(getattr(state, 'agent_dir', 0))
                    except Exception:
                        pass
                    try:
                        topX, topY, _, _ = wenv.get_view_exts(view_size)
                    except Exception:
                        topX, topY = ax - view_size // 2, ay - view_size // 2
                else:
                    topX, topY = ax - view_size // 2, ay - view_size // 2
            except Exception:
                topX, topY = ax - view_size // 2, ay - view_size // 2

            base_x = max(0, int(topX) * tile_w)
            base_y = max(0, int(topY) * tile_h)
            mx = base_x + int((pov_x + 0.5) * tile_w)
            my = base_y + int((pov_y + 0.5) * tile_h)

            # Optionally draw an agent marker; disabled when `self.draw_agent_marker`
            # is False so recorded videos can avoid synthetic overlays.
            if getattr(self, 'draw_agent_marker', True):
                agent_marker_color = (20, 20, 220)
                if getattr(self, '_use_minigrid_renderers', False):
                    # Defer to MG renderer by not adding an extra circle
                    pass
                else:
                    # Draw oriented triangle centered at POV-relative pixel coordinates
                    r = max(2, min(tile_w, tile_h) // 3)
                    if dir_val == 0:  # right
                        tri = [(mx + r, my), (mx - r, my - r), (mx - r, my + r)]
                    elif dir_val == 2:  # left
                        tri = [(mx - r, my), (mx + r, my - r), (mx + r, my + r)]
                    elif dir_val == 3:  # up
                        tri = [(mx, my - r), (mx - r, my + r), (mx + r, my + r)]
                    else:  # down (1)
                        tri = [(mx, my + r), (mx - r, my - r), (mx + r, my - r)]
                    draw.polygon(tri, fill=agent_marker_color, outline=(0, 0, 0))
            # Optionally overlay FOV (semi-transparent square). When using
            # the official MiniGrid renderers we skip this manual overlay
            # because Grid.render_tile already handles highlighting.
            try:
                if not getattr(self, '_use_minigrid_renderers', False) and getattr(self, 'record_fov', True):
                    # Compute triangular FOV overlay and draw it
                    half = int(view_size) // 2
                    fov_x0 = max(0, ax - half) * tile_w
                    fov_y0 = max(0, ay - half) * tile_h
                    fov_x1 = min(W, ax + half + 1) * tile_w
                    fov_y1 = min(H, ay + half + 1) * tile_h
                    overlay = Image.new('RGBA', (W * tile_w, H * tile_h), (0, 0, 0, 0))
                    o_draw = ImageDraw.Draw(overlay)
                    nx0 = fov_x0 / (W * tile_w)
                    ny0 = fov_y0 / (H * tile_h)
                    nx1 = fov_x1 / (W * tile_w)
                    ny1 = fov_y1 / (H * tile_h)
                    if dir_val == 0:  # right
                        tri = [(nx1, (ny0 + ny1) / 2), (nx0, ny0), (nx0, ny1)]
                    elif dir_val == 2:  # left
                        tri = [(nx0, (ny0 + ny1) / 2), (nx1, ny0), (nx1, ny1)]
                    elif dir_val == 3:  # up
                        tri = [((nx0 + nx1) / 2, ny0), (nx0, ny1), (nx1, ny1)]
                    else:  # down
                        tri = [((nx0 + nx1) / 2, ny1), (nx0, ny0), (nx1, ny0)]
                    px_tri = [(int(v[0] * W * tile_w), int(v[1] * H * tile_h)) for v in tri]
                    # Use configurable alpha channel for the overlay so we can
                    # tune or disable it when recordings show color tints.
                    o_draw.polygon(px_tri, fill=(255, 255, 255, int(getattr(self, 'fov_alpha', 60))))
                    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                    draw = ImageDraw.Draw(img)
                    # Draw a small directional agent marker inside the FOV at
                    # the POV-relative tile (bottom-center) so facing is clear.
                    try:
                        if getattr(self, 'draw_agent_marker', True):
                            agent_marker_color = (20, 20, 220)
                            r2 = max(2, min(tile_w, tile_h) // 8)
                            draw.ellipse([mx - r2, my - r2, mx + r2, my + r2], fill=agent_marker_color, outline=(0, 0, 0))
                    except Exception:
                        pass
            except Exception:
                pass

            # If using the official renderer, add a POV-relative directional
            # marker overlay (bottom-center within the POV) so that viewers
            # can clearly see-facing direction relative to the triangular FOV.
            if getattr(self, '_use_minigrid_renderers', False) and getattr(self, 'record_fov', True):
                try:
                    view_size = getattr(self.env_params, 'agent_view_size', None) or getattr(self.env, 'params', None) and getattr(self.env.params, 'agent_view_size', None) or 3
                    try:
                        fov_topX, fov_topY, _, _ = (self.env._create_wrapped_env().get_view_exts(view_size) if hasattr(self, 'env') and hasattr(self.env, '_create_wrapped_env') else (ax - view_size // 2, ay - view_size // 2, 0, 0))
                    except Exception:
                        fov_topX, fov_topY = ax - view_size // 2, ay - view_size // 2

                    pov_x = view_size // 2
                    pov_y = view_size - 1
                    try:
                        base_x = fov_x0
                        base_y = fov_y0
                    except Exception:
                        base_x = max(0, int(fov_topX) * tile_w)
                        base_y = max(0, int(fov_topY) * tile_h)

                    mx = base_x + int((pov_x + 0.5) * tile_w)
                    my = base_y + int((pov_y + 0.5) * tile_h)
                    try:
                        if getattr(self, 'draw_agent_marker', True):
                            agent_marker_color = (20, 20, 220)
                            r2 = max(2, min(tile_w, tile_h) // 8)
                            draw.ellipse([mx - r2, my - r2, mx + r2, my + r2], fill=agent_marker_color, outline=(0, 0, 0))
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

        # Resize to frame_size if needed
        # Remove any accidental agent-marker pixels that ended up on the
        # agent's own tile (we want the agent marker only at the POV-relative
        # tile). This is defensive: if MG or fallback rendering placed a
        # marker on the agent's tile, erase it and replace with floor color.
        # If enabled, remove accidental agent-marker pixels that ended up on
        # the agent's own tile (we want the agent marker only at the POV-
        # relative tile). Skip entirely when agent markers are disabled.
        try:
            if getattr(self, 'draw_agent_marker', True):
                # Convert to array for in-place edits
                arr = np.array(img)
                agent_pos = np.array(state.agent_pos)
                if agent_pos.ndim == 2:
                    agent_pos = agent_pos[0]
                ax = int(agent_pos[0]); ay = int(agent_pos[1])
                if 0 <= ax < W and 0 <= ay < H:
                    x0 = ax * tile_w; y0 = ay * tile_h
                    sub = arr[y0:y0 + tile_h, x0:x0 + tile_w]
                    agent_marker_color = np.array((20, 20, 220), dtype=np.int16)
                    # Use a distance threshold to catch anti-aliased marker pixels
                    diff = np.linalg.norm(sub.astype(np.int16) - agent_marker_color[None, None, :], axis=2)
                    mask = diff < 100
                    if mask.any():
                        floor_rgb = base_color_for_object(3)
                        sub[mask] = np.array(floor_rgb, dtype=sub.dtype)
                        arr[y0:y0 + tile_h, x0:x0 + tile_w] = sub
                        img = Image.fromarray(arr)
        except Exception:
            pass

        img_resized = img.resize((self.frame_size[0], self.frame_size[1]), Image.NEAREST)
        return np.array(img_resized)

    def save_video(self, filename: str, fps: int = 10) -> Optional[str]:
        if len(self.frames) == 0:
            print("⚠️  MiniGrid recorder: No frames to save")
            return None

        output_path = os.path.join(self.output_dir, f"{filename}.mp4")

        try:
            import imageio
            # Use libx264 + yuv420p (YUV420) for broad compatibility and to
            # match the behavior of `EpisodeVideoRecorder`. This codec is
            # widely supported by players and wandb, at the expense of
            # chroma subsampling which can affect very small saturated
            # regions; that tradeoff is intentional to match the project's
            # canonical recorder.
            with imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p') as writer:
                for frame in self.frames:
                    try:
                        from PIL import Image as PILImage
                        frame_rgb = np.asarray(PILImage.fromarray(frame).convert('RGB'), dtype=np.uint8)
                    except Exception:
                        frame_rgb = frame.astype('uint8')
                    writer.append_data(frame_rgb)
            print(f"✅ MiniGrid video saved (yuv420p): {output_path}")
            return output_path
        except Exception as e:
            # If imageio is not present or writing fails, attempt the
            # same OpenCV fallback as `EpisodeVideoRecorder` for parity.
            try:
                import cv2
                height, width = self.frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                for frame in self.frames:
                    try:
                        frame_bgr = cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_RGB2BGR)
                    except Exception:
                        frame_bgr = frame.astype('uint8')
                    out.write(frame_bgr)
                out.release()
                print(f"✅ MiniGrid video saved (opencv mp4v): {output_path}")
                return output_path
            except Exception:
                print(f"⚠️  Could not save miniGrid video: {e}")
                return None

    def log_to_wandb(self, video_path: str, key: str = "video/best_episode", caption: str = "Best Episode"):
        try:
            import wandb
            if wandb.run is None:
                print("⚠️  wandb not initialized, skipping miniGrid video upload")
                return
            wandb.log({key: wandb.Video(video_path, caption=caption, format="mp4")}, commit=False)
            print(f"✅ MiniGrid video logged to wandb: {key}")
        except Exception as e:
            print(f"⚠️  Failed to upload miniGrid video to wandb: {e}")
