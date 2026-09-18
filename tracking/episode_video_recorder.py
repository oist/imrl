"""
Episode Video Recorder for Craftax
Records episodes with full environment map and FOV overlay using textures
"""

import os
import numpy as np
from typing import Optional, Tuple
import wandb


class EpisodeVideoRecorder:
    """Records episode frames showing both full map and agent FOV"""
    
    def __init__(self, 
                 env,
                 env_params,
                 env_name: str,
                 output_dir: str,
                 frame_size: Tuple[int, int] = (1024, 1024),
                 record_full_map: bool = True,
                 record_fov: bool = True):
        """
        Initialize the video recorder
        
        Args:
            env: The Craftax environment
            env_params: Environment parameters
            env_name: Name of the environment
            output_dir: Directory to save videos
            frame_size: Size of the output video frames (width, height)
            record_full_map: Whether to include full map view
            record_fov: Whether to include agent FOV view
        """
        self.env = env
        self.env_params = env_params
        self.env_name = env_name
        # Detect MiniGrid-like envs so we can render observations appropriately.
        # Be conservative: don't assume 'render()' implies MiniGrid (many envs
        # including Craftax implement render). Prefer explicit indicators such
        # as the env name containing 'MiniGrid', the env module containing
        # 'minigrid', or the presence of a MiniGrid-specific helper like
        # `get_obs`.
        # Only consider get_obs as a MiniGrid indicator if the env's module
        # also includes 'minigrid' to avoid misclassifying other envs
        # (e.g., Craftax) that expose a get_obs method.
        self.is_minigrid = (
            ("MiniGrid" in env_name)
            or ('minigrid' in getattr(env.__class__, '__module__', '').lower())
            or (hasattr(env, 'get_obs') and 'minigrid' in getattr(env.__class__, '__module__', '').lower())
        )
        if self.is_minigrid:
            print("ℹ️  MiniGrid environment detected - using observation-based rendering")
        self.output_dir = output_dir
        self.frame_size = frame_size
        self.record_full_map = record_full_map
        self.record_fov = record_fov
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize renderer with textures
        # Try to import and use Craftax's texture-based renderer
        # Do NOT enable Craftax texture rendering for MiniGrid environments.
        # Using Craftax textures when recording MiniGrid videos caused Craftax
        # visuals to appear in saved videos. Respect Minigrid fidelity by
        # preferring native env.render() or the observation/state-based
        # fallback renderers implemented below.
        if self.is_minigrid:
            self.render_function = None
            self.use_textures = False
            print("ℹ️  Texture rendering disabled for MiniGrid environments")
        else:
            try:
                # Determine which Craftax version we're using
                if "Classic" in env_name:
                    from envs.craftax.craftax_classic.renderer import render_craftax_pixels
                else:
                    from envs.craftax.craftax.renderer import render_craftax_pixels

                self.render_function = render_craftax_pixels
                self.use_textures = True
                print("✅ Texture rendering enabled for episode recording")
            except Exception as e:
                print(f"⚠️  Could not load texture renderer: {e}")
                print("   Falling back to simple color rendering")
                self.render_function = None
                self.use_textures = False
        
        # Episode frames buffer
        self.frames = []
        self.recording = False
        
    def start_recording(self):
        """Start recording a new episode"""
        self.frames = []
        self.recording = True
        
    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        
    def _unwrap_state(self, env_state):
        """
        Unwrap environment state if it's wrapped in LogEnvState or similar
        
        Args:
            env_state: Potentially wrapped environment state
            
        Returns:
            The actual Craftax environment state
        """
        # Check if state is wrapped (e.g., LogEnvState)
        if hasattr(env_state, 'env_state'):
            return env_state.env_state
        return env_state
    
    def add_frame(self, env_state, observation):
        """
        Add a frame to the current episode recording
        
        Args:
            env_state: Current environment state (may be wrapped)
            observation: Current observation (agent's view)
        """
        if not self.recording:
            return
            
        try:
            frame = self._render_combined_frame(env_state, observation)
            # For MiniGrid, accept native resolution (typically 416x416 for 13x13 grid with tile_size=32)
            # For other environments, expect configured frame_size
            if frame is not None:
                expected_shape = (self.frame_size[1], self.frame_size[0], 3)
                # Accept any resolution for MiniGrid since native render determines size
                if self.is_minigrid or frame.shape == expected_shape:
                    self.frames.append(frame)
                else:
                    if not hasattr(self, '_frame_shape_error_shown'):
                        print(f"⚠️  Frame has unexpected shape: {frame.shape}, expected {expected_shape}")
                        self._frame_shape_error_shown = True
        except Exception as e:
            if not hasattr(self, '_render_error_shown'):
                print(f"⚠️  Failed to render frame: {e}")
                import traceback
                traceback.print_exc()
                self._render_error_shown = True
    
    def _render_combined_frame(self, env_state, observation) -> np.ndarray:
        """
        Render a combined frame showing full map with FOV overlay, player, and mobs
        
        Returns:
            numpy array of shape (height, width, 3) with RGB values
        """
        # Unwrap state if needed
        env_state = self._unwrap_state(env_state)
        
        # MiniGrid-specific path: render from observation/state if possible
        if self.is_minigrid:
            # For MiniGrid, use native resolution directly without resizing
            # to avoid interpolation artifacts
            width, height = None, None  # Will be determined by native render
        else:
            # Prepare canvas size for other environments
            width, height = self.frame_size
        
        # Create canvas only for non-MiniGrid environments
        if not self.is_minigrid:
            frame = np.ones((height, width, 3), dtype=np.uint8) * 32  # Dark gray background

        # MiniGrid-specific path: render from observation/state if possible
        if self.is_minigrid:
            try:
                # Prefer the environment's render() for a faithful MiniGrid view
                full_rgb = None
                try:
                    # Try JAX MiniGrid render with state parameter first
                    # Include highlight=True to show agent's field of view
                    full_rgb = self.env.render(env_state, mode='rgb_array', tile_size=32, highlight=True)
                    if not hasattr(self, '_render_debug_shown'):
                        print("✅ Using env.render(state, mode='rgb_array', tile_size=32, highlight=True) - native MiniGrid rendering with FOV")
                        self._render_debug_shown = True
                except Exception as e:
                    if not hasattr(self, '_render_debug_shown'):
                        print(f"⚠️  env.render(state, ...) failed: {e}")
                        self._render_debug_shown = True
                    try:
                        # Common Gym API: env.render(mode='rgb_array')
                        full_rgb = self.env.render(mode='rgb_array')
                    except Exception:
                        try:
                            # Some environments accept the mode as a positional arg
                            full_rgb = self.env.render('rgb_array')
                        except Exception:
                            try:
                                # Fallback: call render without args (may return array)
                                full_rgb = self.env.render()
                            except Exception:
                                full_rgb = None

                # If direct render failed, try common wrappers/unwrapped envs
                if full_rgb is None:
                    # Breadth-first search for an inner env object that implements render
                    try:
                        seen = set()
                        queue = [self.env]
                        while queue and full_rgb is None:
                            candidate = queue.pop(0)
                            cid = id(candidate)
                            if cid in seen:
                                continue
                            seen.add(cid)
                            # If candidate has render, try it with a few common signatures
                            if hasattr(candidate, 'render') and callable(getattr(candidate, 'render')):
                                module_name = getattr(candidate.__class__, '__module__', '') or ''
                                # Try a variety of common signatures for render
                                for kwargs in ({'mode': 'rgb_array'}, {'mode': 'rgb_array', 'tile_size': 32}, {}, {'mode': 'rgb_array', 'highlight': False}):
                                    try:
                                        candidate_result = candidate.render(**kwargs) if kwargs else candidate.render()
                                        if candidate_result is None:
                                            continue
                                        # Convert to numpy for basic validation
                                        cand_img = np.array(candidate_result)
                                        if cand_img.ndim == 3 and cand_img.shape[2] >= 3:
                                            # If this candidate comes from Craftax, skip when recording MiniGrid
                                            if self.is_minigrid and 'craftax' in module_name.lower():
                                                # prefer non-Craftax renders for MiniGrid fidelity
                                                continue
                                            full_rgb = candidate_result
                                            break
                                    except Exception:
                                        continue
                                if full_rgb is not None:
                                    break
                            # Enqueue common wrappers/containers to continue searching
                            for attr in ('envs', 'env', 'unwrapped', 'venv'):
                                if hasattr(candidate, attr):
                                    try:
                                        sub = getattr(candidate, attr)
                                        # If it's a list/tuple of envs, extend queue
                                        if isinstance(sub, (list, tuple)):
                                            queue.extend(sub)
                                        else:
                                            queue.append(sub)
                                    except Exception:
                                        continue
                    except Exception:
                        full_rgb = None

                if full_rgb is not None:
                    img = np.array(full_rgb)
                    # Normalize float images to uint8
                    if img.dtype.kind == 'f':
                        img = np.clip((img * 255.0).astype(np.uint8), 0, 255)
                    # Ensure 3 channels
                    if img.ndim == 3 and img.shape[2] >= 3:
                        # For MiniGrid, use native resolution without resize to avoid interpolation artifacts
                        # MiniGrid renders at grid_size * tile_size (e.g., 13x13 grid * 32px = 416x416)
                        # Resizing creates visual artifacts that make tiles appear "doubled"
                        if self.is_minigrid:
                            # Return the native render directly - no canvas, no resize
                            return img[:, :, :3]
                        else:
                            # For other environments, resize and center as before
                            rgb_resized = self._resize_image(img[:, :, :3], (width - 40, height - 40))

                        # For MiniGrid runs, apply conservative recoloring to
                        # better match the canonical MiniGrid palette seen in
                        # integration samples (beige floor, grey walls, green
                        # goals) and make locked doors visible.
                        if self.is_minigrid:
                            try:
                                # Recolor pale blue floor to beige
                                floor_color = np.array((234, 231, 164), dtype=np.uint8)
                                r = rgb_resized[:, :, 0].astype(int)
                                g = rgb_resized[:, :, 1].astype(int)
                                b = rgb_resized[:, :, 2].astype(int)
                                blue_dom = (b > r + 20) & (b > g + 20) & (b > 160)
                                blue_dom = blue_dom | ((b > 150) & (np.abs(r - g) < 30) & (b - np.maximum(r, g) > 40))
                                if blue_dom.any():
                                    rgb_resized[blue_dom] = floor_color

                                # Tile-based recolors (walls/goals/doors) using
                                # FourRooms layout when available to preserve
                                # approximate positions.
                                try:
                                    # Skip tile-based recolors when source render is
                                    # very small - it's likely a localized view.
                                    if img.shape[0] < 200 or img.shape[1] < 200:
                                        raise RuntimeError('source render too small for tile mapping')
                                    from envs.minigrid.minigrid_observation_generator import create_four_rooms_layout

                                    base = create_four_rooms_layout()
                                    map_h = int(base.shape[0])
                                    map_w = int(base.shape[1])
                                    tile_h = max(1, rgb_resized.shape[0] // map_h)
                                    tile_w = max(1, rgb_resized.shape[1] // map_w)

                                    def draw_tile(y, x, color):
                                        y0 = y * tile_h
                                        x0 = x * tile_w
                                        y1 = min(rgb_resized.shape[0], y0 + tile_h)
                                        x1 = min(rgb_resized.shape[1], x0 + tile_w)
                                        rgb_resized[y0:y1, x0:x1] = color

                                    # Key presence and color
                                    key_present = False
                                    key_color = np.array((255, 215, 0), dtype=np.uint8)
                                    if isinstance(env_state, dict) and env_state.get('key_pos') is not None:
                                        try:
                                            kx, ky = map(int, env_state.get('key_pos'))
                                            key_present = (kx >= 0 and 0 <= ky < map_h and 0 <= kx < map_w)
                                        except Exception:
                                            key_present = False

                                    # Walls -> grey
                                    wall_color = np.array((100, 100, 100), dtype=np.uint8)
                                    for yy in range(map_h):
                                        for xx in range(map_w):
                                            if int(base[yy, xx]) == 2:
                                                draw_tile(yy, xx, wall_color)

                                    # Goals
                                    try:
                                        g1 = env_state.get('goal1_pos') if isinstance(env_state, dict) else None
                                        g2 = env_state.get('goal2_pos') if isinstance(env_state, dict) else None
                                        if g1 is None:
                                            g1 = (1, 11)
                                        if g2 is None:
                                            g2 = (11, 11)
                                        if g1 is not None:
                                            gx, gy = map(int, g1)
                                            if 0 <= gy < map_h and 0 <= gx < map_w:
                                                draw_tile(gy, gx, np.array((80, 200, 120), dtype=np.uint8))
                                        if g2 is not None:
                                            gx, gy = map(int, g2)
                                            if 0 <= gy < map_h and 0 <= gx < map_w:
                                                draw_tile(gy, gx, np.array((15, 255, 80), dtype=np.uint8))
                                    except Exception:
                                        pass

                                    # Doors -> key color (when key present), draw knob if locked
                                    try:
                                        door_unlocked = None
                                        if isinstance(env_state, dict) and 'door_unlocked' in env_state:
                                            try:
                                                door_unlocked = bool(env_state.get('door_unlocked'))
                                            except Exception:
                                                door_unlocked = None

                                        for yy in range(map_h):
                                            for xx in range(map_w):
                                                if int(base[yy, xx]) == 4:
                                                    if key_present:
                                                        draw_tile(yy, xx, key_color)
                                                    else:
                                                        draw_tile(yy, xx, np.array((160, 120, 60), dtype=np.uint8))
                                                    if door_unlocked is not None and door_unlocked is False:
                                                        cy = yy * tile_h + tile_h // 2
                                                        cx = xx * tile_w + tile_w // 2
                                                        rr = max(1, tile_h // 6)
                                                        yy_idx, xx_idx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
                                                        km = (xx_idx * xx_idx + yy_idx * yy_idx) <= (rr * rr)
                                                        y0 = max(0, cy - rr)
                                                        x0 = max(0, cx - rr)
                                                        hsub = km.shape[0]
                                                        wsub = km.shape[1]
                                                        sub = rgb_resized[y0:y0 + hsub, x0:x0 + wsub]
                                                        if sub.shape[0] == hsub and sub.shape[1] == wsub:
                                                            if key_present:
                                                                knob_color = (key_color.astype(np.float32) * np.array([0.2, 0.2, 0.15], dtype=np.float32)).astype(np.uint8)
                                                            else:
                                                                knob_color = np.array([80, 50, 20], dtype=np.uint8)
                                                            sub[km] = knob_color
                                                            rgb_resized[y0:y0 + hsub, x0:x0 + wsub] = sub
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        top = (height - rgb_resized.shape[0]) // 2  
                        left = (width - rgb_resized.shape[1]) // 2
                        frame[top:top + rgb_resized.shape[0], left:left + rgb_resized.shape[1]] = rgb_resized

                        # DO NOT apply additional FOV overlay for MiniGrid environments.
                        # The native MiniGrid env.render() already includes FOV highlighting
                        # via the highlight_mask parameter. Adding another overlay causes
                        # misalignment between the agent direction and highlighted FOV area
                        # because Craftax and MiniGrid use different FOV calculation systems.

                        if not hasattr(self, '_minigrid_render_used_shown'):
                            print("ℹ️  MiniGrid env.render() used for frame rendering")
                            self._minigrid_render_used_shown = True
                        return frame

                # Prefer environment-provided observation generator
                obs_for_render = None
                if hasattr(self.env, 'get_obs'):
                    try:
                        obs_for_render = self.env.get_obs(env_state)
                    except Exception:
                        obs_for_render = None
                elif observation is not None:
                    obs_for_render = observation
                # Prefer rendering a top-down full MiniGrid map and overlaying
                # the agent and FOV on top. This avoids returning a centered
                # FOV-only view (agent in center) which looks like the
                # original Craftax local view.
                try:
                    # Build a minimal state dict from env_state so our map
                    # renderer can handle both dict and namedtuple-like states.
                    state_dict = None
                    if isinstance(env_state, dict):
                        state_dict = env_state
                    else:
                        state_dict = {}
                        # Copy common attribute names used by Minigrid/Craftax
                        for key in ('agent_pos', 'agent_dir', 'player_position', 'player_direction', 'key_pos', 'map'):
                            if hasattr(env_state, key):
                                state_dict[key] = getattr(env_state, key)

                    # Only use state-based full-map rendering when we have at
                    # least an agent position (otherwise fall back to FOV-only)
                    if state_dict and ('agent_pos' in state_dict or 'player_position' in state_dict or 'map' in state_dict):
                        mini_img = self._render_minigrid_map_from_state(state_dict)
                        mini_resized = self._resize_image(mini_img, (width - 40, height - 40))
                        top = (height - mini_resized.shape[0]) // 2
                        left = (width - mini_resized.shape[1]) // 2
                        frame[top:top + mini_resized.shape[0], left:left + mini_resized.shape[1]] = mini_resized

                        # If the map image was produced by the env's native grid
                        # renderer or by the MiniGrid helper module, prefer it as
                        # authoritative and skip our custom overlays.
                        if getattr(self, '_minigrid_map_source', None) in ('grid', 'minigrid_module'):
                            # If the authoritative renderer was used, we still
                            # apply the agent FOV highlight (semi-transparent)
                            # according to the environment's view configuration
                            # (e.g., ViewSize-3x3). We avoid drawing the agent
                            # triangle or other custom overlays on top to keep
                            # visuals faithful.
                            try:
                                if self.record_fov:
                                    # Build a lightweight overlay state from state_dict
                                    from types import SimpleNamespace
                                    ap = state_dict.get('agent_pos') or state_dict.get('player_position')
                                    ad = state_dict.get('agent_dir') or state_dict.get('player_direction') or 0
                                    if ap is not None:
                                        # _overlay_fov_highlight expects player_position
                                        # as (row, col) -> (y, x)
                                        try:
                                            py = int(ap[1]) if len(ap) == 2 else int(ap[0])
                                            px = int(ap[0]) if len(ap) == 2 else 0
                                        except Exception:
                                            py, px = 0, 0
                                        overlay_state = SimpleNamespace(player_position=(py, px), player_direction=ad)

                                        # Determine view_size: prefer explicit in state, else parse from env_name
                                        view_size = None
                                        if 'view_size' in state_dict:
                                            try:
                                                view_size = int(state_dict.get('view_size'))
                                            except Exception:
                                                view_size = None
                                        if view_size is None:
                                            import re
                                            m = re.search(r'ViewSize-(\d+)x(\d+)', self.env_name)
                                            if m:
                                                try:
                                                    view_size = int(m.group(1))
                                                except Exception:
                                                    view_size = 3
                                            else:
                                                view_size = 9

                                        # Compute map tile size (reverse engineering of mini_resized)
                                        if hasattr(self, '_minigrid_map_grid_shape'):
                                            map_h, map_w = self._minigrid_map_grid_shape
                                        else:
                                            map_h, map_w = mini_img.shape[0] // 24, mini_img.shape[1] // 24
                                        resized_height, resized_width = mini_resized.shape[:2]
                                        scale_y = resized_height / max(map_h, 1)
                                        scale_x = resized_width / max(map_w, 1)

                                        self._overlay_fov_highlight(frame, overlay_state, (map_h, map_w), (top, left), scale_x, scale_y, view_size=view_size)
                            except Exception:
                                pass

                            if not hasattr(self, '_minigrid_map_used_shown'):
                                print("ℹ️  MiniGrid state-based map (env/grid/module) used for frame rendering")
                                self._minigrid_map_used_shown = True
                            return frame

                        # Overlay FOV highlight (use env_state when possible)
                        try:
                            # For overlay helpers we need an object with attributes
                            overlay_state = env_state
                            # If env_state is a dict, create a lightweight object
                            if isinstance(env_state, dict):
                                from types import SimpleNamespace
                                overlay_state = SimpleNamespace(**{k: v for k, v in state_dict.items() if v is not None})

                            # compute map tile size and offsets used by overlays
                            if 'map' in state_dict:
                                world_map = np.array(state_dict.get('map'))
                                if world_map.ndim == 3:
                                    world_map = world_map[0]
                                map_h, map_w = world_map.shape
                            elif hasattr(self, '_minigrid_map_grid_shape'):
                                map_h, map_w = self._minigrid_map_grid_shape
                            else:
                                # assume default 13x13 mini layout (tile size 24)
                                map_h, map_w = mini_img.shape[0] // 24, mini_img.shape[1] // 24

                            resized_height, resized_width = mini_resized.shape[:2]
                            scale_y = resized_height / max(map_h, 1)
                            scale_x = resized_width / max(map_w, 1)

                            # Overlay FOV highlight and an agent marker
                            if self.record_fov:
                                # The overlay helper expects the map to be at offset (20,20)
                                self._overlay_fov_highlight(frame, overlay_state, (map_h, map_w), (top, left), scale_x, scale_y)
                        except Exception:
                            pass

                        if not hasattr(self, '_minigrid_map_used_shown'):
                            print("ℹ️  MiniGrid state-based map used for frame rendering")
                            self._minigrid_map_used_shown = True
                        return frame

                except Exception:
                    # If any of the map-rendering steps failed, fall back to
                    # older behavior (FOV-only rendering based on obs)
                    pass

                # Render FOV-style image from observation as a last resort
                if self.is_minigrid:
                    # Use a MiniGrid-specific FOV renderer when possible to
                    # avoid falling back to Craftax color mappings.
                    fov_img = self._render_minigrid_fov_from_obs(obs_for_render)
                else:
                    fov_img = self._render_fov_simple(obs_for_render)
                fov_resized = self._resize_image(fov_img, (width - 40, height - 40))
                # Center the FOV image on canvas
                top = (height - fov_resized.shape[0]) // 2
                left = (width - fov_resized.shape[1]) // 2
                frame[top:top + fov_resized.shape[0], left:left + fov_resized.shape[1]] = fov_resized
                return frame
            except Exception as e:
                if not hasattr(self, '_minigrid_render_error_shown'):
                    print(f"⚠️  MiniGrid render failed, falling back to default rendering: {e}")
                    self._minigrid_render_error_shown = True
                # Try to render a simple top-down map from the recorded state dict
                try:
                    if isinstance(env_state, dict) and 'agent_pos' in env_state:
                        mini_img = self._render_minigrid_map_from_state(env_state)
                        mini_resized = self._resize_image(mini_img, (width - 40, height - 40))
                        top = (height - mini_resized.shape[0]) // 2
                        left = (width - mini_resized.shape[1]) // 2
                        frame[top:top + mini_resized.shape[0], left:left + mini_resized.shape[1]] = mini_resized
                        if not hasattr(self, '_minigrid_map_used_shown'):
                            print("ℹ️  MiniGrid state-based map used for frame rendering")
                            self._minigrid_map_used_shown = True
                        return frame
                except Exception:
                    pass
        
        width, height = self.frame_size
        
        # Create canvas
        frame = np.ones((height, width, 3), dtype=np.uint8) * 32  # Dark gray background
        
        # Render full map (terrain only)
        full_map = self._render_full_map(env_state)
        full_map_resized = self._resize_image(full_map, (width - 40, height - 40))
        
        # Place full map on canvas
        frame[20:20 + full_map_resized.shape[0], 20:20 + full_map_resized.shape[1]] = full_map_resized
        
        # Get actual map size in tiles (not pixels)
        if hasattr(env_state, 'map'):
            world_map = np.array(env_state.map)
            if world_map.ndim == 3:
                world_map = world_map[0]
            map_height_tiles, map_width_tiles = world_map.shape
        else:
            # Fallback default
            map_height_tiles, map_width_tiles = 32, 32
        
        # Calculate scaling factor for overlays (from tiles to pixels on canvas)
        resized_height, resized_width = full_map_resized.shape[:2]
        scale_y = resized_height / map_height_tiles  # pixels per tile
        scale_x = resized_width / map_width_tiles    # pixels per tile
        
        # Overlay FOV highlight if enabled (pass map size in tiles)
        if self.record_fov:
            self._overlay_fov_highlight(frame, env_state, (map_height_tiles, map_width_tiles), 
                                       (20, 20), scale_x, scale_y)
        
        # Overlay player with texture
        self._overlay_player_on_frame(frame, env_state, (20, 20), scale_x, scale_y)
        
        # Overlay mobs with textures
        self._overlay_mobs_on_frame(frame, env_state, (20, 20), scale_x, scale_y)
        
        return frame
    
    def _overlay_fov_highlight(self, frame: np.ndarray, env_state, 
                               map_size_tiles: Tuple[int, int], offset: Tuple[int, int],
                               scale_x: float, scale_y: float, view_size: int = 9):
        """
        Overlay FOV highlight on the full map (semi-transparent white overlay)
        Matches view_full_environment.py implementation
        
        Args:
            frame: The frame to draw on (modified in place)
            env_state: Environment state
            map_size_tiles: Map size in tiles (height, width)
            offset: Offset (y, x) of the map on the canvas
            scale_x: Pixels per tile in x direction
            scale_y: Pixels per tile in y direction
        """
        # Get agent position
        if not hasattr(env_state, 'player_position'):
            return
        
        player_pos = np.array(env_state.player_position)
        # Handle vectorized environments (batch dimension)
        if player_pos.ndim == 2:
            player_pos = player_pos[0]
        py, px = int(player_pos[0]), int(player_pos[1])
        
        # Get player direction for FOV calculation
        player_dir = np.array(env_state.player_direction) if hasattr(env_state, 'player_direction') else 4
        if hasattr(player_dir, 'ndim') and player_dir.ndim > 0:
            player_dir = player_dir[0] if player_dir.shape[0] > 0 else 4
        direction = int(player_dir)
        
        # FOV dimensions
        obs_height = 9
        obs_width = 9

        # Calculate agent offset based on direction (same logic as renderer)
        if direction == 1:  # LEFT
            agent_row_offset, agent_col_offset = 4, 8
        elif direction == 2:  # RIGHT
            agent_row_offset, agent_col_offset = 4, 0
        elif direction == 3:  # UP
            agent_row_offset, agent_col_offset = 8, 4
        else:  # DOWN
            agent_row_offset, agent_col_offset = 0, 4

        # Calculate FOV bounds in tile coordinates
        map_height_tiles, map_width_tiles = map_size_tiles

        # Calculate the intended FOV bounds before clamping
        intended_fov_top_row = py - agent_row_offset
        intended_fov_left_col = px - agent_col_offset

        # Calculate actual FOV bounds (what is visible) after clamping to map bounds
        actual_fov_top_row = max(0, intended_fov_top_row)
        actual_fov_left_col = max(0, intended_fov_left_col)
        actual_fov_bottom_row = min(
            map_height_tiles, intended_fov_top_row + obs_height)
        actual_fov_right_col = min(
            map_width_tiles, intended_fov_left_col + obs_width)

        # Calculate the actual dimensions of the visible FOV (in tiles)
        actual_fov_height = actual_fov_bottom_row - actual_fov_top_row
        actual_fov_width = actual_fov_right_col - actual_fov_left_col

        # Only draw the overlay if there's an actual visible area
        if actual_fov_height > 0 and actual_fov_width > 0:
            # Convert tile coordinates to pixel coordinates on canvas
            canvas_top = int(actual_fov_top_row * scale_y) + offset[0]
            canvas_left = int(actual_fov_left_col * scale_x) + offset[1]
            canvas_bottom = int(actual_fov_bottom_row * scale_y) + offset[0]
            canvas_right = int(actual_fov_right_col * scale_x) + offset[1]

            # Clamp to frame bounds
            canvas_top = max(0, min(canvas_top, frame.shape[0]))
            canvas_bottom = max(0, min(canvas_bottom, frame.shape[0]))
            canvas_left = max(0, min(canvas_left, frame.shape[1]))
            canvas_right = max(0, min(canvas_right, frame.shape[1]))

            # Apply semi-transparent white overlay (alpha blending)
            if canvas_bottom > canvas_top and canvas_right > canvas_left:
                alpha = 0.31  # ~80/255 for subtle highlight
                overlay_color = np.array([255, 255, 255], dtype=np.float32)

                # Blend the overlay with existing pixels
                fov_region = frame[canvas_top:canvas_bottom,
                                   canvas_left:canvas_right].astype(np.float32)
                fov_region = (1 - alpha) * fov_region + alpha * overlay_color
                frame[canvas_top:canvas_bottom, canvas_left:canvas_right] = np.clip(
                    fov_region, 0, 255).astype(np.uint8)

    def _overlay_player_on_frame(self, frame: np.ndarray, env_state, offset: Tuple[int, int], 
                                  scale_x: float, scale_y: float):
        """
        Overlay player with authentic Craftax texture on the frame
        
        Args:
            frame: The frame to draw on (modified in place)
            env_state: Environment state
            offset: Offset (y, x) of the map on the canvas
            scale_x: X scaling factor from original to resized
            scale_y: Y scaling factor from original to resized
        """
        if not self.use_textures or not hasattr(env_state, 'player_position'):
            return
        
        try:
            # Import textures
            if "Classic" in self.env_name:
                from envs.craftax.craftax_classic.constants import TEXTURES, BLOCK_PIXEL_SIZE_IMG
            else:
                from envs.craftax.craftax.constants import TEXTURES, BLOCK_PIXEL_SIZE_IMG
            
            block_pixel_size = BLOCK_PIXEL_SIZE_IMG  # 16
            texture_dict = TEXTURES[block_pixel_size]
            player_textures = texture_dict['player_textures']
            
            # Get player position
            player_pos = np.array(env_state.player_position)
            if player_pos.ndim == 2:
                player_pos = player_pos[0]
            py, px = int(player_pos[0]), int(player_pos[1])
            
            # Get player direction
            player_dir = np.array(env_state.player_direction) if hasattr(env_state, 'player_direction') else 4
            if hasattr(player_dir, 'ndim') and player_dir.ndim > 0:
                player_dir = player_dir[0] if player_dir.shape[0] > 0 else 4
            direction = int(player_dir)
            
            # Check if sleeping
            is_sleeping = getattr(env_state, 'is_sleeping', False)
            
            # Select texture based on state
            if is_sleeping:
                texture_index = 4  # Sleep texture
            else:
                texture_index = direction - 1  # Convert direction to texture index
            
            # Clamp to valid range
            texture_index = max(0, min(texture_index, len(player_textures) - 1))
            
            if 0 <= texture_index < len(player_textures):
                # Get player texture
                player_texture = np.array(player_textures[texture_index]).astype(np.float32)
                
                # Apply lighting effects (same as terrain) - only to RGB channels
                light_level = float(getattr(env_state, 'light_level', 1.0))
                
                if light_level < 1.0:
                    dimming_factor = 0.5 + 0.5 * light_level
                    player_texture[:, :, :3] *= dimming_factor  # Apply to RGB channels only
                
                if is_sleeping:
                    # Apply grayscale sleeping effect
                    luminance = (
                        0.299 * player_texture[:, :, 0] +
                        0.587 * player_texture[:, :, 1] + 
                        0.114 * player_texture[:, :, 2]
                    )
                    
                    player_texture[:, :, 0] = 0.5 * luminance  # Red channel
                    player_texture[:, :, 1] = 0.5 * luminance  # Green channel
                    player_texture[:, :, 2] = 0.5 * luminance + 0.5 * 16  # Blue channel with blue tint
                
                # Clamp and convert
                player_texture = np.clip(player_texture, 0, 255).astype(np.uint8)
                
                # Handle RGBA vs RGB: extract only RGB channels if texture has alpha
                if player_texture.shape[2] == 4:
                    # Has alpha channel - use it for transparency
                    player_rgb = player_texture[:, :, :3]
                    player_alpha = player_texture[:, :, 3:4] / 255.0  # Normalize to [0, 1]
                else:
                    # No alpha channel - use as is
                    player_rgb = player_texture
                    player_alpha = np.ones((player_texture.shape[0], player_texture.shape[1], 1))
                
                # Calculate position on frame (tile coordinates to pixel coordinates)
                player_y_scaled = int(py * scale_y) + offset[0]
                player_x_scaled = int(px * scale_x) + offset[1]
                
                # Calculate size on frame (one tile in pixels)
                player_h_scaled = int(scale_y)
                player_w_scaled = int(scale_x)
                
                # Resize player texture to match frame scale
                from PIL import Image
                player_img = Image.fromarray(player_rgb)
                player_img_resized = player_img.resize((player_w_scaled, player_h_scaled), Image.BILINEAR)
                player_rgb_resized = np.array(player_img_resized)
                
                # Resize alpha channel separately
                alpha_img = Image.fromarray((player_alpha[:, :, 0] * 255).astype(np.uint8))
                alpha_img_resized = alpha_img.resize((player_w_scaled, player_h_scaled), Image.BILINEAR)
                player_alpha_resized = np.array(alpha_img_resized).astype(np.float32) / 255.0
                
                # Blit player onto frame with alpha blending (with bounds checking)
                y_end = min(player_y_scaled + player_h_scaled, frame.shape[0])
                x_end = min(player_x_scaled + player_w_scaled, frame.shape[1])
                h_actual = y_end - player_y_scaled
                w_actual = x_end - player_x_scaled
                
                if h_actual > 0 and w_actual > 0:
                    # Alpha blend: result = alpha * foreground + (1 - alpha) * background
                    player_rgb_slice = player_rgb_resized[:h_actual, :w_actual]
                    player_alpha_slice = player_alpha_resized[:h_actual, :w_actual, np.newaxis]
                    background = frame[player_y_scaled:y_end, player_x_scaled:x_end].astype(np.float32)
                    
                    blended = player_alpha_slice * player_rgb_slice + (1 - player_alpha_slice) * background
                    frame[player_y_scaled:y_end, player_x_scaled:x_end] = np.clip(blended, 0, 255).astype(np.uint8)
                    
        except Exception as e:
            if not hasattr(self, '_player_overlay_error_shown'):
                print(f"⚠️  Player overlay failed: {e}")
                self._player_overlay_error_shown = True
    
    def _overlay_mobs_on_frame(self, frame: np.ndarray, env_state, offset: Tuple[int, int],
                                scale_x: float, scale_y: float):
        """
        Overlay mobs with authentic Craftax textures on the frame
        
        Args:
            frame: The frame to draw on (modified in place)
            env_state: Environment state
            offset: Offset (y, x) of the map on the canvas
            scale_x: X scaling factor from original to resized
            scale_y: Y scaling factor from original to resized
        """
        if not self.use_textures:
            return
        
        try:
            # Import textures
            if "Classic" in self.env_name:
                from envs.craftax.craftax_classic.constants import TEXTURES, BLOCK_PIXEL_SIZE_IMG
            else:
                from envs.craftax.craftax.constants import TEXTURES, BLOCK_PIXEL_SIZE_IMG
            
            block_pixel_size = BLOCK_PIXEL_SIZE_IMG  # 16
            texture_dict = TEXTURES[block_pixel_size]
            
            # Draw each mob type
            for mob_name, attr in [('zombie', 'zombies'), ('cow', 'cows'), ('skeleton', 'skeletons'), ('arrow', 'arrows')]:
                mob = getattr(env_state, attr, None)
                if mob is not None and hasattr(mob, 'position') and hasattr(mob, 'mask'):
                    positions = mob.position
                    mask = mob.mask
                    
                    # Get mob texture
                    texture_key = f'{mob_name}_texture'
                    if texture_key not in texture_dict:
                        continue
                    
                    mob_texture = np.array(texture_dict[texture_key]).astype(np.uint8)
                    
                    # Draw each mob instance
                    for i, pos in enumerate(positions):
                        if mask[i]:
                            row, col = int(pos[0]), int(pos[1])
                            
                            # Calculate position on frame (tile coordinates to pixel coordinates)
                            mob_y_scaled = int(row * scale_y) + offset[0]
                            mob_x_scaled = int(col * scale_x) + offset[1]
                            
                            # Calculate size on frame (one tile in pixels)
                            mob_h_scaled = int(scale_y)
                            mob_w_scaled = int(scale_x)
                            
                            # Rotate arrow textures according to arrow_directions when available
                            from PIL import Image
                            mob_img = Image.fromarray(mob_texture)
                            if mob_name == 'arrow' and hasattr(env_state, 'arrow_directions'):
                                try:
                                    dr, dc = int(env_state.arrow_directions[i][0]), int(env_state.arrow_directions[i][1])
                                except Exception:
                                    dr, dc = 0, 0
                                # Map direction vector to rotation angle (PIL rotate is CCW)
                                if dr < 0:
                                    angle = 0  # up (base texture assumed up)
                                elif dr > 0:
                                    angle = 180  # down
                                elif dc < 0:
                                    angle = 90  # left
                                elif dc > 0:
                                    angle = -90  # right (clockwise 90)
                                else:
                                    angle = 0
                                # Rotate without expanding canvas to keep texture size
                                mob_img = mob_img.rotate(angle, expand=False)
                            # Resize mob texture to match frame scale
                            mob_img_resized = mob_img.resize((mob_w_scaled, mob_h_scaled), Image.BILINEAR)
                            mob_texture_resized = np.array(mob_img_resized)
                            
                            # Blit mob onto frame (with bounds checking and transparency)
                            y_end = min(mob_y_scaled + mob_h_scaled, frame.shape[0])
                            x_end = min(mob_x_scaled + mob_w_scaled, frame.shape[1])
                            h_actual = y_end - mob_y_scaled
                            w_actual = x_end - mob_x_scaled
                            
                            if h_actual > 0 and w_actual > 0:
                                # Check if texture has transparency (black = transparent)
                                mob_slice = mob_texture_resized[:h_actual, :w_actual]
                                # Only draw non-black pixels
                                mask_non_black = (mob_slice[:, :, 0] > 10) | (mob_slice[:, :, 1] > 10) | (mob_slice[:, :, 2] > 10)
                                frame[mob_y_scaled:y_end, mob_x_scaled:x_end][mask_non_black] = mob_slice[mask_non_black]
                    
        except Exception as e:
            if not hasattr(self, '_mob_overlay_error_shown'):
                print(f"⚠️  Mob overlay failed: {e}")
                self._mob_overlay_error_shown = True
    
    def _render_full_map(self, env_state) -> np.ndarray:
        """Render the full environment map with textures"""
        if self.use_textures and self.render_function is not None:
            try:
                # Use Craftax's official renderer with textures
                # This renders the full map with textures block by block
                return self._render_full_map_with_textures(env_state)
                
            except Exception as e:
                # Fall back to simple rendering on error
                if not hasattr(self, '_texture_error_shown'):
                    print(f"⚠️  Texture rendering failed, using fallback: {e}")
                    import traceback
                    traceback.print_exc()
                    self._texture_error_shown = True
                return self._render_full_map_simple(env_state)
        else:
            return self._render_full_map_simple(env_state)
    
    def _render_full_map_with_textures(self, env_state) -> np.ndarray:
        """
        Render the full map using authentic Craftax textures
        
        This method uses the same texture processing as render_craftax_pixels
        to render the full map with proper game textures.
        """
        try:
            # Import textures from Craftax
            if "Classic" in self.env_name:
                from envs.craftax.craftax_classic.constants import TEXTURES, BLOCK_PIXEL_SIZE_IMG
            else:
                from envs.craftax.craftax.constants import TEXTURES, BLOCK_PIXEL_SIZE_IMG
            
            # Use block pixel size for texture rendering (16 is available in TEXTURES)
            block_pixel_size = BLOCK_PIXEL_SIZE_IMG  # 16
            texture_dict = TEXTURES[block_pixel_size]
            block_textures = texture_dict['block_textures']
            
            # Get map from state
            if hasattr(env_state, 'map'):
                world_map = np.array(env_state.map)
                # Handle vectorized environments (batch dimension)
                if world_map.ndim == 3:
                    world_map = world_map[0]
            else:
                # Fallback if no map available
                return self._render_full_map_simple(env_state)
            
            map_height, map_width = world_map.shape
            full_map_pixels = np.zeros((map_height * block_pixel_size, map_width * block_pixel_size, 3), dtype=np.uint8)
            
            # Render each block using textures
            for y in range(map_height):
                for x in range(map_width):
                    block_type = int(world_map[y, x])
                    
                    # Use texture if available
                    if 0 <= block_type < len(block_textures):
                        block_texture = np.array(block_textures[block_type])
                        
                        y_start = y * block_pixel_size
                        y_end = (y + 1) * block_pixel_size
                        x_start = x * block_pixel_size
                        x_end = (x + 1) * block_pixel_size
                        
                        full_map_pixels[y_start:y_end, x_start:x_end] = block_texture
            
            # Apply lighting effects if present
            light_level = float(getattr(env_state, 'light_level', 1.0))
            is_sleeping = getattr(env_state, 'is_sleeping', False)
            
            if light_level < 1.0:
                dimming_factor = 0.5 + 0.5 * light_level
                full_map_pixels = (full_map_pixels.astype(np.float32) * dimming_factor).astype(np.uint8)
            
            if is_sleeping:
                # Apply grayscale sleeping effect
                full_map_pixels_float = full_map_pixels.astype(np.float32)
                luminance_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
                sleep_pixels = np.dot(full_map_pixels_float, luminance_weights)
                
                full_map_pixels = np.zeros_like(full_map_pixels_float)
                full_map_pixels[:, :, 0] = 0.5 * sleep_pixels
                full_map_pixels[:, :, 1] = 0.5 * sleep_pixels
                full_map_pixels[:, :, 2] = 0.5 * sleep_pixels + 0.5 * 16  # Blue channel with blue tint
                
                full_map_pixels = np.clip(full_map_pixels, 0, 255).astype(np.uint8)
            
            return full_map_pixels
            
        except Exception as e:
            if not hasattr(self, '_full_map_texture_error'):
                print(f"⚠️  Full map texture rendering failed: {e}")
                import traceback
                traceback.print_exc()
                self._full_map_texture_error = True
            return self._render_full_map_simple(env_state)
    
    def _render_full_map_simple(self, env_state) -> np.ndarray:
        """Simple color-based rendering of full map"""
        # Get map
        if hasattr(env_state, 'map'):
            world_map = np.array(env_state.map)
            # Handle vectorized environments (batch dimension)
            if world_map.ndim == 3:
                # Take first environment from batch [batch, height, width] -> [height, width]
                world_map = world_map[0]
            elif world_map.ndim != 2:
                # Unexpected shape, use empty map
                world_map = np.zeros((32, 32), dtype=np.int32)
        else:
            # Fallback: create empty map
            world_map = np.zeros((32, 32), dtype=np.int32)
        
        height, width = world_map.shape
        block_size = 8
        
        # Create RGB image
        img = np.zeros((height * block_size, width * block_size, 3), dtype=np.uint8)
        
        # Color mapping for different block types
        color_map = {
            0: (255, 0, 255),    # Invalid - magenta
            1: (0, 0, 0),        # Out of bounds - black
            2: (34, 139, 34),    # Grass - green
            3: (30, 144, 255),   # Water - blue
            4: (128, 128, 128),  # Stone - gray
            5: (0, 100, 0),      # Tree - dark green
            6: (139, 69, 19),    # Wood - brown
            7: (210, 180, 140),  # Path - tan
            8: (64, 64, 64),     # Coal - dark gray
            9: (192, 192, 192),  # Iron - light gray
            10: (185, 242, 255), # Diamond - cyan
            11: (160, 82, 45),   # Crafting table - brown
            12: (105, 105, 105), # Furnace - gray
            13: (244, 164, 96),  # Sand - sandy brown
            14: (255, 69, 0),    # Lava - red-orange
        }
        
        # Fill in blocks
        for y in range(height):
            for x in range(width):
                block_type = int(world_map[y, x])
                color = color_map.get(block_type, (200, 200, 200))  # Default gray
                img[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size] = color
        
        # Draw player
        if hasattr(env_state, 'player_position'):
            player_pos = np.array(env_state.player_position)
            # Handle vectorized environments (batch dimension)
            if player_pos.ndim == 2:
                # Take first environment from batch [batch, 2] -> [2]
                player_pos = player_pos[0]
            py, px = int(player_pos[0]), int(player_pos[1])
            if 0 <= py < height and 0 <= px < width:
                # Draw player as bright square
                img[py*block_size:(py+1)*block_size, px*block_size:(px+1)*block_size] = (200, 200, 255)
        
        return img

    def _render_minigrid_map_from_state(self, state_dict) -> np.ndarray:
        """Render a simple top-down MiniGrid full map from a serialized state dict."""
        # If the underlying env provides a Grid object with a native render
        # method (as in Farama MiniGrid), prefer that to get authentic
        # sprites and object placements. We try several common render
        # signatures and forward the agent position/dir when available.
        # If no grid.render is found, fall back to the module helper, then
        # finally to our internal drawing.
        # Breadth-first search through wrapper chain to find a Grid with render()
        seen = set()
        queue = [self.env]
        found = None
        while queue and found is None:
            candidate = queue.pop(0)
            cid = id(candidate)
            if cid in seen:
                continue
            seen.add(cid)
            grid = getattr(candidate, 'grid', None)
            if grid is not None and hasattr(grid, 'render'):
                found = grid
                break
            if hasattr(candidate, 'unwrapped'):
                try:
                    queue.append(getattr(candidate, 'unwrapped'))
                except Exception:
                    pass
            for attr in ('envs', 'env', 'venv', 'wrapped_env'):
                if hasattr(candidate, attr):
                    try:
                        sub = getattr(candidate, attr)
                        if isinstance(sub, (list, tuple)):
                            queue.extend(sub)
                        else:
                            queue.append(sub)
                    except Exception:
                        pass

        # If we found a Grid object try to render from it
        if found is not None:
            try:
                ag_pos = None
                ag_dir = None
                if isinstance(state_dict, dict):
                    ap = state_dict.get('agent_pos') or state_dict.get('player_position')
                    ad = state_dict.get('agent_dir') or state_dict.get('player_direction')
                    if ap is not None:
                        try:
                            ag_pos = (int(ap[0]), int(ap[1]) if len(ap) == 2 else int(ap[0]))
                        except Exception:
                            ag_pos = None
                    if ad is not None:
                        try:
                            ag_dir = int(ad)
                        except Exception:
                            ag_dir = None

                try:
                    if ag_pos is not None and ag_dir is not None:
                        img = found.render(tile_size=24, agent_pos=ag_pos, agent_dir=ag_dir)
                    elif ag_pos is not None:
                        img = found.render(tile_size=24, agent_pos=ag_pos)
                    else:
                        img = found.render(tile_size=24)
                except TypeError:
                    try:
                        if ag_pos is not None and ag_dir is not None:
                            img = found.render(24, ag_pos, ag_dir)
                        elif ag_pos is not None:
                            img = found.render(24, ag_pos)
                        else:
                            img = found.render(24)
                    except Exception:
                        img = found.render(24)

                img_arr = np.array(img)
                if img_arr.ndim == 3 and img_arr.shape[2] >= 3:
                    self._minigrid_map_source = 'grid'
                    try:
                        gw = getattr(found, 'width', None) or getattr(found, 'w', None) or getattr(found, 'grid_width', None)
                        gh = getattr(found, 'height', None) or getattr(found, 'h', None) or getattr(found, 'grid_height', None)
                        if gw is not None and gh is not None:
                            self._minigrid_map_grid_shape = (int(gh), int(gw))
                        else:
                            th = img_arr.shape[0] // 24 if img_arr.shape[0] >= 24 else img_arr.shape[0]
                            tw = img_arr.shape[1] // 24 if img_arr.shape[1] >= 24 else img_arr.shape[1]
                            self._minigrid_map_grid_shape = (int(th), int(tw))
                    except Exception:
                        pass
                    # Post-process native grid renders to ensure locked-door
                    # visibility and optional key-door coloring consistency.
                    try:
                        # Attempt to mark locked doors and color doors to key
                        # color when the state indicates so. We use the
                        # canonical FourRooms layout to find door tile coords.
                        from envs.minigrid.minigrid_observation_generator import create_four_rooms_layout
                        base = np.array(create_four_rooms_layout())
                        map_h = base.shape[0]
                        map_w = base.shape[1]
                        tile_px_h = img_arr.shape[0] // map_h if map_h > 0 else 24
                        tile_px_w = img_arr.shape[1] // map_w if map_w > 0 else tile_px_h

                        # Parse key presence and desired door coloring
                        key_present = False
                        try:
                            if isinstance(state_dict, dict) and state_dict.get('key_pos') is not None:
                                kx, ky = map(int, state_dict.get('key_pos'))
                                key_present = (kx >= 0 and 0 <= ky < map_h and 0 <= kx < map_w)
                        except Exception:
                            key_present = False

                        # If door_unlocked explicitly False, draw a knob
                        door_unlocked = None
                        if isinstance(state_dict, dict) and 'door_unlocked' in state_dict:
                            try:
                                door_unlocked = bool(state_dict.get('door_unlocked'))
                            except Exception:
                                door_unlocked = None

                        # Choose knob color (darker variant of key color when present)
                        if key_present:
                            base_key = np.array((255, 215, 0), dtype=np.float32)
                            knob_color = (base_key * np.array([0.2, 0.2, 0.15], dtype=np.float32)).astype(np.uint8)
                            door_fill = np.array((255, 215, 0), dtype=np.uint8)
                        else:
                            knob_color = np.array([80, 50, 20], dtype=np.uint8)
                            door_fill = np.array([160, 120, 60], dtype=np.uint8)

                        # Mutate a copy to apply overlays
                        out_img = img_arr[:, :, :3].astype(np.uint8).copy()

                        if key_present:
                            # Color door tiles to match key color for clarity
                            for y in range(map_h):
                                for x in range(map_w):
                                    if int(base[y, x]) == 4:  # door
                                        y0 = y * tile_px_h
                                        x0 = x * tile_px_w
                                        out_img[y0:y0 + tile_px_h, x0:x0 + tile_px_w] = door_fill

                        if door_unlocked is not None and door_unlocked is False:
                            # Draw knob marker on each door tile
                            for y in range(map_h):
                                for x in range(map_w):
                                    if int(base[y, x]) == 4:  # door
                                        cy = y * tile_px_h + tile_px_h // 2
                                        cx = x * tile_px_w + tile_px_w // 2
                                        rr = max(1, tile_px_h // 6)
                                        yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
                                        kmask = xx * xx + yy * yy <= rr * rr
                                        y0 = max(0, cy - rr)
                                        x0 = max(0, cx - rr)
                                        sub = out_img[y0:y0 + kmask.shape[0], x0:x0 + kmask.shape[1]]
                                        if sub.shape[0] == kmask.shape[0] and sub.shape[1] == kmask.shape[1]:
                                            sub[kmask] = knob_color
                                            out_img[y0:y0 + kmask.shape[0], x0:x0 + kmask.shape[1]] = sub

                        # Return the post-processed image
                        self._minigrid_map_grid_shape = (int(map_h), int(map_w))
                        return out_img
                    except Exception:
                        # Non-fatal; return original image
                        return img_arr[:, :, :3].astype(np.uint8)
                    
                    # If post-processing block didn't execute for any reason,
                    # fall back to returning raw image
                    return img_arr[:, :, :3].astype(np.uint8)
            except Exception:
                # If grid rendering failed, fall through to module helper
                pass

        # If no grid.render found, try the mini renderer helper from the module
        try:
            from envs.minigrid.minigrid_observation_generator import render_full_map_from_state, create_four_rooms_layout
            helper_img = render_full_map_from_state(state_dict if state_dict is not None else {})
            if isinstance(helper_img, (np.ndarray,)) and helper_img.ndim == 3 and helper_img.shape[2] >= 3:
                self._minigrid_map_source = 'minigrid_module'
                try:
                    base = create_four_rooms_layout()
                    self._minigrid_map_grid_shape = (int(base.shape[0]), int(base.shape[1]))
                except Exception:
                    try:
                        th = helper_img.shape[0] // 24 if helper_img.shape[0] >= 24 else helper_img.shape[0]
                        tw = helper_img.shape[1] // 24 if helper_img.shape[1] >= 24 else helper_img.shape[1]
                        self._minigrid_map_grid_shape = (int(th), int(tw))
                    except Exception:
                        pass
                return helper_img[:, :, :3].astype(np.uint8)
        except Exception:
            # no helper available or it failed; continue to fallback
            pass

        try:
            self._minigrid_map_source = 'fallback'
        except Exception:
            pass
        except Exception:
            # If anything goes wrong, continue with the fallback map renderer
            pass
        # Try to use the canonical FourRooms layout generator if available
        try:
            from envs.minigrid.minigrid_observation_generator import create_four_rooms_layout
            base = np.array(create_four_rooms_layout())
        except Exception:
            # Fallback: 13x13 empty floor
            base = np.zeros((13, 13), dtype=np.int32)

        layout = base.copy()

        # Overlay dynamic objects from state if present
        try:
            # key_pos is stored as (x, y) in saved state
            if 'key_pos' in state_dict:
                kp = np.array(state_dict['key_pos'])
                if kp.shape[0] == 2 and kp[0] >= 0:
                    layout[int(kp[1]), int(kp[0])] = 5  # key
        except Exception:
            pass

        h, w = layout.shape
        block_size = 24
        img_h = h * block_size
        img_w = w * block_size
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        # MiniGrid-like palette (close to Farama/Minigrid defaults)
        palette = {
            0: (234, 231, 164),  # floor (match integration video)
            1: (234, 231, 164),
            2: (100, 100, 100),  # wall
            3: (234, 231, 164),
            4: (160, 120, 60),   # door
            5: (255, 215, 0),    # key
            6: (160, 32, 240),   # ball
            7: (139, 69, 19),    # box
            8: (80, 200, 120),   # goal (emerald)
            9: (255, 69, 0),     # lava
        }

        # Fill tiles with subtle shading and special handling for walls/doors
        for y in range(h):
            for x in range(w):
                t = int(layout[y, x])
                base_color = np.array(palette.get(t, (200, 200, 200)), dtype=np.uint8)
                # subtle checkered variation for floor tiles
                if t in (0, 1, 3):
                    if (x + y) % 2 == 0:
                        base_color = np.clip(base_color + 6, 0, 255)
                y0 = y * block_size
                x0 = x * block_size
                # apply slight vertical gradient for depth
                for yy in range(block_size):
                    shade = 0.92 + 0.08 * (yy / max(1, block_size - 1))
                    row_color = np.clip((base_color.astype(np.float32) * shade).astype(np.uint8), 0, 255)
                    img[y0 + yy, x0:x0 + block_size] = row_color

                # special handling for walls and doors to add shape
                if t == 2:
                    # wall: darker inner rectangle
                    inner = np.clip(base_color * 0.6, 0, 255).astype(np.uint8)
                    pad = max(1, block_size // 8)
                    img[y0 + pad:y0 + block_size - pad, x0 + pad:x0 + block_size - pad] = inner
                if t == 4:
                    # door: draw darker rectangle with knob
                    # If a key is present on the map, color the door to match the key
                    if (layout == 5).any():
                        door_color = np.array(palette[5], dtype=np.uint8)
                    else:
                        door_color = base_color
                    pad = max(1, block_size // 6)
                    img[y0 + pad:y0 + block_size - pad, x0 + pad:x0 + block_size - pad] = door_color
                    # knob (small dark circle on right side)
                    kx = x0 + block_size - pad - 3
                    ky = y0 + block_size // 2
                    rr = 2
                    yyk, xxk = np.ogrid[-rr:rr + 1, -rr:rr + 1]
                    kmask = xxk * xxk + yyk * yyk <= rr * rr
                    sub = img[ky - rr:ky + rr + 1, kx - rr:kx + rr + 1]
                    sub[kmask] = np.array([60, 40, 20], dtype=np.uint8)
                    img[ky - rr:ky + rr + 1, kx - rr:kx + rr + 1] = sub

        # Draw faint grid lines
        grid_color = np.array([200, 200, 200], dtype=np.uint8)
        for y in range(h + 1):
            yy = min(img_h - 1, max(0, y * block_size))
            img[yy:yy + 1, :] = np.minimum(img[yy:yy + 1, :], grid_color)
        for x in range(w + 1):
            xx = min(img_w - 1, max(0, x * block_size))
            img[:, xx:xx + 1] = np.minimum(img[:, xx:xx + 1], grid_color)

        # Draw goals explicitly (diamond marker with border and subtle shine)
        try:
            layout[11, 1] = 8
            layout[11, 11] = 8
            for idx, (gx, gy) in enumerate([(1, 11), (11, 11)]):
                y0 = gy * block_size
                x0 = gx * block_size
                cx = x0 + block_size // 2
                cy = y0 + block_size // 2
                r = block_size // 3
                # Create diamond mask
                yy, xx = np.ogrid[-r:r+1, -r:r+1]
                mask = (np.abs(xx) + np.abs(yy)) <= r
                sub_h = mask.shape[0]
                sub_w = mask.shape[1]
                sub = img[cy - r:cy - r + sub_h, cx - r:cx - r + sub_w]
                # Use emerald for the small goal and neon for the large goal
                if idx == 0:
                    fill = np.array(palette[8], dtype=np.uint8)
                else:
                    fill = np.array((15, 255, 80), dtype=np.uint8)
                border = np.clip(fill * 0.75, 0, 255).astype(np.uint8)
                # Apply gradient: brighter at top-left
                for i in range(sub_h):
                    for j in range(sub_w):
                        if mask[i, j]:
                            # gradient factor
                            factor = 0.9 + 0.2 * (1 - (i / max(1, sub_h - 1))) * (1 - (j / max(1, sub_w - 1)))
                            sub[i, j] = np.clip(fill * factor, 0, 255).astype(np.uint8)
                # border (one-pixel outline)
                border_mask = np.logical_and(mask, ~(
                    np.pad(mask, 1)[1:-1, 1:-1] &
                    np.roll(mask, 1, axis=0) & np.roll(mask, -1, axis=0) &
                    np.roll(mask, 1, axis=1) & np.roll(mask, -1, axis=1)
                ))
                sub[border_mask] = border
                img[cy - r:cy - r + sub_h, cx - r:cx - r + sub_w] = sub
        except Exception:
            pass

        # Draw agent as a directional triangle with soft shadow and anti-aliased outline
        try:
            ap = np.array(state_dict.get('agent_pos'))
            ad = int(np.array(state_dict.get('agent_dir', 0)))
            if ap.shape[0] == 2:
                ax, ay = int(ap[0]), int(ap[1])
                cx = int(ax * block_size + block_size // 2)
                cy = int(ay * block_size + block_size // 2)
                s = block_size // 2 - 2

                # Shadow (soft ellipsis) under agent
                try:
                    rr_x = max(1, s // 2)
                    rr_y = max(1, s // 3)
                    sy0 = max(0, cy + rr_y)
                    sx0 = max(0, cx - rr_x)
                    yy, xx = np.ogrid[-rr_y:rr_y + 1, -rr_x:rr_x + 1]
                    shadow_mask = (xx * xx) / (rr_x * rr_x) + (yy * yy) / (rr_y * rr_y) <= 1.0
                    alpha = 0.28
                    sh_color = np.array([10, 10, 10], dtype=np.uint8)
                    sub = img[sy0 - rr_y:sy0 - rr_y + shadow_mask.shape[0], sx0: sx0 + shadow_mask.shape[1]].astype(np.float32)
                    for i in range(shadow_mask.shape[0]):
                        for j in range(shadow_mask.shape[1]):
                            if shadow_mask[i, j]:
                                sub[i, j] = (1 - alpha) * sub[i, j] + alpha * sh_color
                    img[sy0 - rr_y:sy0 - rr_y + shadow_mask.shape[0], sx0: sx0 + shadow_mask.shape[1]] = np.clip(sub, 0, 255).astype(np.uint8)
                except Exception:
                    pass

                # Triangle points depending on direction (0:right,1:down,2:left,3:up)
                if ad == 0:  # right
                    pts = np.array([[cx + s, cy], [cx - s, cy - s], [cx - s, cy + s]])
                elif ad == 1:  # down
                    pts = np.array([[cx, cy + s], [cx - s, cy - s], [cx + s, cy - s]])
                elif ad == 2:  # left
                    pts = np.array([[cx - s, cy], [cx + s, cy - s], [cx + s, cy + s]])
                else:  # up
                    pts = np.array([[cx, cy - s], [cx - s, cy + s], [cx + s, cy + s]])

                # Rasterize triangle with fill
                minx = max(0, pts[:, 0].min().astype(int))
                maxx = min(img_w - 1, pts[:, 0].max().astype(int))
                miny = max(0, pts[:, 1].min().astype(int))
                maxy = min(img_h - 1, pts[:, 1].max().astype(int))
                tri_color = np.array([255, 0, 0], dtype=np.uint8)
                outline_color = np.array([255, 10, 10], dtype=np.uint8)
                mask = np.zeros((maxy - miny + 1, maxx - minx + 1), dtype=bool)
                v0 = pts[2] - pts[0]
                v1 = pts[1] - pts[0]
                denom = v0[0] * v1[1] - v1[0] * v0[1]
                if denom != 0:
                    for yy in range(miny, maxy + 1):
                        for xx in range(minx, maxx + 1):
                            v2 = np.array([xx, yy]) - pts[0]
                            a = (v2[0] * v1[1] - v1[0] * v2[1]) / denom
                            b = (v0[0] * v2[1] - v2[0] * v0[1]) / denom
                            if a >= 0 and b >= 0 and (a + b) <= 1:
                                mask[yy - miny, xx - minx] = True
                # Apply fill
                img[miny:maxy + 1, minx:maxx + 1][mask] = tri_color

                # Anti-aliased outline: blend boundary pixels
                sub_img = img[miny:maxy + 1, minx:maxx + 1]
                h_sub, w_sub = mask.shape
                blended = sub_img.copy().astype(np.float32)
                for yy in range(h_sub):
                    for xx in range(w_sub):
                        if mask[yy, xx]:
                            # compute neighbor fraction
                            neigh = mask[max(0, yy-1):min(h_sub, yy+2), max(0, xx-1):min(w_sub, xx+2)]
                            frac = neigh.sum() / float(neigh.size)
                            if frac < 1.0:
                                # blend with outline color based on edge proximity
                                alpha = np.clip(1.0 - frac + 0.2, 0.0, 1.0)
                                blended[yy, xx] = (1 - alpha) * blended[yy, xx] + alpha * outline_color
                img[miny:maxy + 1, minx:maxx + 1] = np.clip(blended, 0, 255).astype(np.uint8)

        except Exception:
            pass

        return img

    def _render_minigrid_fov_from_obs(self, observation) -> np.ndarray:
        """Render a MiniGrid-style FOV image from an observation array."""
        # Normalize observation format into 2D tile ids (9x9)
        obs = None
        try:
            arr = np.array(observation)
            if arr.ndim == 4:
                # (batch, H, W, C) -> take first
                arr = arr[0]
            if arr.ndim == 3:
                # (H, W, C) -> assume first channel contains tile ids
                obs = arr[:, :, 0]
            elif arr.ndim == 2:
                obs = arr
            else:
                # Flattened vector case
                if arr.size >= 81:
                    obs = arr.flatten()[:81].reshape(9, 9)
                else:
                    obs = np.zeros((9, 9), dtype=int)
        except Exception:
            obs = np.zeros((9, 9), dtype=int)

        # MiniGrid palette (similar colors used in _render_minigrid_map_from_state)
        palette = {
            0: (220, 220, 220),  # floor
            1: (220, 220, 220),
            2: (100, 100, 100),  # wall
            3: (220, 220, 220),
            4: (160, 120, 60),   # door
            5: (255, 215, 0),    # key
            6: (160, 32, 240),   # ball
            7: (139, 69, 19),    # box
            8: (80, 200, 120),   # goal (emerald)
            9: (255, 69, 0),     # lava
        }

        h, w = obs.shape
        block_size = 64
        img = np.zeros((h * block_size, w * block_size, 3), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                t = int(obs[y, x]) if not np.isnan(obs[y, x]) else 0
                color = palette.get(t, (200, 200, 200))
                img[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size] = color

        # Agent marker at center (agent is at center of FOV)
        cy = (h // 2) * block_size + block_size // 2
        cx = (w // 2) * block_size + block_size // 2
        rr = block_size // 4
        yy, xx = np.ogrid[-rr:rr, -rr:rr]
        mask = xx*xx + yy*yy <= rr*rr
        sub = img[cy-rr:cy+rr, cx-rr:cx+rr]
        sub[mask] = np.array([255, 230, 80], dtype=np.uint8)  # agent color
        img[cy-rr:cy+rr, cx-rr:cx+rr] = sub

        return img
    
    def _render_fov(self, env_state, observation) -> np.ndarray:
        """Render the agent's field of view"""
        if self.use_textures and self.render_function is not None:
            try:
                # Render just the FOV using Craftax renderer with block_pixel_size
                # The observation is 9x9 centered on agent
                pixels = self.render_function(env_state, block_pixel_size=16)
                
                # Extract the FOV region
                if hasattr(env_state, 'player_position'):
                    player_pos = np.array(env_state.player_position)
                    # Handle vectorized environments (batch dimension)
                    if player_pos.ndim == 2:
                        player_pos = player_pos[0]
                    py, px = int(player_pos[0]), int(player_pos[1])
                    
                    # Calculate FOV bounds based on player direction
                    player_dir = np.array(env_state.player_direction) if hasattr(env_state, 'player_direction') else 4
                    # Handle batch dimension
                    if hasattr(player_dir, 'ndim') and player_dir.ndim > 0:
                        player_dir = player_dir[0] if player_dir.shape[0] > 0 else 4
                    direction = int(player_dir)
                    
                    # Agent offset based on direction (matches viewer logic)
                    if direction == 1:  # LEFT
                        agent_row_offset, agent_col_offset = 4, 8
                    elif direction == 2:  # RIGHT
                        agent_row_offset, agent_col_offset = 4, 0
                    elif direction == 3:  # UP
                        agent_row_offset, agent_col_offset = 8, 4
                    else:  # DOWN
                        agent_row_offset, agent_col_offset = 0, 4
                    
                    # Calculate FOV top-left corner
                    fov_top = max(0, py - agent_row_offset)
                    fov_left = max(0, px - agent_col_offset)
                    fov_bottom = min(pixels.shape[0] // 16, fov_top + 9)  # Assuming 16px blocks
                    fov_right = min(pixels.shape[1] // 16, fov_left + 9)
                    
                    # Extract FOV (convert from block coords to pixel coords)
                    fov_pixels = pixels[fov_top*16:fov_bottom*16, fov_left*16:fov_right*16]
                    return np.array(fov_pixels) if hasattr(fov_pixels, '__array__') else fov_pixels
                    
            except Exception as e:
                # Silently fall back (error already shown for full map)
                if not hasattr(self, '_fov_error_shown'):
                    print(f"⚠️  FOV rendering failed, using fallback: {e}")
                    self._fov_error_shown = True
        
        # Fallback: render observation as simple colored grid
        return self._render_fov_simple(observation)
    
    def _render_fov_simple(self, observation) -> np.ndarray:
        """Simple rendering of the observation"""
        # Observation is typically (9, 9, 21) for Craftax symbolic
        # Or (batch, 9, 9, 21) for vectorized environments
        if hasattr(observation, 'shape'):
            obs_array = np.array(observation)
            
            # Debug: print shape once to understand observation structure
            if not hasattr(self, '_obs_shape_logged'):
                print(f"📊 Observation shape: {obs_array.shape}, dtype: {obs_array.dtype}")
                if len(obs_array.shape) >= 3:
                    print(f"📊 First block values (sample): {obs_array[0, 0, :5] if obs_array.shape[0] > 0 else 'empty'}")
                self._obs_shape_logged = True
            
            # Handle batch dimension
            if len(obs_array.shape) == 4:
                # (batch, height, width, channels) -> (height, width, channels)
                obs_array = obs_array[0]
            
            if len(obs_array.shape) == 3 and obs_array.shape[2] > 0:
                # Take first channel (block types)
                obs_2d = obs_array[:, :, 0]
            elif len(obs_array.shape) == 2:
                obs_2d = obs_array
            else:
                obs_2d = np.zeros((9, 9))
        else:
            obs_2d = np.zeros((9, 9))
        
        # Debug: print value range once
        if not hasattr(self, '_obs_values_logged'):
            print(f"📊 Observation values - min: {obs_2d.min()}, max: {obs_2d.max()}, unique: {len(np.unique(obs_2d))}")
            self._obs_values_logged = True
        
        height, width = obs_2d.shape
        block_size = 64  # Larger blocks for FOV (more detail)
        
        # Create RGB image
        img = np.zeros((height * block_size, width * block_size, 3), dtype=np.uint8)
        
        # Use same color map as full map (matching Craftax block types)
        color_map = {
            0: (255, 0, 255),    # Invalid - magenta
            1: (0, 0, 0),        # Out of bounds - black
            2: (34, 139, 34),    # Grass - green
            3: (30, 144, 255),   # Water - blue
            4: (128, 128, 128),  # Stone - gray
            5: (0, 100, 0),      # Tree - dark green
            6: (139, 69, 19),    # Wood - brown
            7: (210, 180, 140),  # Path - tan
            8: (64, 64, 64),     # Coal - dark gray
            9: (192, 192, 192),  # Iron - light gray
            10: (185, 242, 255), # Diamond - cyan
            11: (160, 82, 45),   # Crafting table - brown
            12: (105, 105, 105), # Furnace - gray
            13: (244, 164, 96),  # Sand - sandy brown
            14: (255, 69, 0),    # Lava - red-orange
        }
        
        for y in range(height):
            for x in range(width):
                block_type = int(obs_2d[y, x])
                color = color_map.get(block_type, (200, 200, 200))  # Default gray
                img[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size] = color
        
        # Highlight agent position in center (agent is at center of FOV)
        center_y, center_x = 4, 4
        # Draw a yellow circle/square for the agent
        agent_color = (255, 255, 0)  # Bright yellow
        # Draw agent marker (slightly smaller than full block)
        margin = block_size // 4
        img[center_y*block_size+margin:(center_y+1)*block_size-margin, 
            center_x*block_size+margin:(center_x+1)*block_size-margin] = agent_color
        
        return img
    
    def _resize_image(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio"""
        try:
            from PIL import Image
            
            pil_img = Image.fromarray(img)
            
            # Calculate aspect ratio
            img_ratio = pil_img.width / pil_img.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                # Image is wider than target
                new_width = target_size[0]
                new_height = int(target_size[0] / img_ratio)
            else:
                # Image is taller than target
                new_height = target_size[1]
                new_width = int(target_size[1] * img_ratio)
            
            resized = pil_img.resize((new_width, new_height), Image.Resampling.NEAREST)
            return np.array(resized)
            
        except ImportError:
            # Fallback: simple nearest neighbor resize without PIL
            return self._simple_resize(img, target_size)
    
    def _simple_resize(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Simple resize without external dependencies"""
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factors
        scale_h = target_h / h
        scale_w = target_w / w
        scale = min(scale_h, scale_w)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Simple nearest neighbor
        indices_h = np.clip(np.arange(new_h) / scale, 0, h - 1).astype(int)
        indices_w = np.clip(np.arange(new_w) / scale, 0, w - 1).astype(int)
        
        resized = img[indices_h[:, None], indices_w]
        return resized
    
    def save_video(self, filename: str, fps: int = 10) -> Optional[str]:
        """
        Save recorded frames as video
        
        Args:
            filename: Output filename (without extension)
            fps: Frames per second
            
        Returns:
            Path to saved video file, or None if failed
        """
        if len(self.frames) == 0:
            print("⚠️  No frames to save")
            return None
        
        output_path = os.path.join(self.output_dir, f"{filename}.mp4")
        
        try:
            # Try using imageio (preferred)
            import imageio
            
            with imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p') as writer:
                for frame in self.frames:
                    writer.append_data(frame)
            
            print(f"✅ Video saved: {output_path}")
            return output_path
            
        except ImportError:
            print("⚠️  imageio not available, trying opencv")
            
            try:
                import cv2
                
                height, width = self.frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                for frame in self.frames:
                    # Convert RGB to BGR for opencv
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                print(f"✅ Video saved: {output_path}")
                return output_path
                
            except ImportError:
                print("⚠️  Neither imageio nor opencv available, cannot save video")
                print("   Install with: pip install imageio imageio-ffmpeg")
                return None
    
    def log_to_wandb(self, video_path: str, key: str = "video/best_episode", caption: str = "Best Episode"):
        """
        Log video to wandb
        
        Args:
            video_path: Path to video file
            key: wandb log key
            caption: Video caption
        """
        if wandb.run is None:
            print("⚠️  wandb not initialized, skipping video upload")
            return
        
        try:
            wandb.log({key: wandb.Video(video_path, caption=caption, format="mp4")}, commit=False)
            print(f"✅ Video logged to wandb: {key}")
        except Exception as e:
            print(f"⚠️  Failed to log video to wandb: {e}")
    
    def clear_frames(self):
        """Clear recorded frames to free memory"""
        self.frames = []
        self.recording = False
