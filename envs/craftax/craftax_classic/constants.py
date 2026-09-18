import os
import pathlib
from enum import Enum

import jax
import jax.numpy as jnp
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageEnhance
from envs.craftax.environment_base.util import load_compressed_pickle, save_compressed_pickle

# GAME CONSTANTS
OBS_DIM = (9, 9)
MAX_OBS_DIM = max(OBS_DIM)
# Note: OBS_DIM no longer needs to be odd since agent is not centered
# assert OBS_DIM[0] % 2 == 1 and OBS_DIM[1] % 2 == 1
BLOCK_PIXEL_SIZE_HUMAN = 64
BLOCK_PIXEL_SIZE_IMG = 16
BLOCK_PIXEL_SIZE_AGENT = 7
INVENTORY_OBS_HEIGHT = 2
TEXTURE_CACHE_FILE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "assets", "texture_cache_classic.pbz2"
)

# ENUMS
class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16


class Action(Enum):
    NOOP = 0  #
    TURN_LEFT = 1  # a - turn counter-clockwise
    TURN_RIGHT = 2  # d - turn clockwise
    FORWARD = 3  # w - move forward in current direction
    DO = 4  # space - was 5
    SLEEP = 5  # tab - was 6
    PLACE_STONE = 6  # r - was 7
    PLACE_TABLE = 7  # t - was 8
    PLACE_FURNACE = 8  # f - was 9
    PLACE_PLANT = 9  # p - was 10
    MAKE_WOOD_PICKAXE = 10  # 1 - was 11
    MAKE_STONE_PICKAXE = 11  # 2 - was 12
    MAKE_IRON_PICKAXE = 12  # 3 - was 13
    MAKE_WOOD_SWORD = 13  # 4 - was 14
    MAKE_STONE_SWORD = 14  # 5 - was 15
    MAKE_IRON_SWORD = 15  # 6 - was 16


# Player direction constants (for internal use)
# LEFT = 1, RIGHT = 2, UP = 3, DOWN = 4
DIRECTION_LEFT = 1
DIRECTION_RIGHT = 2
DIRECTION_UP = 3
DIRECTION_DOWN = 4

# Direction deltas: maps player_direction to movement delta [row, col]
# Index 0 is unused, indices 1-4 correspond to LEFT, RIGHT, UP, DOWN
DIRECTION_DELTAS = jnp.array([
    [0, 0],      # 0: unused
    [0, -1],     # 1: LEFT
    [0, 1],      # 2: RIGHT
    [-1, 0],     # 3: UP
    [1, 0],      # 4: DOWN
], dtype=jnp.int32)

# GAME MECHANICS
# DIRECTIONS array: maps actions to movement deltas
# For turn actions (1, 2), the delta is [0, 0] (no movement, just rotation)
# For forward action (3), we compute the actual direction delta in move_player
# Remaining slots for non-movement actions also have [0, 0]
DIRECTIONS = jnp.concatenate(
    (
        jnp.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32),  # NOOP, TURN_LEFT, TURN_RIGHT, FORWARD (computed dynamically)
        jnp.zeros((12, 2), dtype=jnp.int32),  # All other actions (DO through MAKE_IRON_SWORD)
    ),
    axis=0,
)

CLOSE_BLOCKS = jnp.array(
    [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    dtype=jnp.int32,
)

# Can't walk through these
SOLID_BLOCKS = jnp.array(
    [
        BlockType.WATER.value,
        BlockType.STONE.value,
        BlockType.TREE.value,
        BlockType.COAL.value,
        BlockType.IRON.value,
        BlockType.DIAMOND.value,
        BlockType.CRAFTING_TABLE.value,
        BlockType.FURNACE.value,
        BlockType.PLANT.value,
        BlockType.RIPE_PLANT.value,
    ],
    dtype=jnp.int32,
)


# ACHIEVEMENTS
class Achievement(Enum):
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    MAKE_WOOD_PICKAXE = 5
    MAKE_WOOD_SWORD = 6
    PLACE_PLANT = 7
    DEFEAT_ZOMBIE = 8
    COLLECT_STONE = 9
    PLACE_STONE = 10
    EAT_PLANT = 11
    DEFEAT_SKELETON = 12
    MAKE_STONE_PICKAXE = 13
    MAKE_STONE_SWORD = 14
    WAKE_UP = 15
    PLACE_FURNACE = 16
    COLLECT_COAL = 17
    COLLECT_IRON = 18
    COLLECT_DIAMOND = 19
    MAKE_IRON_PICKAXE = 20
    MAKE_IRON_SWORD = 21


# Achievement reward values - Diamond-focused progression with reduced hidden sub-goals
# Total rewards: 100 points distributed across multiple tiers to emphasize diamond collection
# Strategy: Keep only 4 critical hidden sub-goals to speed up intrinsic learning while still challenging pure PPO
# Reward tiers: 0 (hidden), 1-2 (trivial), 3 (easy), 5 (moderate), 7 (medium), 10 (hard), 40 (ultimate)
ACHIEVEMENT_REWARDS = jnp.array([
    # Basic resource gathering - Now rewarded to guide progression
    1.0,  # COLLECT_WOOD - (Basic resource needed for all crafting, now rewarded)
    0.0,  # PLACE_TABLE - Hidden sub-goal (Critical bottleneck: enables all crafting)
    
    # Basic survival
    1.0,  # EAT_COW - (Basic survival: food from hunting)
    1.0,  # COLLECT_SAPLING - (Enables farming, now rewarded to guide agents)
    1.0,  # COLLECT_DRINK - (Basic survival: thirst management)
    
    # Tool crafting progression - Mixed rewards
    3.0,  # MAKE_WOOD_PICKAXE - (First tool, unlocks mining - now rewarded to speed progression)
    3.0,  # MAKE_WOOD_SWORD - (Early combat capability, basic weapon crafting)
    2.0,  # PLACE_PLANT - (Farming setup, now rewarded)
    
    # Combat achievements
    6.0,  # DEFEAT_ZOMBIE - (Combat mastery: defeating aggressive mob)
    
    # Mid-tier progression
    3.0,  # COLLECT_STONE - (Necessary for stone tools, now rewarded)
    4.0,  # PLACE_STONE - (Infrastructure: building capability)
    3.0,  # EAT_PLANT - (Sustainable food: farming reward)
    
    # Advanced combat
    7.0,  # DEFEAT_SKELETON - (Advanced combat: defeating ranged enemy)
    
    # Tool progression - Critical hidden bottleneck
    5.0,  # MAKE_STONE_PICKAXE - Hidden sub-goal (Critical: unlocks coal/iron mining) (2SG -> 5; 4SG -> 0)
    5.0,  # MAKE_STONE_SWORD - (Better combat: stone weapon crafting)
    2.0,  # WAKE_UP - (Survival: energy/sleep management)
    
    # Late-game infrastructure - Key hidden bottlenecks
    0.0,  # PLACE_FURNACE - Hidden sub-goal (Critical: essential for metal processing)
    3.0,  # COLLECT_COAL - (Required for smelting, now rewarded)
    
    # Endgame progression
    5.0,  # COLLECT_IRON - (Required for iron tools, now rewarded)
    
    # Ultimate goal
    25.0, # COLLECT_DIAMOND - Ultimate goal with highest reward (requires complete tech tree) (2SG -> 25; 4SG -> 40)
    
    # Final tool - Critical hidden bottleneck
    10.0,  # MAKE_IRON_PICKAXE - Hidden sub-goal (Critical: required for diamond mining) (2SG -> 10; 4SG -> 0)
    
    # Endgame mastery
    10.0,  # MAKE_IRON_SWORD - (Endgame weapon: iron sword mastery)
], dtype=jnp.float32)  # Total: 100 points with 4 strategic hidden sub-goals
# Note: Only 4 hidden sub-goals (PLACE_TABLE, MAKE_STONE_PICKAXE, PLACE_FURNACE, MAKE_IRON_PICKAXE)
# These are the absolute minimum bottlenecks that intrinsic motivation can discover faster than pure PPO
# This allows intrinsically motivated agents to reach diamond ~2-3x faster while still challenging vanilla PPO


# TEXTURES
def load_texture(filename, block_pixel_size, clamp_alpha=True):
    filename = os.path.join(pathlib.Path(__file__).parent.resolve(), "assets", filename)
    img = iio.imread(filename)
    jnp_img = jnp.array(img).astype(int)
    assert jnp_img.shape[:2] == (16, 16)

    if jnp_img.shape[2] == 4 and clamp_alpha:
        jnp_img = jnp_img.at[:, :, 3].set(jnp_img[:, :, 3] // 255)

    if block_pixel_size != 16:
        img = np.array(jnp_img, dtype=np.uint8)
        image = Image.fromarray(img)
        image = image.resize(
            (block_pixel_size, block_pixel_size), resample=Image.NEAREST
        )
        jnp_img = jnp.array(image, dtype=jnp.int32)

    return jnp_img


def load_all_textures(block_pixel_size):
    small_block_pixel_size = int(block_pixel_size * 0.8)

    # blocks
    texture_names = [
        "debug_tile.png",
        "debug_tile.png",
        "grass.png",
        "water.png",
        "stone.png",
        "tree.png",
        "wood.png",
        "path.png",
        "coal.png",
        "iron.png",
        "diamond.png",
        "table.png",
        "furnace.png",
        "sand.png",
        "lava.png",
        "plant_on_grass.png",
        "ripe_plant_on_grass.png",
    ]

    block_textures = jnp.array(
        [
            load_texture("debug_tile.png", block_pixel_size),
            jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32) * 128,
            load_texture("grass.png", block_pixel_size),
            load_texture("water.png", block_pixel_size),
            load_texture("stone.png", block_pixel_size),
            load_texture("tree.png", block_pixel_size),
            load_texture("wood.png", block_pixel_size)[:, :, :3],
            load_texture("path.png", block_pixel_size)[:, :, :3],
            load_texture("coal.png", block_pixel_size)[:, :, :3],
            load_texture("iron.png", block_pixel_size)[:, :, :3],
            load_texture("diamond.png", block_pixel_size)[:, :, :3],
            load_texture("table.png", block_pixel_size)[:, :, :3],
            load_texture("furnace.png", block_pixel_size)[:, :, :3],
            load_texture("sand.png", block_pixel_size)[:, :, :3],
            load_texture("lava.png", block_pixel_size)[:, :, :3],
            load_texture("plant_on_grass.png", block_pixel_size)[:, :, :3],
            load_texture("ripe_plant_on_grass.png", block_pixel_size)[:, :, :3],
        ]
    )

    block_textures = jnp.array(
        [load_texture(fname, block_pixel_size)[:, :, :3] for fname in texture_names]
    )
    block_textures = block_textures.at[1].set(
        jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32) * 128
    )

    # rng = jax.random.prngkey(0)
    # block_textures = jax.random.permutation(rng, block_textures)

    smaller_block_textures = jnp.array(
        [
            load_texture(fname, int(block_pixel_size * 0.8))[:, :, :3]
            for fname in texture_names
        ]
    )

    full_map_block_textures = jnp.array(
        [jnp.tile(block_textures[block.value], (*OBS_DIM, 1)) for block in BlockType]
    )

    # player
    pad_pixels = (
        (OBS_DIM[0] // 2) * block_pixel_size,
        (OBS_DIM[1] // 2) * block_pixel_size,
    )

    player_textures = [
        load_texture("player-left.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-right.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-up.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-down.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-sleep.png", block_pixel_size, clamp_alpha=False),
    ]

    full_map_player_textures_rgba = [
        jnp.pad(
            player_texture,
            ((pad_pixels[0], pad_pixels[0]), (pad_pixels[1], pad_pixels[1]), (0, 0)),
        )
        for player_texture in player_textures
    ]

    full_map_player_textures = jnp.array(
        [player_texture[:, :, :3] for player_texture in full_map_player_textures_rgba]
    )

    full_map_player_textures_alpha = jnp.array(
        [
            jnp.repeat(
                jnp.expand_dims(player_texture[:, :, 3], axis=-1).astype(float) / 255,
                repeats=3,
                axis=2,
            )
            for player_texture in full_map_player_textures_rgba
        ]
    )

    # inventory

    empty_texture = jnp.zeros((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)
    smaller_empty_texture = jnp.zeros(
        (int(block_pixel_size * 0.8), int(block_pixel_size * 0.8), 3), dtype=jnp.int32
    )

    ones_texture = jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)

    number_size = int(block_pixel_size * 0.6)

    number_textures_rgba = [
        jnp.zeros((number_size, number_size, 3), dtype=jnp.int32),
        load_texture("1.png", number_size),
        load_texture("2.png", number_size),
        load_texture("3.png", number_size),
        load_texture("4.png", number_size),
        load_texture("5.png", number_size),
        load_texture("6.png", number_size),
        load_texture("7.png", number_size),
        load_texture("8.png", number_size),
        load_texture("9.png", number_size),
    ]

    number_textures = jnp.array(
        [
            number_texture[:, :, :3]
            * jnp.repeat(jnp.expand_dims(number_texture[:, :, 3], axis=-1), 3, axis=-1)
            for number_texture in number_textures_rgba
        ]
    )

    number_textures_alpha = jnp.array(
        [
            jnp.repeat(
                jnp.expand_dims(number_texture[:, :, 3], axis=-1), repeats=3, axis=2
            )
            for number_texture in number_textures_rgba
        ]
    )

    health_texture = jnp.array(
        load_texture("health.png", small_block_pixel_size)[:, :, :3]
    )
    hunger_texture = jnp.array(
        load_texture("food.png", small_block_pixel_size)[:, :, :3]
    )
    thirst_texture = jnp.array(
        load_texture("drink.png", small_block_pixel_size)[:, :, :3]
    )
    energy_texture = jnp.array(
        load_texture("energy.png", small_block_pixel_size)[:, :, :3]
    )

    # get rid of the cow ghost
    def apply_alpha(texture):
        return texture[:, :, :3] * jnp.repeat(
            jnp.expand_dims(texture[:, :, 3], axis=-1), 3, axis=-1
        )

    wood_pickaxe_texture = jnp.array(
        load_texture("wood_pickaxe.png", small_block_pixel_size)[:, :, :3]
    )  # no ghosts :)
    stone_pickaxe_texture = jnp.array(
        load_texture("stone_pickaxe.png", small_block_pixel_size)
    )
    stone_pickaxe_texture = apply_alpha(stone_pickaxe_texture)
    iron_pickaxe_texture = jnp.array(
        load_texture("iron_pickaxe.png", small_block_pixel_size)
    )
    iron_pickaxe_texture = apply_alpha(iron_pickaxe_texture)

    wood_sword_texture = jnp.array(
        load_texture("wood_sword.png", small_block_pixel_size)
    )
    wood_sword_texture = apply_alpha(wood_sword_texture)
    stone_sword_texture = jnp.array(
        load_texture("stone_sword.png", small_block_pixel_size)
    )
    stone_sword_texture = apply_alpha(stone_sword_texture)
    iron_sword_texture = jnp.array(
        load_texture("iron_sword.png", small_block_pixel_size)
    )
    iron_sword_texture = apply_alpha(iron_sword_texture)

    sapling_texture = jnp.array(
        load_texture("sapling.png", small_block_pixel_size)[:, :, :3]
    )

    # entities
    zombie_texture_rgba = jnp.array(
        load_texture("zombie.png", block_pixel_size, clamp_alpha=False)
    )
    zombie_texture = zombie_texture_rgba[:, :, :3]
    zombie_texture_alpha = jnp.repeat(
        jnp.expand_dims(zombie_texture_rgba[:, :, 3], axis=-1).astype(float) / 255,
        repeats=3,
        axis=2,
    )

    cow_texture_rgba = jnp.array(
        load_texture("cow.png", block_pixel_size, clamp_alpha=False)
    )
    cow_texture = cow_texture_rgba[:, :, :3]
    cow_texture_alpha = jnp.repeat(
        jnp.expand_dims(cow_texture_rgba[:, :, 3], axis=-1).astype(float) / 255,
        repeats=3,
        axis=2,
    )

    skeleton_texture_rgba = jnp.array(
        load_texture("skeleton.png", block_pixel_size, clamp_alpha=False)
    )
    skeleton_texture = skeleton_texture_rgba[:, :, :3]
    skeleton_texture_alpha = jnp.repeat(
        jnp.expand_dims(skeleton_texture_rgba[:, :, 3], axis=-1).astype(float) / 255,
        repeats=3,
        axis=2,
    )

    arrow_texture_rgba = jnp.array(load_texture("arrow-up.png", block_pixel_size))
    arrow_texture = apply_alpha(arrow_texture_rgba)
    arrow_texture_alpha = jnp.repeat(
        jnp.expand_dims(arrow_texture_rgba[:, :, 3], axis=-1), repeats=3, axis=2
    )

    night_texture = (
        jnp.array([[[0, 16, 64]]])
        .repeat(OBS_DIM[0] * block_pixel_size, axis=0)
        .repeat(OBS_DIM[1] * block_pixel_size, axis=1)
    )

    xs, ys = np.meshgrid(
        np.linspace(-1, 1, OBS_DIM[0] * block_pixel_size),
        np.linspace(-1, 1, OBS_DIM[1] * block_pixel_size),
    )
    night_noise_intensity_texture = (
        1 - np.exp(-0.5 * (xs**2 + ys**2) / (0.5**2)).T
    )

    night_noise_intensity_texture = jnp.expand_dims(
        night_noise_intensity_texture, axis=-1
    ).repeat(3, axis=-1)

    return {
        "block_textures": block_textures,
        "smaller_block_textures": smaller_block_textures,
        "full_map_block_textures": full_map_block_textures,
        "player_textures": player_textures,
        "full_map_player_textures": full_map_player_textures,
        "full_map_player_textures_alpha": full_map_player_textures_alpha,
        "empty_texture": empty_texture,
        "smaller_empty_texture": smaller_empty_texture,
        "ones_texture": ones_texture,
        "number_textures": number_textures,
        "number_textures_alpha": number_textures_alpha,
        "health_texture": health_texture,
        "hunger_texture": hunger_texture,
        "thirst_texture": thirst_texture,
        "energy_texture": energy_texture,
        "wood_pickaxe_texture": wood_pickaxe_texture,
        "stone_pickaxe_texture": stone_pickaxe_texture,
        "iron_pickaxe_texture": iron_pickaxe_texture,
        "wood_sword_texture": wood_sword_texture,
        "stone_sword_texture": stone_sword_texture,
        "iron_sword_texture": iron_sword_texture,
        "sapling_texture": sapling_texture,
        "zombie_texture": zombie_texture,
        "zombie_texture_alpha": zombie_texture_alpha,
        "cow_texture": cow_texture,
        "cow_texture_alpha": cow_texture_alpha,
        "skeleton_texture": skeleton_texture,
        "skeleton_texture_alpha": skeleton_texture_alpha,
        "arrow_texture": arrow_texture,
        "arrow_texture_alpha": arrow_texture_alpha,
        "night_texture": night_texture,
        "night_noise_intensity_texture": night_noise_intensity_texture,
    }


load_cached_textures_success = True
if os.path.exists(TEXTURE_CACHE_FILE) and not os.environ.get(
    "CRAFTAX_RELOAD_TEXTURES", False
):
    print("Loading Craftax-Classic textures from cache.")
    TEXTURES = load_compressed_pickle(TEXTURE_CACHE_FILE)
    # Check validity of texture cache
    for ts in (BLOCK_PIXEL_SIZE_AGENT, BLOCK_PIXEL_SIZE_IMG, BLOCK_PIXEL_SIZE_HUMAN):
        tex_shape = TEXTURES[ts]["full_map_block_textures"].shape
        if (
            tex_shape[0] != len(BlockType)
            or tex_shape[1] != OBS_DIM[0] * ts
            or tex_shape[2] != OBS_DIM[1] * ts
            or tex_shape[3] != 3
        ):
            load_cached_textures_success = False
            print("Invalid texture cache, going to reload textures.")
            break
    print("Textures successfully loaded from cache.")
else:
    load_cached_textures_success = False

if not load_cached_textures_success:
    print(
        "Processing Craftax-Classic textures. This will take a minute but will be cached for future use."
    )
    TEXTURES = {
        BLOCK_PIXEL_SIZE_AGENT: load_all_textures(BLOCK_PIXEL_SIZE_AGENT),
        BLOCK_PIXEL_SIZE_IMG: load_all_textures(BLOCK_PIXEL_SIZE_IMG),
        BLOCK_PIXEL_SIZE_HUMAN: load_all_textures(BLOCK_PIXEL_SIZE_HUMAN),
    }

    save_compressed_pickle(TEXTURE_CACHE_FILE, TEXTURES)
    print("Textures loaded and saved to cache.")
