from gym_treechop.game.structures import Vec3

block_names = {
    0: "air",
    1: "ground",
    2: "wood",
    3: "leaf"
}


class Blocks:
    AIR = 0
    GROUND = 1
    WOOD = 2
    LEAF = 3

    @staticmethod
    def toName(blockId: int) -> str:
        return block_names[blockId]


BlockHardness = {  # Default seconds required to break
    Blocks.AIR: 0.0000001,
    Blocks.GROUND: 0.5,
    Blocks.WOOD: 2,
    Blocks.LEAF: 0.2
}
BLOCK_TYPES = [0, 1, 2, 3]

BREAKING_RANGE = 4.5  # Blocks

HARDNESS_MULTIPLIER = 1.5
NOT_STANDING_BREAK_SLOWDOWN = 5  # When not standing the block is broken 5x longer

# Environment is 9x9x9 blocks -> 729 blocks in total
WORLD_SHAPE = Vec3(9, 9, 9)
WORLD_SHAPE_TUPLE = (WORLD_SHAPE.x, WORLD_SHAPE.y, WORLD_SHAPE.z)

PLAYER_RADIUS = 0.3
PLAYER_HEIGHT = 1.8

MIN_TREE_HEIGHT = 6
MAX_TREE_HEIGHT = 6  # 7 might be undestroyable in some situations without building blocks

GRAVITY = 0.08  # Blocks / 1 tick^2

TERMINAL_VELOCITY = 3.92  # Blocks / 1 tick
JUMP_VELOCITY = 0.45  # 0.42  # 0.12522 # 0.25044  # Blocks / 1 tick
WALK_VELOCITY = 0.21585  # Blocks / 1 tick
