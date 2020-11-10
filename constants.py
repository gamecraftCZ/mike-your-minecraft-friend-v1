from structures import Vec3


class Blocks:
    AIR = 0
    GROUND = 1
    WOOD = 2
    LEAF = 3


# Environment is 9x9x9 blocks -> 729 blocks in total
WORLD_SHAPE = Vec3(9, 9, 9)

MIN_TREE_HEIGHT = 5
MAX_TREE_HEIGHT = 7

GRAVITY = 0.08  # Blocks / 1 tick^2

TERMINAL_VELOCITY = 3.92  # Blocks / 1 tick
JUMP_VELOCITY = 0.42  # 0.12522 # 0.25044  # Blocks / 1 tick
MOVE_VELOCITY = 0.21585  # Blocks / 1 tick
