import math
from typing import List

import numpy as np

from constants import WORLD_SHAPE
from structures import Vec2, Vec3


def getCollisionsBottom(posX: int, posY: int) -> List[Vec2]:
    relPosX = posX % 1
    relPosY = posY % 1

    X = []

    if -0.3 < posX < WORLD_SHAPE.x - 0.3:  # <= WORLD_SHAPE.x - 0.5:
        X.append(math.floor(posX + 0.3))
    if relPosX < 0.3 or relPosX > 0.7:
        # Collision with 2nd block on X axis
        if 0.7 <= posX < WORLD_SHAPE.x + 0.7:
            X.append(math.floor(posX + 0.3) - 1)

    Y = []

    if -0.3 < posY < WORLD_SHAPE.y - 0.3:  # <= WORLD_SHAPE.y - 0.5:
        Y.append(math.floor(posY + 0.3))
    if relPosY < 0.3 or relPosY > 0.7:
        # Collision with 2nd block on Y axis
        if 0.7 <= posY < WORLD_SHAPE.y + 0.7:
            Y.append(math.floor(posY + 0.3) - 1)

    return [Vec2(x, y) for x in X for y in Y]


def isStanding(position: Vec3, environment: np.ndarray) -> bool:
    if position.z % 1:
        return False  # Not a integer -> Not standing on a block

    posZ = int(position.z) - 1
    if posZ < 0:
        posZ = 0
    if posZ >= environment.shape[0]:
        posZ = environment.shape[0] - 1

    collisions = getCollisionsBottom(position.x, position.y)

    for collision in collisions:
        block = environment[posZ][collision.y][collision.x]
        if block:
            return True

    return False
