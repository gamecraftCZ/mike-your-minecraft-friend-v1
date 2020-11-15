import math
from typing import List

import numpy as np

from constants import WORLD_SHAPE, PLAYER_RADIUS
from structures import Vec2, Vec3, Axis


def getCollisionsBottom(pos: Vec2) -> List[Vec2]:
    points = getRectanglePointsAroundPointVec2(pos, PLAYER_RADIUS)
    points = [point for point in points if 0 <= point.x < WORLD_SHAPE.x]
    points = [point for point in points if 0 <= point.y < WORLD_SHAPE.y]
    return points


def playerIsStanding(position: Vec3, environment: np.ndarray) -> bool:
    if position.z % 1:
        return False  # Not a integer -> Not standing on a block

    posZ = int(position.z) - 1
    if posZ < 0:
        posZ = 0
    if posZ >= environment.shape[0]:
        posZ = environment.shape[0] - 1

    collisions = getCollisionsBottom(position.toVec2(Axis.z))

    for collision in collisions:
        block = getCollision(environment, Vec3(collision.x, collision.y, posZ))
        if block:
            return True

    return False


def limit(number: int, minNumber: int, maxNumber: int):
    number = max(number, minNumber)
    return min(number, maxNumber)


def getRectanglePointsAroundPointVec2(point: Vec2, radius: int) -> List[Vec2]:
    return [
        Vec2(point.x + radius, point.y + radius),
        Vec2(point.x + radius, point.y - radius),
        Vec2(point.x - radius, point.y + radius),
        Vec2(point.x - radius, point.y - radius),
    ]


def getRectanglePointsAroundPointVec3(point: Vec3, radius: int, lockedAxis: Axis):
    vec2 = point.toVec2(lockedAxis)
    pointsVec2 = getRectanglePointsAroundPointVec2(vec2, radius)
    return [Vec3.fromVec2(point, lockedAxis, point) for point in pointsVec2]


def getCollision(environment: np.ndarray, pos: Vec3) -> int:
    """
    Returns what block is this point in collision with.
    :param environment: Game environment
    :param pos: Point positions
    """
    posX = limit(math.floor(pos.x), 0, WORLD_SHAPE.x)
    posY = limit(math.floor(pos.y), 0, WORLD_SHAPE.y)
    posZ = limit(math.floor(pos.z), 0, WORLD_SHAPE.z)

    block = environment[posZ][posY][posX]
    return block
