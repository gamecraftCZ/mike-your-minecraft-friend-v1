import math
from typing import List

import numpy as np

from constants import WORLD_SHAPE, PLAYER_RADIUS, Blocks
from structures import Vec2, Vec3, Axis


def getCollisionsBottom(pos: Vec2) -> List[Vec2]:
    points = getRectanglePointsAroundPointVec2(pos, PLAYER_RADIUS, PLAYER_RADIUS)
    points = [point.floor() for point in points if 0 <= point.x < WORLD_SHAPE.x]
    points = [point.floor() for point in points if 0 <= point.y < WORLD_SHAPE.y]
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


def getRectanglePointsAroundPointVec2(point: Vec2, radiusX: int, radiusY) -> List[Vec2]:
    # We use 0.00001 to prevent bad errors to happen. eg. 0.7 + 0.3 -> 1 (next block)
    return [
        Vec2(point.x + radiusX - 0.00001, point.y + radiusY - 0.00001),
        Vec2(point.x + radiusX - 0.00001, point.y - radiusY + 0.00001),
        Vec2(point.x - radiusX + 0.00001, point.y + radiusY - 0.00001),
        Vec2(point.x - radiusX + 0.00001, point.y - radiusY + 0.00001),
    ]


def getRectanglePointsAroundPointVec3(point: Vec3, radiusX: int, radiusY: int, lockedAxis: Axis):
    vec2 = point.toVec2(lockedAxis)
    pointsVec2 = getRectanglePointsAroundPointVec2(vec2, radiusX, radiusY)
    return [Vec3.fromVec2(p, lockedAxis, point) for p in pointsVec2]


def getCollision(environment: np.ndarray, pos: Vec3) -> int:
    """
    Returns what block is this point in collision with.
    :param environment: Game environment
    :param pos: Point positions
    """
    posX = math.floor(pos.x)
    posY = math.floor(pos.y)
    posZ = math.floor(pos.z)

    if 0 <= posX < WORLD_SHAPE.x and 0 <= posY < WORLD_SHAPE.y and 0 <= posZ < WORLD_SHAPE.z:
        # Block is in environment
        return environment[posZ][posY][posX]

    return Blocks.AIR


def nthRoot(number: float, exponent: float) -> float:
    num = abs(number)
    result = num ** (1 / float(exponent))
    return math.copysign(result, number)
