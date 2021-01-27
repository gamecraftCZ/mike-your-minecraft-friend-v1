import math
from enum import Enum

import numpy as np
from numba import jit


class Axis(Enum):
    x = 0
    y = 1
    z = 2


class Vec2:
    x: float = 0
    y: float = 0

    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def copy(self) -> 'Vec2':
        return Vec2(self.x, self.y)

    def floor(self) -> 'Vec2':
        return Vec2(math.floor(self.x), math.floor(self.y))

    def __str__(self):
        return f"(x: {self.x / 1.0:.3}, y: {self.y / 1.0:.3})"

    def toNumpy(self) -> np.array:
        return np.array([self.x, self.y], dtype=np.float32)

    def getLengthTo(self, point: 'Vec2') -> float:
        lengthX = self.x - point.x
        lengthY = self.y - point.y
        return math.sqrt(lengthX ** 2 + lengthY ** 2)

    def __eq__(self, other: 'Vec2'):
        if isinstance(other, Vec2):
            return self.x == other.x and self.y == other.y
        else:
            return False

    def __truediv__(self, other: 'Vec2' or float):
        if isinstance(other, Vec2):
            # Vec2 / Vec2
            return Vec2(self.x / other.x, self.y / other.y)
        else:
            # Vec2 / float
            return Vec2(self.x / other, self.y / other)

    def normalize(self) -> 'Vec2':
        # Normalize vector to has one of the directions == 1 or -1
        m = max(abs(self.x), abs(self.y))
        if m:
            return self / m
        else:
            return Vec2(1, 1)


class Vec3:
    x: float = 0
    y: float = 0
    z: float = 0  # Z coordinate is elevation (up/down)

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def toVec2(self, ignoredAxis: Axis) -> Vec2:
        if ignoredAxis == Axis.x:
            return Vec2(self.y, self.z)
        if ignoredAxis == Axis.y:
            return Vec2(self.x, self.z)
        if ignoredAxis == Axis.z:
            return Vec2(self.x, self.y)

    @staticmethod
    def fromVec2(vec2: Vec2, ignoredAxis: Axis = Axis.z, ignoredValue: int or 'Vec3' = 0) -> 'Vec3':
        if isinstance(ignoredValue, Vec3):
            if ignoredAxis == Axis.x:
                ignoredValue = ignoredValue.x
            if ignoredAxis == Axis.y:
                ignoredValue = ignoredValue.y
            if ignoredAxis == Axis.z:
                ignoredValue = ignoredValue.z

        if ignoredAxis == Axis.x:
            return Vec3(ignoredValue, vec2.x, vec2.y)
        if ignoredAxis == Axis.y:
            return Vec3(vec2.x, ignoredValue, vec2.y)
        if ignoredAxis == Axis.z:
            return Vec3(vec2.x, vec2.y, ignoredValue)

    def copy(self) -> 'Vec3':
        return Vec3(self.x, self.y, self.z)

    def floor(self) -> 'Vec3':
        return Vec3(math.floor(self.x), math.floor(self.y), math.floor(self.z))

    def round(self) -> 'Vec3':
        return Vec3(round(self.x), round(self.y), round(self.z))

    def __str__(self):
        return f"(x: {self.x / 1.0:.3}, y: {self.y / 1.0:.3}, z: {self.z / 1.0:.3})"

    def toNumpy(self) -> np.array:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def getLengthTo(self, point: 'Vec3') -> float:
        lengthX = self.x - point.x
        lengthY = self.y - point.y
        lengthZ = self.z - point.z
        return math.sqrt(lengthX ** 2 + lengthY ** 2 + lengthZ ** 2)

    def __eq__(self, other: 'Vec3'):
        if isinstance(other, Vec3):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return False

    def __truediv__(self, other: 'Vec3' or float):
        if isinstance(other, Vec3):
            # Vec3 / Vec3
            return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            # Vec3 / float
            return Vec3(self.x / other, self.y / other, self.z / other)

    def normalize(self) -> 'Vec3':
        # Normalize vector to has one of the directions == 1
        m = max(abs(self.x), abs(self.y), abs(self.z))
        if m:
            return self / m
        else:
            return Vec3(1, 1, 1)

    def length(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    @staticmethod
    @jit(nopython=True)
    def __numba_rotate(vX, vY, vZ, upDown: float, leftRight: float) -> (float, float, float):
        l = math.sqrt(vX ** 2 + vY ** 2 + vZ ** 2)
        if l == 0:
            return 0, 0, 0

        x = abs(vX)
        y = abs(vY)
        z = abs(vZ)

        alpha = math.atan(y / (x or 0.000000000000001))
        beta = math.asin(z / l)

        if vX > 0 and vY > 0:
            pass
        if vX < 0 and vY > 0:
            alpha = math.pi - alpha
        if vX < 0 and vY < 0:
            alpha += math.pi
        if vX > 0 and vY < 0:
            alpha = math.tau - alpha

        if vZ < 0:
            beta = -beta

        # Add angles
        beta = math.copysign(abs(beta + upDown) % math.pi, (beta + upDown))

        flip = False
        if beta > math.pi / 2 or beta < -math.pi / 2:
            flip = True
            if beta > 0:
                beta = math.pi - beta
            else:
                beta = -math.pi - beta
            alpha += math.pi

        alpha = (alpha + leftRight) % math.tau

        # Convert back to vector
        outZ = l * math.sin(beta)
        m = math.sqrt(l ** 2 - outZ ** 2)
        outY = m * math.sin(alpha)
        outX = math.sqrt(m ** 2 - outY ** 2)

        outX = math.copysign(outX, vX)
        outY = math.copysign(outY, vY)
        if flip:
            outX = -outX
            outY = -outY

        outX = abs(outX)
        if math.pi / 2 < alpha < math.pi / 2 * 3:
            outX = -outX
        outY = abs(outY)
        if alpha > math.pi:
            outY = -outY

        return outX, outY, outZ

    # rotate tables for testing:
    # upDown, leftRight - in radians
    def rotate(self, upDown: float, leftRight: float) -> 'Vec3':
        x, y, z = self.__numba_rotate(self.x, self.y, self.z, upDown, leftRight)
        return Vec3(x, y, z)
