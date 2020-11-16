import math
from enum import Enum

import numpy as np


class Axis(Enum):
    x = 0
    y = 1
    z = 2


class Vec2:
    x: int or float = 0
    y: int or float = 0

    def __init__(self, x=0, y=0):
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

    def __eq__(self, other: 'Vec2'):
        if isinstance(other, Vec2):
            return self.x == other.x and self.y == other.y
        else:
            return False

class Vec3:
    x: int or float = 0
    y: int or float = 0
    z: int or float = 0  # Z coordinate is elevation (up/down)

    def __init__(self, x=0, y=0, z=0):
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
    def fromVec2(vec2: Vec2, ignoredAxis: Axis, ignoredValue: int or 'Vec3') -> 'Vec3':
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

    def getLengthTo(self, point: 'Vec3'):
        lengthX = self.x - point.x
        lengthY = self.y - point.y
        lengthZ = self.z - point.z
        return math.sqrt(lengthX ** 2 + lengthY ** 2 + lengthZ ** 2)

    def __eq__(self, other: 'Vec3'):
        if isinstance(other, Vec3):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return False
