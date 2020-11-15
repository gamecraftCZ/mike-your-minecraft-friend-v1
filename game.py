import math
from random import random

import numpy as np

from constants import MIN_TREE_HEIGHT, WORLD_SHAPE, MAX_TREE_HEIGHT, Blocks, JUMP_VELOCITY, WALK_VELOCITY
from structures import Vec3, Vec2
from utils import playerIsStanding


def randNotInCenter(size: int, centerDiameter: int = 1):
    # 0, 1, _2_, 3, __4__, 5, _6_, 7, 8
    center = size // 2
    if centerDiameter > center:
        centerDiameter = center

    move = centerDiameter + random() * (center - centerDiameter)
    if random() > 0.5:
        move = -move

    return center + move


class Player:
    position: Vec3 = Vec3()
    rotation: Vec2 = Vec2()  # X -> left/right (0-2PI), Y -> up/down (0(down) - 1PI(up))
    velocity: Vec3 = Vec3()

    def __init__(self):
        self.position.x = randNotInCenter(WORLD_SHAPE.x)  # 0-maxX, not in center
        self.position.y = randNotInCenter(WORLD_SHAPE.y)  # 0-maxY, not in center
        self.position.z = 2

        self.rotation.x = random() * 2 * math.pi
        self.rotation.y = random() * math.pi


class Game:
    # environment[z, y, x]
    environment: np.ndarray = np.zeros((WORLD_SHAPE.z, WORLD_SHAPE.y, WORLD_SHAPE.x), dtype=np.uint8)
    player: Player = Player()
    center: int = WORLD_SHAPE.x // 2

    def getWoodLeft(self):
        return np.count_nonzero(self.environment.flatten() == Blocks.WOOD)

    def _generateGround(self):
        for x in range(WORLD_SHAPE.x):
            for y in range(WORLD_SHAPE.y):
                height = int(random() * 2)  # 0 or 1
                self.environment[0][y][x] = Blocks.GROUND
                self.environment[height][y][x] = Blocks.GROUND

    def _generateTree(self):
        height = MIN_TREE_HEIGHT + int(
            random() * (1 + MAX_TREE_HEIGHT - MIN_TREE_HEIGHT))  # minTreeHeight - maxTreeHeight
        for i in range(height):
            self.environment[1 + i][self.center][self.center] = Blocks.WOOD

        self.__generateLeafs(1, height)
        for h in range(height - 1, height - 3, -1):
            self.__generateLeafs(2, h)

        return height

    def __generateLeafs(self, radius: int, height: int):
        for y in range(0, radius + 1):
            for x in range(0, radius + 1):
                if x == y == 0:
                    continue

                self.environment[height][self.center + y][self.center + x] = Blocks.LEAF
                self.environment[height][self.center + y][self.center - x] = Blocks.LEAF

                self.environment[height][self.center - y][self.center + x] = Blocks.LEAF
                self.environment[height][self.center - y][self.center - x] = Blocks.LEAF

    def __init__(self) -> None:
        self._generateGround()

        treeHeight = self._generateTree()
        if treeHeight == MIN_TREE_HEIGHT:
            self.player.position.x = randNotInCenter(WORLD_SHAPE.x, 5)  # 0-maxX, not 5 blocks around center
            self.player.position.y = randNotInCenter(WORLD_SHAPE.y, 5)  # 0-maxY, not 5 blocks around center

        print("Initialized Game")

    # TODO account for player rotation in movement!

    def forward(self):
        self.player.velocity.x = -WALK_VELOCITY

    def backward(self):
        self.player.velocity.x = WALK_VELOCITY

    def left(self):
        self.player.velocity.y = WALK_VELOCITY

    def right(self):
        self.player.velocity.y = -WALK_VELOCITY

    def jump(self):
        if playerIsStanding(self.player.position, self.environment):
            self.player.velocity.z = JUMP_VELOCITY

    # 0 - 360 degrees -> 0 - 2PI rad
    def lookLeftRight(self, radian: float):
        self.player.rotation.x = radian

    # 0 - 360 degrees -> 0 - 2PI rad
    def lookUpDown(self, radian: float):
        self.player.rotation.y = radian
