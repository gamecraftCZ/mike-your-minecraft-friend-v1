from random import random
from typing import List

import numpy as np


class Blocks:
    AIR = 0
    GROUND = 1
    WOOD = 2
    LEAF = 3


class Vec3:
    x: int or float = 0
    y: int or float = 0
    z: int or float = 0

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


def randNotInCenter(size: int):
    # 0, 1, _2_, 3, __4__, 5, _6_, 7, 8
    center = size // 2

    move = 1 + random() * (center - 1)
    if random() > 0.5:
        move = -move

    return center + move


class Player:
    position: Vec3 = Vec3()
    rotation: Vec3 = Vec3()
    velocity: Vec3 = Vec3()

    def __init__(self):
        self.position.x = randNotInCenter(9)  # 0-8
        self.position.y = randNotInCenter(9)  # 0-8
        self.position.z = 2

        self.rotation.x = random()
        self.rotation.y = random()
        self.rotation.z = random()


MIN_TREE_HEIGHT = 5
MAX_TREE_HEIGHT = 7
CENTER = 4


class Game:
    environment: np.ndarray = np.zeros((9, 9, 9),  # environment[z, y, x]
                                       dtype=np.uint8)  # Environment is 9x9x9 blocks -> 729 blocks in total
    player: Player = Player()

    def getWoodLeft(self):
        unique, counts = self.environment.flatten().unique(return_counts=True)
        return dict(zip(unique, counts))[Blocks.WOOD]

    def createTree(self):
        height = MIN_TREE_HEIGHT + int(random() * (1 + MAX_TREE_HEIGHT - MIN_TREE_HEIGHT))  # 5-7
        for i in range(height):
            self.environment[1 + i][CENTER][CENTER] = Blocks.WOOD

        self.generateLeafs(1, height)
        for h in range(height - 1, height - 3, -1):
            self.generateLeafs(2, h)

    def generateLeafs(self, diameter: int, height: int):
        for y in range(1, diameter):
            for x in range(1, diameter):
                self.environment[height][CENTER + y][CENTER + x] = Blocks.LEAF
                self.environment[height][CENTER - y][CENTER - x] = Blocks.LEAF

                self.environment[height][CENTER + x][CENTER + y] = Blocks.LEAF
                self.environment[height][CENTER - x][CENTER - y] = Blocks.LEAF

    def step(self, delta: float):
        print(f"Running next frame step with delta {delta}s")

    def __init__(self) -> None:
        for x in range(9):
            for y in range(9):
                height = int(random() * 2)  # 0 or 1
                self.environment[0][y][x] = Blocks.GROUND
                self.environment[height][y][x] = Blocks.GROUND

        self.createTree()
        print("Initialized Game")
