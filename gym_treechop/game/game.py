import math
from math import copysign
from random import random

import numpy as np
from tensorflow.python.keras.utils import to_categorical

from gym_treechop.game.constants import MIN_TREE_HEIGHT, WORLD_SHAPE, MAX_TREE_HEIGHT, Blocks, JUMP_VELOCITY, \
    WALK_VELOCITY, BLOCK_TYPES, BlockHardness, HARDNESS_MULTIPLIER, BREAKING_RANGE
from gym_treechop.game.structures import Vec3, Vec2, Axis
from gym_treechop.game.utils import playerIsStanding


def randNotInCenter(size: int, centerDiameter: int = 1):
    # 0, 1, _2_, 3, __4__, 5, _6_, 7, 8
    center = size // 2
    if centerDiameter > center:
        centerDiameter = center

    move = centerDiameter + random() * (center - centerDiameter - 1)
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

    def getHeadPosition(self) -> Vec3:
        return Vec3(self.position.x, self.position.y, self.position.z + 1)

    def getLookingDirectionVector(self) -> Vec3:
        y = math.sin(self.rotation.x)
        x = math.cos(self.rotation.x)

        z = math.sin(self.rotation.y - math.pi / 2)
        zRot = math.cos(self.rotation.y - math.pi / 2)

        return Vec3(x * zRot, y * zRot, z)


class Game:
    # environment[z, y, x]
    environment: np.ndarray
    player: Player
    center: int = WORLD_SHAPE.x // 2

    oneHotEncodedCache = None
    woodLeftCache = None

    def getEnvironmentOneHotEncoded(self):
        if self.oneHotEncodedCache is None:
            self.oneHotEncodedCache = to_categorical(self.environment.flatten(), num_classes=len(BLOCK_TYPES),
                                                     dtype=np.uint8)
        return self.oneHotEncodedCache

    def getWoodLeft(self):
        if self.woodLeftCache is None:
            self.woodLeftCache = np.count_nonzero(self.environment.flatten() == Blocks.WOOD)
        return self.woodLeftCache

    # region Environment Generation
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

    # endregion

    def isGameOver(self):
        return (WORLD_SHAPE.x < self.player.position.x or self.player.position.x < 0
                or WORLD_SHAPE.y < self.player.position.y or self.player.position.y < 0
                or WORLD_SHAPE.z < self.player.position.z or self.player.position.z < 0)

    def __init__(self, renderer=None) -> None:
        self.renderer = renderer
        self.environment = np.zeros((WORLD_SHAPE.z, WORLD_SHAPE.y, WORLD_SHAPE.x), dtype=np.uint8)
        self.player = Player()

        self.attackedBlockCoords = None
        self.attackTicksRemaining = 0

        self._generateGround()

        treeHeight = self._generateTree()
        if treeHeight == MIN_TREE_HEIGHT:
            self.player.position.x = 7 + random()  # randNotInCenter(WORLD_SHAPE.x, 5)  # 0-maxX, not 5 blocks around center
            self.player.position.y = 7 + random()  # randNotInCenter(WORLD_SHAPE.y, 5)  # 0-maxY, not 5 blocks around center

        # print("Initialized Game")

    # region Player Movement
    def forward(self):
        x = math.cos(self.player.rotation.x)
        y = math.sin(self.player.rotation.x)

        self.player.velocity.x = WALK_VELOCITY * x
        self.player.velocity.y = WALK_VELOCITY * y

    def backward(self):
        x = math.cos(self.player.rotation.x)
        y = math.sin(self.player.rotation.x)

        self.player.velocity.x = - WALK_VELOCITY * x
        self.player.velocity.y = - WALK_VELOCITY * y

    def left(self):
        x = math.cos(self.player.rotation.x - math.pi / 2)
        y = math.sin(self.player.rotation.x - math.pi / 2)

        self.player.velocity.x = WALK_VELOCITY * x
        self.player.velocity.y = WALK_VELOCITY * y

    def right(self):
        x = math.cos(self.player.rotation.x - math.pi / 2)
        y = math.sin(self.player.rotation.x - math.pi / 2)

        self.player.velocity.x = - WALK_VELOCITY * x
        self.player.velocity.y = - WALK_VELOCITY * y

    def jump(self):
        if playerIsStanding(self.player.position, self.environment):
            self.player.velocity.z = JUMP_VELOCITY

    # 0 - 360 degrees -> 0 - 2PI rad
    def lookLeftRight(self, radian: float):
        self.player.rotation.x = radian

    # 0 - 180 degrees -> 0 - 1PI rad
    def lookUpDown(self, radian: float):
        self.player.rotation.y = radian

    # endregion

    # region Breaking Blocks
    attackedBlockCoords: Vec3 or None  # Which block is being attacked
    attackTicksRemaining: int  # How long is the block being attacked in Ticks

    # Returns destroyed block id (AIR(0) if none)
    def attackBlock(self, delta: float) -> int:
        block, coords = self.getBlockInFrontOfPlayer()
        if coords != self.attackedBlockCoords:
            # Currently attacked block is no longer attacked
            self.stopBlockAttack()
        if block:
            if not self.attackedBlockCoords:
                self.attackedBlockCoords = coords
                self.attackTicksRemaining = BlockHardness[block] * HARDNESS_MULTIPLIER * 20

            # Some block is attacked
            attackStrength = 20 * delta  # 20ticks per second
            if not playerIsStanding(self.player.position, self.environment):
                attackStrength /= 5

            self.attackTicksRemaining -= attackStrength

            # print(f"self.attackTicksRemaining: ", self.attackTicksRemaining)

            if self.attackTicksRemaining <= 0:
                self._setBlock(coords, 0)
                print("Destroyed Coords: ", coords)
                return block

    def stopBlockAttack(self):
        self.attackedBlockCoords = None

    def getBlockInFrontOfPlayer(self, zPos: float = 1, lookingRange: float = BREAKING_RANGE) -> (int, Vec3):
        # position = self.player.getHeadPosition()
        position = Vec3(self.player.position.x, self.player.position.y, self.player.position.z + zPos)
        direction = self.player.getLookingDirectionVector()

        blockPos = self.__getNextBlock(position, direction)
        while True:
            if not self._isInEnvironment(blockPos):
                return 0, None

            if position.getLengthTo(blockPos) > lookingRange:
                return 0, None

            if self._getBlock(blockPos) != 0:
                # self._setBlock(blockPos, 0)  # DEBUG
                return self._getBlock(blockPos), blockPos.floor()

            # self._setBlock(blockPos, Blocks.WOOD) # DEBUG
            # self.renderer.render(self) # DEBUG
            blockPos = self.__getNextBlock(blockPos, direction)

    @staticmethod
    def __getNextBlock(position: Vec3, direction: Vec3) -> Vec3:
        toFullX = copysign((copysign(1 - (position.x % 1), direction.x) % 1), direction.x)
        toFullY = copysign((copysign(1 - (position.y % 1), direction.y) % 1), direction.y)
        toFullZ = copysign((copysign(1 - (position.z % 1), direction.z) % 1), direction.z)

        toFullX = toFullX or copysign(1, toFullX)
        toFullY = toFullY or copysign(1, toFullY)
        toFullZ = toFullZ or copysign(1, toFullZ)

        relativeX = toFullX / (direction.x or 0.00000000001)
        relativeY = toFullY / (direction.y or 0.00000000001)
        relativeZ = toFullZ / (direction.z or 0.00000000001)

        relativeMove = min(relativeX, relativeY, relativeZ)

        moved = position.copy()
        moved.x += direction.x * relativeMove
        moved.y += direction.y * relativeMove
        moved.z += direction.z * relativeMove

        if relativeMove == relativeX:
            moved.x = float(round(moved.x))
        if relativeMove == relativeY:
            moved.y = float(round(moved.y))
        if relativeMove == relativeZ:
            moved.z = float(round(moved.z))

        return moved

    # endregion

    def _isInEnvironment(self, pos: Vec3):
        pos = pos.floor()
        if (WORLD_SHAPE.x <= pos.x or pos.x < 0
                or WORLD_SHAPE.y <= pos.y or pos.y < 0
                or WORLD_SHAPE.z <= pos.z or pos.z < 0):
            return False
        else:
            return True

    def _getBlock(self, pos: Vec3) -> int:
        pos = pos.floor()
        if self._isInEnvironment(pos):
            return self.environment[pos.z, pos.y, pos.x]
        else:
            return 0

    def _setBlock(self, pos: Vec3, block: int) -> bool:
        pos = pos.floor()
        if self._isInEnvironment(pos):
            if self.environment[pos.z, pos.y, pos.x] != block:
                self.environment[pos.z, pos.y, pos.x] = block
                self.oneHotEncodedCache = None
                if block == Blocks.WOOD:
                    self.woodLeftCache = None
            return True
        else:
            return False

    def getPlayerDistanceToCenter(self) -> float:
        center = Vec2(self.center, self.center)
        toCenter = self.player.position.toVec2(Axis.z).getLengthTo(center)
        return toCenter
