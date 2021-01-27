import math
from math import copysign
from random import random, randint
from typing import List

import numpy as np
from numba import jit
from tensorflow.python.keras.utils import to_categorical

from gym_treechop.game.constants import MIN_TREE_HEIGHT, WORLD_SHAPE, MAX_TREE_HEIGHT, Blocks, JUMP_VELOCITY, \
    WALK_VELOCITY, BLOCK_TYPES, BlockHardness, HARDNESS_MULTIPLIER, BREAKING_RANGE, WORLD_SHAPE_TUPLE
from gym_treechop.game.structures import Vec3, Vec2, Axis
from gym_treechop.game.utils import playerIsStanding, getDistance


##### NUMBA functions #####
@jit(nopython=True)
def numba_getBlock(environment: np.ndarray, pos: (float, float, float)) -> int:
    pos = (math.floor(pos[0]), math.floor(pos[1]), math.floor(pos[2]))
    if numba_isInEnvironment(pos):
        return environment[pos[2], pos[1], pos[0]]
    else:
        return 0


@jit(nopython=True)
def numba_isInEnvironment(pos: (float, float, float)):
    posX = math.floor(pos[0])
    posY = math.floor(pos[1])
    posZ = math.floor(pos[2])
    if (WORLD_SHAPE_TUPLE[0] <= posX or posX < 0
            or WORLD_SHAPE_TUPLE[1] <= posY or posY < 0
            or WORLD_SHAPE_TUPLE[2] <= posZ or posZ < 0):
        return False
    else:
        return True


@jit(nopython=True)
def numba_getNextBlock(pos: (float, float, float),
                       vec: (float, float, float)) -> ((float, float, float), (float, float, float)):
    toFullX = copysign((copysign(1 - (pos[0] % 1), vec[0]) % 1), vec[0])
    toFullY = copysign((copysign(1 - (pos[1] % 1), vec[1]) % 1), vec[1])
    toFullZ = copysign((copysign(1 - (pos[2] % 1), vec[2]) % 1), vec[2])

    toFullX = toFullX or copysign(1, toFullX)
    toFullY = toFullY or copysign(1, toFullY)
    toFullZ = toFullZ or copysign(1, toFullZ)

    relativeX = toFullX / (vec[0] or 0.00000000001)
    relativeY = toFullY / (vec[1] or 0.00000000001)
    relativeZ = toFullZ / (vec[2] or 0.00000000001)

    relativeMove = min(relativeX, relativeY, relativeZ)

    movedX = pos[0] + vec[0] * relativeMove
    movedY = pos[1] + vec[1] * relativeMove
    movedZ = pos[2] + vec[2] * relativeMove

    rayX, rayY, rayZ = movedX, movedY, movedZ

    if relativeMove == relativeX:
        movedX = float(round(movedX))
    if relativeMove == relativeY:
        movedY = float(round(movedY))
    if relativeMove == relativeZ:
        movedZ = float(round(movedZ))

    return (movedX, movedY, movedZ), (rayX, rayY, rayZ)


@jit(nopython=True)
def numba_getBlockDistance(pos: (float, float, float), vec: (float, float, float),
                           maxDistance: float, environment: np.ndarray) -> float:
    blockPos, rayPos = numba_getNextBlock(pos, vec)
    while True:
        distanceToBlock = getDistance(pos, rayPos)
        if not numba_isInEnvironment(blockPos):
            return maxDistance

        if distanceToBlock > maxDistance:
            return maxDistance

        if numba_getBlock(environment, blockPos) != 0:
            return distanceToBlock

        blockPos, rayPos = numba_getNextBlock(blockPos, vec)


##### REST of the CODE #####

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

        vec = Vec3(x * zRot, y * zRot, z)
        return vec.normalize()

    def getLookingDirectionVector2d(self) -> Vec2:
        y = math.sin(self.rotation.x)
        x = math.cos(self.rotation.x)
        return Vec2(x, y).normalize()


class Game:
    # environment[z, y, x]
    environment: np.ndarray
    player: Player
    center: int = WORLD_SHAPE.x // 2

    oneHotEncodedCache = None
    woodLeftCache = None
    woodBlocksCache = None

    def getEnvironmentOneHotEncoded(self):
        if self.oneHotEncodedCache is None:
            self.oneHotEncodedCache = to_categorical(self.environment.flatten(), num_classes=len(BLOCK_TYPES),
                                                     dtype=np.uint8)
        return self.oneHotEncodedCache

    def getWoodLeft(self):
        if self.woodLeftCache is None:
            self.woodLeftCache = np.count_nonzero(self.environment.flatten() == Blocks.WOOD)
        return self.woodLeftCache

    # Return if there is block on each layer of environment
    def getWoodBlocks(self) -> List[bool]:
        if not self.woodBlocksCache:
            self.woodBlocksCache = []
            for z in range(WORLD_SHAPE.z):
                if np.count_nonzero(self.environment[z].flatten() == Blocks.WOOD):
                    self.woodBlocksCache.append(True)
                else:
                    self.woodBlocksCache.append(False)
        return self.woodBlocksCache

    # region Environment Generation
    def _generateGround(self):
        for x in range(WORLD_SHAPE.x):
            for y in range(WORLD_SHAPE.y):
                height = int(random() * 2)  # 0 or 1
                self.environment[0][y][x] = Blocks.GROUND
                self.environment[height][y][x] = Blocks.GROUND

    def _generateTree(self, tree_blocks_to_generate=6):
        height = randint(MIN_TREE_HEIGHT, MAX_TREE_HEIGHT)

        # generate tree from top
        for i in range(height, height - tree_blocks_to_generate, -1):
            self.environment[i][self.center][self.center] = Blocks.WOOD

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

    def __init__(self, renderer=None, tree_blocks_to_generate=6) -> None:
        self.environment = np.zeros((WORLD_SHAPE.z, WORLD_SHAPE.y, WORLD_SHAPE.x), dtype=np.uint8)
        self.player = Player()
        self.renderer = renderer
        self._generateGround()

        self.attackedBlockCoords = None
        self.attackTicksRemaining = 0

        treeHeight = self._generateTree(tree_blocks_to_generate)
        if tree_blocks_to_generate > 4:
            self.player.position.x = randNotInCenter(WORLD_SHAPE.x, 5)  # 0-maxX, not in center where the tree could be
            self.player.position.y = randNotInCenter(WORLD_SHAPE.y, 5)  # 0-maxY, not in center where the tree could be

        # print("Initialized Game")
        print(f"----------- {self.getWoodLeft()} wood ---------------")

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
        self.player.rotation.x = (radian + 100 * math.pi) % math.tau

    # 0 - 180 degrees -> 0 - 1PI rad
    def lookUpDown(self, radian: float):
        self.player.rotation.y = max(0, min(math.pi, radian))

    # endregion

    # region Breaking Blocks
    attackedBlockCoords: Vec3 or None  # Which block is being attacked
    attackTicksRemaining: int  # How long is the block being attacked in Ticks

    # Returns destroyed block id, None otherwise
    def attackBlock(self, delta: float) -> int:
        block, coords = self.getBlockInFrontOfPlayer()
        if coords != self.attackedBlockCoords:
            # Currently attacked block is no longer attacked
            self.stopBlockAttack()
        if block:
            if not self.attackedBlockCoords:
                self.attackedBlockCoords = coords
                self.attackTicksRemaining = BlockHardness[
                                                block] * HARDNESS_MULTIPLIER * 20 - 1e-6  # for rounding errors

            # Some block is attacked
            attackStrength = 20 * delta  # 20ticks per second
            if not playerIsStanding(self.player.position, self.environment):
                attackStrength /= 5

            self.attackTicksRemaining -= attackStrength

            # print(f"self.attackTicksRemaining: ", self.attackTicksRemaining)

            if self.attackTicksRemaining <= 0:
                self._setBlock(coords, Blocks.AIR)
                print("Destroyed Coords: ", coords)
                return block

    def stopBlockAttack(self):
        self.attackedBlockCoords = None
        self.attackTicksRemaining = 0

    def getBlockInFrontOfPlayer(self, zPos: float = 1, lookingRange: float = BREAKING_RANGE,
                                direction: Vec3 = None) -> (int, Vec3):
        # position = self.player.getHeadPosition()
        position = Vec3(self.player.position.x, self.player.position.y, self.player.position.z + zPos)
        if not direction:
            direction = self.player.getLookingDirectionVector()

        blockPos, _ = self.__getNextBlock(position, direction)
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
            blockPos, _ = self.__getNextBlock(blockPos, direction)

    def getBlockDistance(self, position: Vec3 = None, vector: Vec3 = None, maxDistance: float = 8) -> float:
        if not position:
            position = self.player.getHeadPosition()
        if not vector:
            vector = self.player.getLookingDirectionVector()

        vector = vector.normalize()

        return numba_getBlockDistance(position.asTuple(), vector.asTuple(), maxDistance, self.environment)

    @staticmethod
    # returns (blockPosition, rayCollisionWithTheBLock)
    def __getNextBlock(pos: Vec3, direct: Vec3) -> (Vec3, Vec3):
        blockPos, rayPos = numba_getNextBlock(pos.asTuple(), direct.asTuple())
        return Vec3.fromTuple(blockPos), Vec3.fromTuple(rayPos)

    # endregion

    def getNextWoodBlock(self) -> Vec3:
        for z in range(WORLD_SHAPE.z):
            if np.count_nonzero(self.environment[z].flatten() == Blocks.WOOD):
                return Vec3(self.center, self.center, float(z))
        return Vec3(-1, -1, -1)

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
                self.woodBlocksCache = None
                self.woodLeftCache = None
            return True
        else:
            return False

    def getPlayerDistanceToCenter(self) -> float:
        center = Vec2(self.center, self.center)
        toCenter = self.player.position.toVec2(Axis.z).getLengthTo(center)
        return toCenter
