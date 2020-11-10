import math
from math import copysign
from typing import Tuple, List, Union

from constants import GRAVITY, TERMINAL_VELOCITY, WORLD_SHAPE
from game import Game
from structures import Vec2
from utils import getCollisionsBottom, isStanding


class Physics:
    # Do 0.1tick step
    def step(self, game: Game, delta: int = 0.1):
        self._resolveGravity(game, delta)
        self._resolveMovement(game, delta)
        self._slowDownXYVelocity(game, delta)

    def _resolveGravity(self, game: Game, delta: int):
        game.player.velocity.z -= GRAVITY * delta
        game.player.velocity.z *= 1 - (0.02 * delta)  # 0.98 -> https://www.mcpk.wiki/wiki/Vertical_Movement_Formulas
        if abs(game.player.velocity.z) > TERMINAL_VELOCITY:
            game.player.velocity.z = copysign(TERMINAL_VELOCITY, game.player.velocity.z)

        posX = game.player.position.x
        posY = game.player.position.y
        posZ = game.player.position.z

        newZ = game.player.position.z + game.player.velocity.z * delta

        collisions = getCollisionsBottom(posX, posY)

        z = int(newZ)
        if z < 0:
            z = 0
        for collision in collisions:
            block = game.environment[z][collision.y][collision.x]
            if block:
                if int(posZ) > int(newZ):  # Only prevent falling into the block
                    newZ = z + 1
                game.player.velocity.z = 0

        game.player.position.z = newZ

    def _resolveMovement(self, game: Game, delta: int):
        game.player.position.x += game.player.velocity.x * delta
        game.player.position.y += game.player.velocity.y * delta
        # TODO

    def _slowDownXYVelocity(self, game: Game, delta: int):
        if isStanding(game.player.position, game.environment):
            pass
