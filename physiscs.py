import math
from math import copysign
from typing import Tuple, List, Union

from constants import GRAVITY, TERMINAL_VELOCITY, WORLD_SHAPE
from game import Game
from structures import Vec2


def getCollisionsBottom(posX, posY) -> List[Vec2]:
    relPosX = posX % 1
    relPosY = posY % 1

    X = []

    if posX <= WORLD_SHAPE.x - 0.5:
        X.append(math.floor(posX + 0.5))
    if relPosX < 0.3 or relPosX > 0.7:
        # Collision with 2nd block on X axis
        if posX >= 0.5:
            X.append(math.floor(posX + 0.5) - 1)

    Y = []

    if posY <= WORLD_SHAPE.y - 0.5:
        Y.append(math.floor(posY + 0.5))
    if relPosY < 0.3 or relPosY > 0.7:
        # Collision with 2nd block on Y axis
        if posY >= 0.5:
            Y.append(math.floor(posY + 0.5) - 1)

    return [Vec2(x, y) for x in X for y in Y]


class Physics:
    # Do 0.1tick step
    def step(self, game: Game):
        self.resolveGravity(game)

    def resolveGravity(self, game: Game):
        game.player.velocity.z -= GRAVITY
        if abs(game.player.velocity.z) > TERMINAL_VELOCITY:
            game.player.velocity.z = copysign(TERMINAL_VELOCITY, game.player.velocity.z)

        posX = game.player.position.x
        posY = game.player.position.y

        newZ = game.player.position.z + game.player.velocity.z

        collisions = getCollisionsBottom(posX, posY)

        z = int(newZ)
        if z < 0:
            z = 0
        for collision in collisions:
            block = game.environment[z][collision.y][collision.x]
            if block:
                newZ = z + 1
                game.player.velocity.z = 0

        game.player.position.z = newZ
