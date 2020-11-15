from math import copysign

from constants import GRAVITY, TERMINAL_VELOCITY
from game import Game
from structures import Vec2, Vec3
from utils import getCollisionsBottom, playerIsStanding, getCollision


class Physics:
    # Do 0.1tick step
    @staticmethod
    def step(game: Game, delta: int = 0.1):
        Physics._resolveGravity(game, delta)
        Physics._resolveMovement(game, delta)
        Physics._slowDownXYVelocity(game, delta)

    @staticmethod
    def _resolveGravity(game: Game, delta: int):
        ## region # Resolve velocity #
        game.player.velocity.z -= GRAVITY * delta
        game.player.velocity.z *= 1 - (0.02 * delta)  # 0.98 -> https://www.mcpk.wiki/wiki/Vertical_Movement_Formulas
        if abs(game.player.velocity.z) > TERMINAL_VELOCITY:
            game.player.velocity.z = copysign(TERMINAL_VELOCITY, game.player.velocity.z)
        ## endregion # Resolve velocity #

        posX = game.player.position.x
        posY = game.player.position.y
        posZ = game.player.position.z

        newZ = game.player.position.z + game.player.velocity.z * delta

        collisions = getCollisionsBottom(Vec2(posX, posY))

        z = int(newZ)
        if z < 0:
            z = 0
        for collision in collisions:
            block = getCollision(game.environment, Vec3(collision.x, collision.y, z))
            if block:
                if int(posZ) > int(newZ):  # Only prevent falling into the block
                    newZ = z + 1
                game.player.velocity.z = 0

        game.player.position.z = newZ

    @staticmethod
    def _resolveMovement(game: Game, delta: int):
        game.player.position.x += game.player.velocity.x * delta
        game.player.position.y += game.player.velocity.y * delta
        # TODO

    @staticmethod
    def _slowDownXYVelocity(game: Game, delta: int):
        if playerIsStanding(game.player.position, game.environment):
            print("TODO - _slowDownXYVelocity")
            pass
