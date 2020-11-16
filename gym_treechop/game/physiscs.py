import math
from math import copysign

from gym_treechop.game.constants import GRAVITY, TERMINAL_VELOCITY, PLAYER_RADIUS, PLAYER_HEIGHT, WORLD_SHAPE
from gym_treechop.game.game import Game
from gym_treechop.game.structures import Vec3, Axis
from gym_treechop.game.utils import getCollisionsBottom, playerIsStanding, getCollision, \
    getRectanglePointsAroundPointVec3, nthRoot


class Physics:
    # Do 0.1tick step
    @staticmethod
    def step(game: Game, delta: float = 0.1):  # Delta larger than 0.1 causes problems!
        Physics._resolveGravity(game, delta)
        Physics._resolveMovement(game, delta)
        Physics._slowDownXYVelocity(game, delta)

    @staticmethod
    def _resolveGravity(game: Game, delta: float):
        ## region # Resolve gravity acceleration #
        game.player.velocity.z -= GRAVITY * delta
        game.player.velocity.z *= nthRoot(0.98,
                                          1 / delta)  # 0.98 -> https://www.mcpk.wiki/wiki/Vertical_Movement_Formulas
        if abs(game.player.velocity.z) > TERMINAL_VELOCITY:
            game.player.velocity.z = copysign(TERMINAL_VELOCITY, game.player.velocity.z)
        ## endregion #                           #

        posZ = game.player.position.z
        newZ = game.player.position.z + game.player.velocity.z * delta

        collisions = getCollisionsBottom(game.player.position.toVec2(Axis.z))

        zBottom = int(newZ)
        if zBottom < 0:
            zBottom = 0

        zTop = int(newZ + PLAYER_HEIGHT)
        if zTop >= WORLD_SHAPE.z:
            zTop = WORLD_SHAPE.z - 1

        for collision in collisions:
            # Not falling thru floor
            blockBottom = getCollision(game.environment, Vec3(collision.x, collision.y, zBottom))
            if blockBottom:
                if int(posZ) > int(newZ):  # Only prevents falling into the block
                    newZ = zBottom + 1
                game.player.velocity.z = 0

            # Hitting head to ceiling detection
            blockTop = getCollision(game.environment, Vec3(collision.x, collision.y, zTop))
            if blockTop:
                # print("COLLISION top")
                newZ = zTop - PLAYER_HEIGHT - 0.00001  # Because float is not accurate enough
                game.player.velocity.z = 0

        game.player.position.z = newZ

    @staticmethod
    def _resolveMovement(game: Game, delta: float):
        ## Resolve X velocity
        game.player.position.x += game.player.velocity.x * delta

        if game.player.velocity.x < 0:
            minusXPos = game.player.position.copy()
            minusXPos.x -= PLAYER_RADIUS
            minusXPos.z += PLAYER_HEIGHT / 2 + 0.000001  # Because float is not accurate enough
            pointsMinusX = getRectanglePointsAroundPointVec3(minusXPos, PLAYER_RADIUS, PLAYER_HEIGHT / 2, Axis.x)
            for point in pointsMinusX:
                if getCollision(game.environment, point):
                    # print("COLLISION front")
                    game.player.position.x = math.floor(point.x) + 1 + PLAYER_RADIUS
                    game.player.velocity.x = 0
                    break
        elif game.player.velocity.x > 0:
            plusXPos = game.player.position.copy()
            plusXPos.x += PLAYER_RADIUS
            plusXPos.z += PLAYER_HEIGHT / 2 + 0.000001  # Because float is not accurate enough
            pointsPlusX = getRectanglePointsAroundPointVec3(plusXPos, PLAYER_RADIUS, PLAYER_HEIGHT / 2, Axis.x)
            for point in pointsPlusX:
                if getCollision(game.environment, point):
                    # print("COLLISION back")
                    game.player.position.x = math.floor(point.x) - PLAYER_RADIUS
                    game.player.velocity.x = 0
                    break

        ## resolve Y velocity ##
        game.player.position.y += game.player.velocity.y * delta

        if game.player.velocity.y < 0:
            minusYPos = game.player.position.copy()
            minusYPos.y -= PLAYER_RADIUS
            minusYPos.z += PLAYER_HEIGHT / 2 + 0.000001  # Because float is not accurate enough
            pointsMinusY = getRectanglePointsAroundPointVec3(minusYPos, PLAYER_RADIUS, PLAYER_HEIGHT / 2, Axis.y)
            for point in pointsMinusY:
                if getCollision(game.environment, point):
                    # print("COLLISION right")
                    game.player.position.y = math.floor(point.y) + 1 + PLAYER_RADIUS
                    game.player.velocity.y = 0
                    break

        elif game.player.velocity.y > 0:
            plusYPos = game.player.position.copy()
            plusYPos.y += PLAYER_RADIUS
            plusYPos.z += PLAYER_HEIGHT / 2 + 0.000001  # Because float is not accurate enough
            pointsPlusY = getRectanglePointsAroundPointVec3(plusYPos, PLAYER_RADIUS, PLAYER_HEIGHT / 2, Axis.y)
            for point in pointsPlusY:
                if getCollision(game.environment, point):
                    # print("COLLISION left")
                    game.player.position.y = math.floor(point.y) - PLAYER_RADIUS
                    game.player.velocity.y = 0
                    break

    @staticmethod
    def _slowDownXYVelocity(game: Game, delta: float):
        # Air slipperiness
        game.player.velocity.x *= nthRoot(0.91, 1 / delta)  # 0.91 -> https://www.mcpk.wiki/wiki/Slipperiness
        game.player.velocity.y *= nthRoot(0.91, 1 / delta)  # 0.91 -> https://www.mcpk.wiki/wiki/Slipperiness

        if playerIsStanding(game.player.position, game.environment):
            # Block slipperiness
            game.player.velocity.x *= nthRoot(0.6, 1 / delta)  # 0.6 -> https://www.mcpk.wiki/wiki/Slipperiness
            game.player.velocity.y *= nthRoot(0.6, 1 / delta)  # 0.6 -> https://www.mcpk.wiki/wiki/Slipperiness
