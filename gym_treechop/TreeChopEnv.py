import math
from random import randint
from typing import List

import gym
import numpy as np
from gym import spaces
from numba import jit

from gym_treechop.game.constants import Blocks
from gym_treechop.game.game import Game, numba_getBlockDistance
from gym_treechop.game.physiscs import Physics
from gym_treechop.game.renderer import Renderer
from gym_treechop.game.structures import Vec3, numba_Vec3Rotate
from gym_treechop.game.utils import limit, playerIsStanding


class REWARDS:
    # Sweeties
    LONG_LOOK_AT_TARGET = 100
    LOOKING_AT_TARGET = 1  # For each tick
    BEING_CLOSE_TO_TREE = 1  # 0.05  # For closer each block closer to the center ???
    MOVE_REWARD = 0.01  # This will encourage Mike to move

    # Deprecated sweeties
    WOOD_CHOPPED = 0  # 1_000
    WOOD_CHOPPING_PER_TICK = 0  # 2  # Up to reward x*10 = 20 ???

    CHOPPING_REWARD = 0  # 0.02  # This will encourage Mike to chop

    # Punishments
    TICK_PASSED = -0.04
    WRONG_BLOCK_DESTROYED = -5
    DIED = 0  # -1_000  # -10_000


DELTA = 0.1


class TreeChopEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, maxGameLengthSteps: int = 50, endAfterOneBlock: bool = True, fixedTreeHeight: int = None):
        self.game = Game()
        self.renderer = None

        self.setup = {
            "max_game_length_steps": maxGameLengthSteps,
            "end_after_one_block": endAfterOneBlock,
            "fixed_tree_height": fixedTreeHeight
        }
        self.state = self._getDefaultState()  # Must be after self.game initialization!

        # Action Space
        action_attack = 1  # Attack (-1; 1) - Attacks if 0.5+
        action_forward = 1  # Forward (-1; 1) - Forward if 0.5+
        action_jump = 1  # Jump (-1; 1) - Jumps if 0.5+
        action_move_left_right = 2  # Left/Right (-1; 1) - Left/Right if 0.5+
        action_rotation = 4  # Rotation X(left/right) ,Y( (-1; 1) - upDown 0-1PI, leftRight 0-2PI
        actions_count = action_attack + action_forward + action_jump + action_move_left_right + action_rotation
        self.action_space = spaces.Box(low=-1, high=1, shape=(actions_count,), dtype=np.float32)

        # Observation Space
        player_velocity_upDown = 1
        distance_to_block_to_destroy = 1
        rotation_to_block_to_destroy = 2
        looking_at = 2  # block with penalty for destroy / no penalty for destroy
        viewport = 64 * 64  # Distance to blocks in front of Mike 64x64 - 128° field of view -> 1 point / 2°

        observations_count = player_velocity_upDown \
                             + distance_to_block_to_destroy + rotation_to_block_to_destroy \
                             + looking_at + viewport
        self.observation_space = spaces.Box(low=-1, high=1, shape=(observations_count,), dtype=np.float32)


    def step(self, action: List[float]):
        actions = {
            "attack": action[0] > 0.5,
            "forward": action[1] > 0.5,
            "jump": action[2] > 0.5,
            "left": action[3] > 0.5,
            "right": action[4] > 0.5,
            "rotation-up": action[5],
            "rotation-down": action[6],
            "rotation-left": action[7],
            "rotation-right": action[8],
        }
        targetBlockPosition = self.game.getNextWoodBlock()

        reward = 0
        if not self._isDone():
            # 1.1. Move
            if actions["jump"]:
                self.game.jump()

            if actions["forward"]:
                self.game.forward()
                reward += REWARDS.MOVE_REWARD

            # # 1.2. Look - discrete actions
            if actions["rotation-up"] > 0.5: self.game.lookUpDown(self.game.player.rotation.y + 0.1)
            if actions["rotation-down"] > 0.5: self.game.lookUpDown(self.game.player.rotation.y - 0.1)
            if actions["rotation-right"] > 0.5: self.game.lookLeftRight(self.game.player.rotation.x + 0.1)
            if actions["rotation-left"] > 0.5: self.game.lookLeftRight(self.game.player.rotation.x - 0.1)

            # # 1.2. Look - exact
            # self.game.player.rotation.y = (actions["rotation-up-down"] + 1) / 2 * math.pi  # -1 - 1 -> 0-2 -> 0-1PI
            # self.game.player.rotation.x = (actions["rotation-left-right"] + 1) * math.pi  # -1 - 1 -> 0-2 -> 0-2PI

            # 2. Physics
            for i in range(int(1 / DELTA)):  # 0.1*10 = 1tick
                Physics.step(self.game, DELTA)

            # 3. Attack blocks | REWARD +- wood chopped, wrong block destroyed
            if actions["attack"]:
                block = self.game.attackBlock(DELTA)
                if block:
                    print(f"Chopped: {Blocks.toName(block)}")
                    if block == Blocks.WOOD:
                        pass
                    elif block == Blocks.LEAF:
                        pass
                    else:
                        reward += REWARDS.WRONG_BLOCK_DESTROYED
            else:
                self.game.stopBlockAttack()

            # 4.1. REWARD - game over
            if self.game.isGameOver():
                reward += REWARDS.DIED

            # 4.2. REWARD +- looking at wood, chopping wood
            block, blockPosition = self.game.getBlockInFrontOfPlayer()
            # if block:
            #     print("Looking at: ", Blocks.toName(block))

            if blockPosition == targetBlockPosition:
                if playerIsStanding(self.game.player.position, self.game.environment):
                    self.state["done"] = True
                    reward += REWARDS.LONG_LOOK_AT_TARGET
                    # DO NOT encourage him to jump just to get more sweeties.
                    reward -= self.state["looking_reward"]
                    self.state["looking_reward"] = 0
                    print("SUCCESS Block Look standing!")
                else:
                    print("Block Look!")
                    reward += REWARDS.LOOKING_AT_TARGET
                    self.state["looking_reward"] += REWARDS.LOOKING_AT_TARGET
            else:
                # DO NOT encourage him to look away and then back again to get more sweeties.
                reward -= self.state["looking_reward"]
                self.state["looking_reward"] = 0

            if block:
                # print("Look: ", Blocks.toName(block))
                self.state["latest_look_block_pos"] = blockPosition

            # 4.3. REWARD + moving to the center
            newDistanceToCenter = self.game.getPlayerDistanceToCenter()
            distanceChange = self.state["center"] - newDistanceToCenter
            reward += (distanceChange * REWARDS.BEING_CLOSE_TO_TREE)

            self.state["center"] = newDistanceToCenter

            # 4.4. REWARD - time passes
            reward += REWARDS.TICK_PASSED

        # print("Obs: ", obs.tolist())
        # if done:
        #     print("DONE")

        self.state["steps_passed"] += 1

        info = {"wood_left": self.game.getWoodLeft(), "self": self}
        obs = self._getObservation()
        done = self._isDone()
        return obs, reward, done, info

    def sample(self):
        return self.action_space.sample()

    def reset(self):
        del self.game
        # Create new game, tree has 1-6 lock remaining
        self.game = Game(tree_blocks_to_generate=self.setup["fixed_tree_height"] or randint(1, 6))

        self.state = self._getDefaultState()
        return self._getObservation()

    def render(self, mode='human', close=False):
        if not self.renderer:
            self.renderer = Renderer()

        self.renderer.render(self.game)

    def close(self):
        self.renderer.canvas.delete()
        self.renderer = None

    def _getObservation(self):
        obs = np.array([])  # self.game.getEnvironmentOneHotEncoded().flatten()

        # player_velocity_upDown - only up/down
        obs = np.append(obs, self.game.player.velocity.z / 3.92)  # Terminal velocity = 3.92 -> -1 - 1

        # distance_to_block_to_destroy
        blockToDestroy = self.game.getNextWoodBlock()
        distanceToBlock = self.game.player.getHeadPosition().getLengthTo(blockToDestroy)
        obs = np.append(obs, limit(distanceToBlock / 5, 0, 1))

        # rotation_to_block_to_destroy - leftRight -> -1PI - 1PI -> -1 - 1
        b = blockToDestroy.x + 0.5 - self.game.player.getHeadPosition().x
        a = blockToDestroy.y + 0.5 - self.game.player.getHeadPosition().y
        leftRight = math.atan(b / a) - math.pi / 2  # angle beta is by the block (viz. looking_angle.png)
        if a < 0:
            leftRight += math.pi
        leftRight = -leftRight
        # print(f"leftRight: {leftRight/math.pi*180}")
        obs = np.append(obs, leftRight / math.pi)

        # rotation_to_block_to_destroy - upDown -> -0.5PI - 0.5PI -> -1 - 1
        c = distanceToBlock
        b = limit(blockToDestroy.z + 0.5 - self.game.player.getHeadPosition().z, -c, c)
        upDown = math.asin(b / c)
        # print(f"upDown: {upDown/math.pi*180}")
        obs = np.append(obs, upDown / (math.pi / 2))

        # looking_at [ground, wood, leaf]
        lookingBlock, blockPos = self.game.getBlockInFrontOfPlayer()
        obs = np.append(obs, 1 if lookingBlock == Blocks.GROUND else 0)  # Penalty for destroy
        obs = np.append(obs, 1 if lookingBlock == Blocks.LEAF or lookingBlock == Blocks.WOOD else 0)  # No penalty

        # viewport - Distance to blocks in front of Mike 64x64 - 128° field of view -> 1point/2°x2°, max block dis.=8
        lookingVector = self.game.player.getLookingDirectionVector()
        viewport = numba_getViewport(lookingVector.asTuple(), self.game.environment,
                                     self.game.player.position.asTuple())

        viewport = np.array(viewport).flatten()
        obs = np.append(obs, viewport)

        # Clip everything in range -1 to 1
        return obs.clip(-1, 1)

    def _getDefaultState(self):
        return {
            "center": self.game.getPlayerDistanceToCenter(),
            "look": False,
            "chopping_reward": 0,
            "looking_reward": 0,
            "steps_passed": 0,
            "latest_look_block_pos": None,
            "to_destroy": Vec3(4, 4, 1),
            "done": False
        }

    def _isDone(self) -> bool:
        return self.state["done"] \
               or self.game.isGameOver() \
               or self.game.getWoodLeft() == 0 \
               or self.state["steps_passed"] >= self.setup["max_game_length_steps"]


VIEWPORT_RES_X = 64
VIEWPORT_RES_Y = 64
VIEWPORT_FOV = 128 / 180 * math.pi  # 128° in radians
MAX_BLOCK_DISTANCE = 8


@jit(nopython=True)
def numba_getViewport(lookingVector: (float, float, float),
                      environment: np.ndarray,
                      playerPos: (float, float, float)) -> np.ndarray:
    viewport = []
    for y in range(VIEWPORT_RES_Y):
        y -= VIEWPORT_RES_Y / 2  # Convert to be from -ymax/2 to +ymax/2
        y = y / (VIEWPORT_RES_Y / 2) * VIEWPORT_FOV  # Convert to degrees offset from center
        viewY = []
        for x in range(VIEWPORT_RES_X):
            x -= VIEWPORT_RES_X / 2  # Convert to be from -xmax/2 to +xmax/2
            x = x / (VIEWPORT_RES_X / 2) * VIEWPORT_FOV  # Convert to degrees offset from center

            pointingVector = numba_Vec3Rotate(lookingVector, y, x)
            blockDistance = numba_getBlockDistance(playerPos, pointingVector, MAX_BLOCK_DISTANCE, environment)
            blockClose = MAX_BLOCK_DISTANCE - blockDistance  # 8 = Right in front of player, 0 = Far away
            viewY.append(((blockClose / 2) ** 2))  # Block distance close=4^2=16 ->0,.25,1,2.25,4,6.25,9,12.5,16 ->/16

        viewport.append(viewY)

    return np.array(viewport)
