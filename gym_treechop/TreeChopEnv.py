import math
from random import randint
from typing import List

import gym
import numpy as np
from gym import spaces

from gym_treechop.game.constants import Blocks, BlockHardness, HARDNESS_MULTIPLIER
from gym_treechop.game.game import Game
from gym_treechop.game.physiscs import Physics
from gym_treechop.game.renderer import Renderer
from gym_treechop.game.structures import Vec3
from gym_treechop.game.utils import limit


class REWARDS:
    # Sweeties
    LOOK_AT_TARGET = 100
    BEING_CLOSE_TO_TREE = 1  # 0.05  # For closer each block closer to the center ???
    MOVE_REWARD = 0.01  # This will encourage Mike to move

    # Deprecated sweeties
    WOOD_CHOPPED = 0  # 1_000
    LOOKING_AT_WOOD = 0  # 5  # 10x time punishment ???
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
        action_rotation = 2  # Rotation X,Y (-1; 1) - upDown 0-1PI, leftRight 0-2PI
        actions_count = action_attack + action_forward + action_jump + action_move_left_right + action_rotation
        self.action_space = spaces.Box(low=-1, high=1, shape=(actions_count,), dtype=np.float32)

        # Observation Space
        player_velocity_upDown = 1
        distance_to_block_to_destroy = 1
        rotation_to_block_to_destroy = 2
        looking_at = 2  # block with penalty for destroy, no penalty for destroy
        kicking_block = 1

        observations_count = player_velocity_upDown \
                             + distance_to_block_to_destroy + rotation_to_block_to_destroy \
                             + looking_at + kicking_block
        self.observation_space = spaces.Box(low=-1, high=1, shape=(observations_count,), dtype=np.float32)


    def step(self, action: List[float]):
        actions = {
            "attack": action[0] > 0.5,
            "forward": action[1] > 0.5,
            "jump": action[2] > 0.5,
            "left": action[3] > 0.5,
            "right": action[4] > 0.5,
            "rotation-up-down": action[5],
            "rotation-left-right": action[6],
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

            # 1.2. Look - exact
            self.game.player.rotation.y = (actions["rotation-up-down"] + 1) / 2 * math.pi  # -1 - 1 -> 0-2 -> 0-1PI
            self.game.player.rotation.x = (actions["rotation-left-right"] + 1) * math.pi  # -1 - 1 -> 0-2 -> 0-2PI

            # 2. Physics
            for i in range(int(1 / DELTA)):  # 0.1*10 = 1tick
                Physics.step(self.game, DELTA)

            # 3. Attack blocks | REWARD +- wood chopped, wrong block destroyed
            if actions["attack"]:
                reward += REWARDS.CHOPPING_REWARD
                block = self.game.attackBlock(DELTA)
                if block:
                    if block == Blocks.WOOD:
                        reward += REWARDS.WOOD_CHOPPED
                        self.state["chopping_reward"] = 0
                        if self.setup["end_after_one_block"]:
                            self.state["done"] = True
                        print("Chopped FULL Wood Block")
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
                print("Block Found!")
                self.state["done"] = True
                reward += REWARDS.LOOK_AT_TARGET

            if block == Blocks.WOOD:
                if not self.state["look"]:
                    reward += REWARDS.LOOKING_AT_WOOD
                    # print("WOOD look!")
                self.state["look"] = True

                if self.game.attackTicksRemaining > 0:
                    TICKS_TO_DESTROY_WOOD = BlockHardness[Blocks.WOOD] * HARDNESS_MULTIPLIER * 20
                    rewardToGet = max(1, (
                            TICKS_TO_DESTROY_WOOD - self.game.attackTicksRemaining))
                    rewardToGet = math.log(rewardToGet / 2 + 1) * 2 * REWARDS.WOOD_CHOPPING_PER_TICK
                    reward += rewardToGet
                    self.state["chopping_reward"] += rewardToGet
                    print(f"Hit Wood. remaining: {self.game.attackTicksRemaining}, reward: {rewardToGet}")

            else:
                if self.state["look"]:
                    reward -= REWARDS.LOOKING_AT_WOOD
                    self.state["look"] = False
                    # print("No Wood look!")

            if blockPosition != self.state["latest_look_block_pos"] or self.game.attackTicksRemaining <= 0:
                # stopped chopping
                if self.state["chopping_reward"]:
                    print("UnHit Wood")
                    reward -= self.state["chopping_reward"]  # / 2
                    self.state["chopping_reward"] = 0

            if block:
                # print("Attack: ", Blocks.toName(block))
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

        # player_rotation - leftRight
        # obs = np.append(obs, (self.game.player.rotation.x - math.pi) / math.pi)  # 0 - 2PI -> -1PI - 1PI -> -1 - 1

        # player_rotation - upDown
        # obs = np.append(obs, (self.game.player.rotation.y - (math.pi / 2)) / (
        #         math.pi / 2))  # 0 - 1PI -> -0.5PI - 0.5PI -> -1 - 1

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

        # kicking_block - Is kicking to some block, either Ground, Wood or Leaf
        lookingDirection = self.game.player.getLookingDirectionVector2d()
        lookingDirection = Vec3.fromVec2(lookingDirection)
        kickingBlock, blockPos = self.game.getBlockInFrontOfPlayer(0, 0.5, lookingDirection)
        obs = np.append(obs, 1 if kickingBlock else 0)

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
