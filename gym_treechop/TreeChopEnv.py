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
    WOOD_CHOPPED = 1_000
    LOOKING_AT_WOOD = 5  # 10x time punishment ???
    WOOD_CHOPPING_PER_TICK = 2  # Up to reward x*10 = 20 ???
    BEING_CLOSE_TO_TREE = 1  # 0.05  # For closer each block closer to the center ???

    MOVE_REWARD = 0.01  # This will encourage Mike to move

    # Punishments
    TICK_PASSED = -0.04
    WRONG_BLOCK_DESTROYED = -25
    DIED = 0  # -1_000  # -10_000


DELTA = 0.1


class TreeChopEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, maxGameLengthSteps: int = 50, endAfterOneBlock: bool = True):
        self.setup = {"max_game_length_steps": maxGameLengthSteps, "end_after_one_block": endAfterOneBlock}
        # PPO2 algorithm does not support spaces.Tuple!
        # self.action_space = spaces.Tuple((
        #     spaces.Discrete(2),  # Attack 0/1
        #     spaces.Discrete(2),  # Jump 0/1
        #     spaces.Discrete(2),  # Forward 0/1
        #     spaces.Discrete(2),  # Backward 0/1
        #     spaces.Discrete(2),  # Left 0/1
        #     spaces.Discrete(2),  # Right 0/1
        #     spaces.Box(low=-1, high=1, shape=(1,)),  # Up/Down 0 - 1PI
        #     spaces.Box(low=-1, high=1, shape=(1,)),  # Left/Right 0 - 2PI
        # ))
        action_attack = 1
        action_forward = 1
        action_jump = 1
        action_left_right = 2
        action_rotation = 2
        actions_count = action_attack + action_forward + action_jump + action_left_right + action_rotation
        self.action_space = spaces.Box(low=-1, high=1, shape=(actions_count,), dtype=np.float32)  # low=-1, high=1,
        # Attack (-1; 1) - Attacks if 0.5+
        # Jump (-1; 1) - Jumps if 0.5+
        # Backward/Forward (-1; 1) - Backward (-0.5-), Forward (0.5+)
        # Left/Right (-1; 1) - Left (-0.5-), Right (0.5+)
        # Up/Down (-1; 1) - 0 - 1PI
        # Left/Right (-1; 1) - 0 - 2PI

        # PPO2 algorithm does not support spaces.Tuple!
        # self.observation_space = spaces.Tuple((
        #     # Environment - one hot encoded
        #     spaces.MultiBinary((WORLD_SHAPE.x, WORLD_SHAPE.y, WORLD_SHAPE.z, len(BLOCK_TYPES))),
        #     # Player position
        #     spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
        #     # Player velocity
        #     spaces.Box(low=0, high=3.92, shape=(3,), dtype=np.float32),
        #     # Player rotation
        #     spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        # ))
        # world_size = 0  # WORLD_SHAPE.x * WORLD_SHAPE.y * WORLD_SHAPE.z * len(BLOCK_TYPES)  # one-hot encoded
        # player_observations = 3 + 3 + 2  # position, velocity, rotation
        # looking_at_block_one_hot = 4  # air, ground, wood, leaf
        # kicking_blocks = 1  # true/false
        # chopping = 1  # 0 - fully chopped
        # distance_to_tree = 1
        # wood_blocks = WORLD_SHAPE.z
        # time_remaining = 1
        # player_rotation = 2
        player_velocity_upDown = 1
        distance_to_block_to_destroy = 1
        rotation_to_block_to_destroy = 2
        looking_at = 3  # block with penalty for destroy, no penalty for destroy, reward for destroy
        kicking_block = 1
        chopping_progress = 1  # 0 = fully chopped

        observations_count = player_velocity_upDown \
                             + distance_to_block_to_destroy + rotation_to_block_to_destroy \
                             + looking_at + kicking_block + chopping_progress
        # observations_count = world_size + player_observations \
        #                      + looking_at_block_one_hot + kicking_blocks \
        #                      + chopping + distance_to_tree + wood_blocks + time_remaining
        self.observation_space = spaces.Box(low=-1, high=1, shape=(observations_count,), dtype=np.float32)

        self.game = Game()
        self.state = self._getDefaultState()
        self.renderer = None

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
                block = self.game.attackBlock(DELTA)
                if block:
                    if block == Blocks.WOOD:
                        reward += REWARDS.WOOD_CHOPPED
                        self.state["chopping_reward"] = 0
                        if self.setup["end_after_one_block"]:
                            self.state["done"] = True
                        print("Chopped FULL Block")
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

            if block == Blocks.WOOD:
                if not self.state["look"]:
                    reward += REWARDS.LOOKING_AT_WOOD
                    # print("WOOD look!")
                self.state["look"] = True

                if self.game.attackTicksRemaining > 0:
                    TICKS_TO_DESTROY_WOOD = BlockHardness[Blocks.WOOD] * HARDNESS_MULTIPLIER * 20
                    rewardToGet = max(1, (
                            TICKS_TO_DESTROY_WOOD - self.game.attackTicksRemaining))
                    rewardToGet = math.log(rewardToGet + 1) * REWARDS.WOOD_CHOPPING_PER_TICK
                    reward += rewardToGet
                    self.state["chopping_reward"] += rewardToGet
                    # print(f"Hit Wood. remaining: {self.game.attackTicksRemaining}, reward: {rewardToGet}")

            else:
                if self.state["look"]:
                    reward -= REWARDS.LOOKING_AT_WOOD
                    self.state["look"] = False
                    # print("No Wood look!")

            if blockPosition != self.state["latest_look_block_pos"] or self.game.attackTicksRemaining <= 0:
                # stopped chopping
                if self.state["chopping_reward"]:
                    # print("UnHit Wood")
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

        info = {"wood_left": self.game.getWoodLeft()}
        obs = self._getObservation()
        done = self._isDone()
        return obs, reward, done, info

    def sample(self):
        return self.action_space.sample()

    def reset(self):
        del self.game
        self.game = Game(tree_blocks_to_generate=randint(1, 6))  # Create new game, tree has 1-6 lock remaining
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
        obs = np.append(obs, 1 if blockPos == self.game.getNextWoodBlock().z else 0)  # Reward for destroy
        obs = np.append(obs,
                        1 if lookingBlock == Blocks.LEAF or lookingBlock == Blocks.WOOD else 0)  # No penalty, No reward

        # kicking_block - Is kicking to some block, either Ground, Wood or Leaf
        lookingDirection = self.game.player.getLookingDirectionVector()
        lookingDirection.z = 0
        kickingBlock, blockPos = self.game.getBlockInFrontOfPlayer(0, 0.5, lookingDirection)
        obs = np.append(obs, 1 if kickingBlock else 0)

        # chopping_progress
        obs = np.append(obs, 1 - (self.game.attackTicksRemaining
                                  / (BlockHardness[lookingBlock]
                                     * HARDNESS_MULTIPLIER)) if self.game.attackTicksRemaining else -1)

        # Clip everything in range -1 to 1
        return obs.clip(-1, 1)

    def _getDefaultState(self):
        return {"center": self.game.getPlayerDistanceToCenter(),
                "look": False,
                "chopping_reward": 0,
                "steps_passed": 0,
                "latest_look_block_pos": None,
                "to_destroy": Vec3(4, 4, 1),
                "done": False}

    def _isDone(self) -> bool:
        return self.state["done"] \
               or self.game.isGameOver() \
               or self.game.getWoodLeft() == 0 \
               or self.state["steps_passed"] >= self.setup["max_game_length_steps"]
