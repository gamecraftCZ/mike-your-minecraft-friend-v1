import math
from typing import List

import gym
import numpy as np
from gym import spaces

from gym_treechop.game.constants import WORLD_SHAPE, Blocks, BlockHardness, HARDNESS_MULTIPLIER
from gym_treechop.game.game import Game
from gym_treechop.game.physiscs import Physics
from gym_treechop.game.renderer import Renderer


class REWARDS:
    # Sweeties
    WOOD_CHOPPED = 1000
    LOOKING_AT_WOOD = 50  # 10x time punishment ???
    WOOD_CHOPPING_PER_TICK = 5  # Up to reward x*10 = 20 ???
    BEING_CLOSE_TO_TREE = 8  # 0.05  # For closer each block closer to the center ???

    # Punishments
    TICK_PASSED = -0.004
    WRONG_BLOCK_DESTROYED = -100
    DIED = -100  # -10_000


DELTA = 0.1


# MAX_GAME_LENGTH_STEPS = 100  # 10s
# MAX_GAME_LENGTH_STEPS = 50  # 5s


# MAX_GAME_LENGTH_STEPS = 300  # 30s
# MAX_GAME_LENGTH_STEPS = 500  # 50s

class TreeChopEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, maxGameLengthSteps: int = 50):
        self.setup = {"max_game_length_steps": maxGameLengthSteps}
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
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
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
        world_size = 0  # WORLD_SHAPE.x * WORLD_SHAPE.y * WORLD_SHAPE.z * len(BLOCK_TYPES)  # one-hot encoded
        player_observations = 3 + 3 + 2  # position, velocity, rotation
        looking_at_block_one_hot = 4  # air, ground, wood, leaf
        kicking_blocks = 1  # true/false
        chopping = 1  # 0 - fully chopped
        observations_count = world_size + player_observations + looking_at_block_one_hot + kicking_blocks + chopping
        self.observation_space = spaces.Box(low=-1, high=1, shape=(observations_count,), dtype=np.float32)

        self.game = Game()
        self.state = self._getDefaultState()
        self.renderer = None

    def _getDefaultState(self):
        return {"center": self.game.getPlayerDistanceToCenter(),
                "look": False,
                "chopping_reward": 0,
                "steps_passed": 0, }

    def _isDone(self) -> bool:
        return self.game.isGameOver() \
               or self.game.getWoodLeft() == 0 \
               or self.state["steps_passed"] >= self.setup["max_game_length_steps"]

    def step(self, action: List[float]):
        # print("ACTION: ", action)
        wood_left = self.game.getWoodLeft()
        # obs = self._getObservation()
        done = self._isDone()

        reward = 0

        if not done:
            # Move
            if action[1] > 0.5:  # Jump
                self.game.jump()

            # if action[2] < -0.5:  # Backward
            #     self.game.backward()
            if action[2] > -0.5:  # .5:  # Forward
                self.game.forward()

            # if action[3] < -0.5:  # Left
            #     self.game.left()
            # if action[3] < -0.5:  # Right
            #     self.game.right()

            # Look - continuous
            # upDown = self.game.player.rotation.y + (action[4] / 8)  # -1 to 1 -> 0-1PI
            # leftRight = self.game.player.rotation.x + (action[5] / 8)  # -1 to 1 -> 0-2PI
            #
            # self.game.player.rotation.y = limit(upDown, 0, math.pi)
            # self.game.player.rotation.x = limit(leftRight, 0, 2 * math.pi)

            # Look - exact
            upDown = (action[4] + 1) / 2 * math.pi  # 0-2 -> 0-1PI
            leftRight = (action[5] + 1) * math.pi  # 0-2 -> 0-2PI

            self.game.player.rotation.y = upDown
            self.game.player.rotation.x = leftRight

            # Physics
            for i in range(int(1 / DELTA)):  # 0.1*10 = 1tick
                Physics.step(self.game, DELTA)

            # Attack blocks, REWARD +- wood chopped, wrong block destroyed
            if action[0]:
                block = self.game.attackBlock(DELTA)
                if block:
                    if block == Blocks.WOOD:
                        reward += REWARDS.WOOD_CHOPPED
                        self.state["chopping_reward"] = 0
                        wood_left = self.game.getWoodLeft()
                        # wood_left = 0 # TODO remove this line, makes the AI to choop only 1 wood
                        print("Chopped FULL WOOD Block")
                    elif block == Blocks.LEAF:
                        pass
                    else:
                        reward += REWARDS.WRONG_BLOCK_DESTROYED
            else:
                self.game.stopBlockAttack()

            # REWARD - game over
            if self.game.isGameOver():
                reward += REWARDS.DIED

            # REWARD +- looking at wood, chopping wood
            block, blockPosition = self.game.getBlockInFrontOfPlayer()
            # if block:
            #     print("Looking at: ", Blocks.toName(block))

            if block == Blocks.WOOD:
                if not self.state["look"]:
                    reward += REWARDS.LOOKING_AT_WOOD
                    print("WOOD look!")
                self.state["look"] = True

                if self.game.attackTicksRemaining > 0:
                    rewardToGet = max(0, (40 - self.game.attackTicksRemaining)) * REWARDS.WOOD_CHOPPING_PER_TICK
                    reward += rewardToGet
                    self.state["chopping_reward"] += rewardToGet
                    print("Hit Wood")

            else:
                if self.state["look"]:
                    reward -= REWARDS.LOOKING_AT_WOOD
                    self.state["look"] = False
                    print("No Wood look!")

            if self.game.attackTicksRemaining <= 0:
                # stopped chopping
                if self.state["chopping_reward"]:
                    print("UnHit Wood")
                reward -= self.state["chopping_reward"]
                self.state["chopping_reward"] = 0

            # REWARD + moving to the center
            newDistanceToCenter = self.game.getPlayerDistanceToCenter()
            distanceChange = self.state["center"] - newDistanceToCenter
            reward += (distanceChange * REWARDS.BEING_CLOSE_TO_TREE)

            self.state["center"] = newDistanceToCenter

            # REWARD - time passes
            reward += REWARDS.TICK_PASSED

            # New observation
            # wood_left = self.game.getWoodLeft()
            done = self._isDone()

        info = {"wood_left": wood_left}

        # print("Obs: ", obs.tolist())
        # if done:
        #     print("DONE")

        self.state["steps_passed"] += 1

        obs = self._getObservation()
        return obs, reward, done, info

    def reset(self):
        del self.game
        self.game = Game()  # Create new game
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

        # Player position (-0.5 - 0.5), rotation, velocity
        obs = np.append(obs,
                        self.game.player.position.toNumpy() / max(WORLD_SHAPE.x, WORLD_SHAPE.y, WORLD_SHAPE.z) - [.5,
                                                                                                                  .5,
                                                                                                                  .5])
        obs = np.append(obs, (self.game.player.rotation.x - math.pi) / math.pi)  # 0 - 2PI -> -1PI - 1PI
        obs = np.append(obs, (self.game.player.rotation.y - (math.pi / 2)) / (math.pi / 2))  # 0 - 1PI -> -0.5PI - 0.5PI
        obs = np.append(obs, self.game.player.velocity.toNumpy() / 3.92)  # Terminal velocity = 3.92

        # Looking at block  [air, ground, wood, leaf]
        lookingBlock, blockPos = self.game.getBlockInFrontOfPlayer()
        obs = np.append(obs, 1 if lookingBlock == Blocks.AIR else 0)
        obs = np.append(obs, 1 if lookingBlock == Blocks.GROUND else 0)
        obs = np.append(obs, 1 if lookingBlock == Blocks.WOOD else 0)
        obs = np.append(obs, 1 if lookingBlock == Blocks.LEAF else 0)

        # Is kicking to some block [air, ground, wood, leaf]
        kickingBlock, blockPos = self.game.getBlockInFrontOfPlayer(0, 0.5)
        obs = np.append(obs, 1 if kickingBlock else 0)

        # Chopping block progress
        obs = np.append(obs, 1 - (self.game.attackTicksRemaining
                                  / (BlockHardness[lookingBlock]
                                     * HARDNESS_MULTIPLIER)) if self.game.attackTicksRemaining else -1)

        # Clip everything in range -1 to 1
        return obs.clip(-1, 1)
