import math

import gym
import numpy as np
from gym import spaces

from gym_treechop.game.constants import WORLD_SHAPE, BLOCK_TYPES
from gym_treechop.game.game import Game
from gym_treechop.game.physiscs import Physics
from gym_treechop.game.renderer import Renderer


class REWARDS:
    TICK_PASSED = -0.1
    WRONG_BLOCK_DESTROYED = -100
    WOOD_CHOPPED = 1_000
    DIED = -1_000_000


DELTA = 0.1
class TreeChopEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
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
        world_size = WORLD_SHAPE.x * WORLD_SHAPE.y * WORLD_SHAPE.z * len(BLOCK_TYPES)  # one-hot encoded
        player_observations = 3 + 3 + 2  # position, velocity, rotation
        observations_count = world_size + player_observations
        self.observation_space = spaces.Box(low=-1, high=1, shape=(observations_count,), dtype=np.float32)

        self.game = Game()
        self.renderer = None

    def step(self, action):
        print("ACTION: ", action)
        wood_left = self.game.getWoodLeft()

        obs = self._getObservation()
        done = self.game.isGameOver() or self.game.getWoodLeft() == 0
        reward = 0

        if not done:
            # Move
            if action[1] > 0.5:  # Jump
                self.game.jump()

            if action[2] < -0.5:  # Backward
                self.game.backward()
            if action[2] > 0.5:  # Forward
                self.game.forward()

            if action[3] < -0.5:  # Left
                self.game.left()
            if action[3] < -0.5:  # Right
                self.game.right()

            # Look
            upDown = (self.game.player.rotation.y + 1) / 2 * math.pi  # 0-2 -> 0-PI
            leftRight = (self.game.player.rotation.x + 1) * math.pi  # 0-2 -> 0-2PI

            self.game.player.rotation.y = upDown
            self.game.player.rotation.x = leftRight

            # Attack blocks
            if action[0]:
                self.game.attackBlock(DELTA)
            else:
                self.game.stopBlockAttack()

            for i in range(int(1 / DELTA)):  # 0.1*10 = 1tick
                Physics.step(self.game, DELTA)

            if self.game.isGameOver():
                reward += REWARDS.DIED

            reward += REWARDS.TICK_PASSED

            done = self.game.isGameOver() or self.game.getWoodLeft() == 0

        info = {"wood_left": wood_left}
        return obs, reward, done, info

    def reset(self):
        del self.game
        self.game = Game()  # Create new game

    def render(self, mode='human', close=False):
        if not self.renderer:
            self.renderer = Renderer()

        self.renderer.render(self.game)

    def close(self):
        self.renderer.canvas.delete()
        self.renderer = None

    def _getObservation(self):
        obs = self.game.getEnvironmentOneHotEncoded().flatten()
        obs = np.append(obs, self.game.player.position.toNumpy() / max(WORLD_SHAPE.x, WORLD_SHAPE.y, WORLD_SHAPE.z))
        obs = np.append(obs, (self.game.player.rotation.x - math.pi) / math.pi)  # 0 - 2PI -> -1PI - 1PI
        obs = np.append(obs, (self.game.player.rotation.y - (math.pi / 2)) / (math.pi / 2))  # 0 - 1PI -> -0.5PI - 0.5PI
        obs = np.append(obs, self.game.player.velocity.toNumpy() / 3.92)  # Terminal velocity = 3.92
        return obs
