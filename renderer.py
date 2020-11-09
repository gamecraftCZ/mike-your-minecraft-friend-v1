from typing import List

import numpy as np
import vpython
from vpython import vector as v

from game import Player, Game


BLOCK_COLORS = {1: v(0.9, 0.7, 0.5), 2: v(0.8, 0.8, 0.4), 3: vpython.color.green}

class Renderer:
    blocks: List[vpython.box] = []

    def render(self, game: Game):
        for object in self.blocks:
            object.delete()

        for z in range(game.environment.shape[0]):
            for y in range(game.environment.shape[0]):
                for x in range(game.environment.shape[0]):
                    block = game.environment[z][y][x]
                    if block:
                        color = BLOCK_COLORS[block]
                        obj = vpython.box(pos=v(x, z, y), length=1, width=1, color=color)
                        self.blocks.append(obj)


        print("Rendered")

    def __init__(self):
        self.canvas = vpython.canvas(title="Be more of who you are!", width=400, height=400)
        print("Initialized Renderer")
