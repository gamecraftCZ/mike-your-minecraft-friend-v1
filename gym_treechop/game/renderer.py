from typing import List

import vpython
from vpython import vector as v

from gym_treechop.game.constants import WORLD_SHAPE
from gym_treechop.game.game import Game

BLOCK_COLORS = {1: v(0.9, 0.7, 0.5), 2: v(0.8, 0.8, 0.4), 3: vpython.color.green}


class Renderer:
    mainRenderer: 'Renderer'

    blocks: List[List[List[vpython.box]]] = []
    player: vpython.cylinder
    playerLook: vpython.cylinder

    def _render_blocks(self, game: Game):
        for z in range(game.environment.shape[0]):
            for y in range(game.environment.shape[1]):
                for x in range(game.environment.shape[2]):
                    block = game.environment[z][y][x]
                    old = self.blocks[z][y][x]
                    if block:
                        old.visible = True
                        color = BLOCK_COLORS[block]

                        if old.color != color:
                            old.color = color

                    elif old:
                        old.visible = False
                        # old.delete()
                        # self.blocks[z][y][x] = None

    lastX = 0
    lastY = 0

    def _render_player(self, game: Game):
        position = game.player.position
        self.player.pos = v(position.x, position.z, position.y)
        self.playerLook.pos = v(position.x, position.z + 1, position.y)

        LOOK_LENGTH = 10

        lookingVector = game.player.getLookingDirectionVector()
        self.playerLook.axis = v(lookingVector.x * LOOK_LENGTH, lookingVector.z * LOOK_LENGTH,
                                 lookingVector.y * LOOK_LENGTH)

    def render(self, game: Game):
        self._render_blocks(game)
        self._render_player(game)

    def __init__(self):
        self.mainRenderer = self

        self.canvas = vpython.canvas(title="Be more of who you are!", width=800, height=800)
        self.canvas.center = v(WORLD_SHAPE.y / 2, 0, WORLD_SHAPE.x / 2)  # Camera to rotate around real center

        self.player = vpython.cylinder(axis=v(0, 1.8, 0), up=v(0, 0, 1), radius=0.3, color=vpython.color.red)
        self.playerLook = vpython.cylinder(axis=v(0, 0, 0), up=v(0, 0, 1), radius=0.02, color=vpython.color.orange)

        self.blocks = []
        for z in range(WORLD_SHAPE.z):
            self.blocks.append([])
            for y in range(WORLD_SHAPE.y):
                self.blocks[z].append([])
                for x in range(WORLD_SHAPE.x):
                    block = vpython.box(pos=v(x + 0.5, z + 0.5, y + 0.5), size=v(1, 1, 1))
                    block.visible = False
                    self.blocks[z][y].append(block)

        print("Initialized Renderer")
