import math
from time import time, sleep

from gym_treechop.game.game import Game
from gym_treechop.game.physiscs import Physics
from gym_treechop.game.renderer import Renderer
from gym_treechop.game.utils import limit

WAIT_BETWEEN_FRAMES_TICKS = 1  # should be 1 tick
WAIT_BETWEEN_FRAMES_SECONDS = WAIT_BETWEEN_FRAMES_TICKS / 20  # should be 0.1 s for 2 ticks

ADDITIONAL_WAIT_SECONDS = 0.05  # To reduce lag in preview


def processKeyboardInput(game: Game, key: str):
    if "down" in key:
        game.lookUpDown(max(0, (game.player.rotation.y - 0.1)))
    elif "up" in key:
        game.lookUpDown(min(math.pi, (game.player.rotation.y + 0.1)))

    elif "left" in key:
        game.lookLeftRight(game.player.rotation.x - 0.1)
    elif "right" in key:
        game.lookLeftRight(game.player.rotation.x + 0.1)

    elif "a" in key:
        game.left()
    elif "d" in key:
        game.right()
    elif "w" in key:
        game.forward()
    elif "s" in key:
        game.backward()
    elif " " in key:
        game.jump()

    else:
        print(f"Unknown key input: '{key}'")


mouseDown = False


def main():
    global mouseDown
    print("Hi")

    renderer = Renderer()
    game = Game(renderer, tree_blocks_to_generate=6)

    # input("Are you ready!")

    def onKeyDown(evt):
        k = evt.key
        processKeyboardInput(game, k)

    renderer.canvas.bind("keydown", onKeyDown)

    def onMouseDown():
        global mouseDown
        mouseDown = True
        print("MouseDown")

    def onMouseUp():
        global mouseDown
        mouseDown = False
        print("MouseUp")

    renderer.canvas.bind("mousedown", onMouseDown)
    renderer.canvas.bind("mouseup", onMouseUp)

    try:
        renderer.render(game)

        while True:
            startTime = time()

            # print(f"Running next frame step with delta {WAIT_BETWEEN_FRAMES_TICKS / 10} ticks")
            for i in range(WAIT_BETWEEN_FRAMES_TICKS * 10):
                Physics.step(game, 0.1)
                if mouseDown:
                    block = game.attackBlock(0.1)
                    if block:
                        print("Destroyed: ", block)
                else:
                    game.stopBlockAttack()

            renderer.render(game)

            #####################################################################
            blockToDestroy = game.getNextWoodBlock()
            distanceToBlock = game.player.getHeadPosition().getLengthTo(blockToDestroy)

            # rotation_to_block_to_destroy - leftRight -> 0PI - 2PI -> -1PI - 1PI -> -1 - 1
            b = blockToDestroy.x + 0.5 - game.player.getHeadPosition().x
            a = blockToDestroy.y + 0.5 - game.player.getHeadPosition().y
            leftRight = math.atan(b / a) - math.pi / 2  # angle beta is by the block (viz. looking_angle.png)
            if a < 0:
                leftRight += math.pi
            leftRight = -leftRight
            # print(f"posit: {game.player.position}")
            print()
            print(f"  rotX: {game.player.rotation.x / math.pi * 180 - 180 :.2f}°")
            print(f"  angleLR: {leftRight / math.pi * 180 :.2f}°")

            leftRightToBeDeleted = (game.player.rotation.x - leftRight) % math.tau
            if leftRightToBeDeleted > math.pi:
                leftRightToBeDeleted = 2 * math.pi - leftRightToBeDeleted
            print(f"    leftRight: {leftRightToBeDeleted / math.pi * 180 :.2f}°")

            # print(f"posit: {game.player.position}")
            # print(f"block: {blockToDestroy}")

            # rotation_to_block_to_destroy - upDown -> 0PI - 1PI -> -0.5PI - 0.5PI -> -1 - 1
            c = distanceToBlock
            b = limit(blockToDestroy.z + 0.5 - game.player.getHeadPosition().z, -c, c)
            upDown = math.asin(b / c)
            print()
            print(f"  rotY: {game.player.rotation.y / math.pi * 180 - 90 :.2f}°")
            print(f"  angleUD: {upDown / math.pi * 180 :.2f}°")

            upDownToBeDeleted = (game.player.rotation.y - upDown) % math.pi - math.pi / 2
            print(f"    upDown: {upDownToBeDeleted / math.pi * 180 :.2f}°")
            #####################################################################

            # print(f"Wood left: {game.getWoodLeft()}")

            delta = time() - startTime
            sleep_time = WAIT_BETWEEN_FRAMES_SECONDS - delta + ADDITIONAL_WAIT_SECONDS
            if sleep_time > 0:
                sleep(sleep_time)

    except KeyboardInterrupt:
        print("Exiting")

    print("Hasta La Vista, baby!")


if __name__ == '__main__':
    main()
