import math
from time import time, sleep

from gym_treechop.game.game import Game
from gym_treechop.game.physiscs import Physics
from gym_treechop.game.renderer import Renderer

WAIT_BETWEEN_FRAMES_TICKS = 1  # should be 1 tick
WAIT_BETWEEN_FRAMES_SECONDS = WAIT_BETWEEN_FRAMES_TICKS / 20  # should be 0.1 s for 2 ticks

ADDITIONAL_WAIT_SECONDS = 0.05  # To reduce lag in preview


def processKeyboardInput(game: Game, key: str):
    print(f"Key input: '{key}'")

    if "down" in key:
        game.lookUpDown(max(0, (game.player.rotation.y - 0.1)))
    elif "up" in key:
        game.lookUpDown(min(math.pi, (game.player.rotation.y + 0.1)))

    elif "left" in key:
        game.lookLeftRight((game.player.rotation.x - 0.1) % (2 * math.pi))
    elif "right" in key:
        game.lookLeftRight((game.player.rotation.x + 0.1) % (2 * math.pi))

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


mouseDown = False
def main():
    global mouseDown
    print("Hi")

    renderer = Renderer()
    game = Game(renderer)

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
