from time import time, sleep

from vpython import keysdown

from game import Game
from physiscs import Physics
from renderer import Renderer

# TODO rework because step size is 0.2 tick now
WAIT_BETWEEN_FRAMES_TICKS = 1  # should be 2 ticks
WAIT_BETWEEN_FRAMES_SECONDS = WAIT_BETWEEN_FRAMES_TICKS / 20  # should be 0.1 s for 2 ticks

ADDITIONAL_WAIT_SECONDS = 0.2  # To reduce lag in preview


def processKeyboardInput(game: Game, key: str):
    if "a" in key:
        game.left()
    if "d" in key:
        game.right()
    if "w" in key:
        game.forward()
    if "s" in key:
        game.backard()
    if " " in key:
        game.jump()


def main():
    print("Hi")
    game = Game()

    renderer = Renderer()
    physics = Physics()

    # input("Are you ready!")

    def onKeyDown(evt):
        k = evt.key
        processKeyboardInput(game, k)

    renderer.canvas.bind("keydown", onKeyDown)

    try:
        while True:
            startTime = time()

            # print(f"Running next frame step with delta {WAIT_BETWEEN_FRAMES_TICKS / 10} ticks")
            for i in range(WAIT_BETWEEN_FRAMES_TICKS):
                physics.step(game)

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
