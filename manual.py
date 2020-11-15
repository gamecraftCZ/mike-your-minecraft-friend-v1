import math
from time import time, sleep

from game import Game
from physiscs import Physics
from renderer import Renderer

WAIT_BETWEEN_FRAMES_TICKS = 2  # should be 2 ticks
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


def main():
    print("Hi")
    game = Game()

    renderer = Renderer()

    # input("Are you ready!")

    def onKeyDown(evt):
        k = evt.key
        processKeyboardInput(game, k)

    renderer.canvas.bind("keydown", onKeyDown)

    try:
        renderer.render(game)

        while True:
            startTime = time()

            # print(f"Running next frame step with delta {WAIT_BETWEEN_FRAMES_TICKS / 10} ticks")
            for i in range(WAIT_BETWEEN_FRAMES_TICKS * 10):
                Physics.step(game, 0.1)

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
