from time import time, sleep

from game import Game
from physiscs import Physics
from renderer import Renderer

WAIT_BETWEEN_FRAMES_TICKS = 1  # should be 2 ticks
WAIT_BETWEEN_FRAMES_SECONDS = WAIT_BETWEEN_FRAMES_TICKS / 20  # should be 0.1 s for 2 ticks

ADDITIONAL_WAIT_SECONDS = 0.3  # To reduce lag in preview


def main():
    print("Hi")
    game = Game()

    renderer = Renderer()
    physics = Physics()
    # input("Are you ready!")

    try:
        while True:
            startTime = time()

            print(f"Running next frame step with delta {WAIT_BETWEEN_FRAMES_TICKS / 10} ticks")
            for i in range(WAIT_BETWEEN_FRAMES_TICKS):
                physics.step(game)

            renderer.render(game)

            print(f"Wood left: {game.getWoodLeft()}")

            delta = time() - startTime
            sleep_time = WAIT_BETWEEN_FRAMES_SECONDS - delta + ADDITIONAL_WAIT_SECONDS
            if sleep_time > 0:
                sleep(sleep_time)

    except KeyboardInterrupt:
        print("Exiting")

    print("Hasta La Vista, baby!")


if __name__ == '__main__':
    main()
