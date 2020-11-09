from time import time, sleep

from game import Game
from renderer import Renderer


WAIT_BETWEEN_FRAMES = 0.1  # FPS
def main():
    print("Hi")
    renderer = Renderer()
    game = Game(renderer)
    # input("Are you ready!")

    try:
        while True:
            startTime = time()

            game.step(WAIT_BETWEEN_FRAMES)
            renderer.render(game)
            print(f"Wood left: {game.getWoodLeft()}")

            delta = time() - startTime
            sleep_time = WAIT_BETWEEN_FRAMES - delta
            if sleep_time > 0:
                sleep(sleep_time)

    except KeyboardInterrupt:
        print("Exiting")

    print("Hasta La Vista, baby!")


if __name__ == '__main__':
    main()
