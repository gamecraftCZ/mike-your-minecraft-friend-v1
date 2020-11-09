from time import time, sleep

from game import Game
from renderer import Renderer


WAIT_BETWEEN_FRAMES = 0.1  # FPS
def main():
    print("Hi")
    game = Game()
    renderer = Renderer()
    input("Are you ready!")

    while True:
        startTime = time()
        game.step(WAIT_BETWEEN_FRAMES)
        renderer.render(game)
        delta = time() - startTime
        sleep(WAIT_BETWEEN_FRAMES - delta)

    print("Hasta La Vista, baby!")


if __name__ == '__main__':
    main()
