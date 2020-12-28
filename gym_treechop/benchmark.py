from time import time

from gym_treechop.TreeChopEnv import TreeChopEnv
from gym_treechop.game.game import Game
from gym_treechop.game.physiscs import Physics

PHYSICS_RUNS = 100_000
BLOCK_ATTACK_RUNS = 100_000
TREE_CHOP_ENV_RUNS = 100_000


# 100_000 PHYSICS_RUNS took 15-25 seconds on my 7300 HQ laptop before Numba optimizations.
def benchmark_physics():
    game = Game()
    print(f"Running for {PHYSICS_RUNS} ticks.")

    startTime = time()
    for tick in range(PHYSICS_RUNS):
        if not tick % 10_000:
            print(f"Benchmark Physics: {tick}/{PHYSICS_RUNS} - {(tick / PHYSICS_RUNS * 100):.2f}%")
        Physics.step(game, 0.1)

    elapsed = time() - startTime
    print(f"Physics - {elapsed} seconds")


# 100_000 BLOCK_ATTACK_RUNS took about 4-30 seconds on my 7300 HQ laptop before Numba optimizations.
def benchmark_blockAttack():
    game = Game()
    print(f"Running for {BLOCK_ATTACK_RUNS} runs.")

    startTime = time()
    for tick in range(BLOCK_ATTACK_RUNS):
        if not tick % 10_000:
            print(f"Benchmark Block attack: {tick}/{BLOCK_ATTACK_RUNS} - {(tick / BLOCK_ATTACK_RUNS * 100):.2f}%")
        game.attackBlock(0.1)

    elapsed = time() - startTime
    print(f"Block attack - {elapsed} seconds")


# 100_000 TREE_CHOP_ENV_RUNS took about 400 seconds on my 7300 HQ laptop before Numba optimizations.
#  -> 10_000/4s -> 2500/s
def benchmark_TreeChopEnv():
    env = TreeChopEnv()

    print(f"Running for {TREE_CHOP_ENV_RUNS} runs.")
    startTime = time()
    for tick in range(TREE_CHOP_ENV_RUNS):
        if not tick % 10_000:
            print(
                f"Benchmark TreeChopEnd: {tick}/{TREE_CHOP_ENV_RUNS} - {(tick / TREE_CHOP_ENV_RUNS * 100):.2f}% - {time() - startTime}s"
            )

        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)

        if done:
            env.reset()

    elapsed = time() - startTime
    print(f"TreeChopEnd - {elapsed} seconds")


def main():
    # benchmark_physics()
    # benchmark_blockAttack()
    benchmark_TreeChopEnv()


if __name__ == '__main__':
    main()
