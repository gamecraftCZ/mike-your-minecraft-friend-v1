from random import random
from time import time

from gym_treechop.TreeChopEnv import TreeChopEnv
from gym_treechop.game.game import Game
from gym_treechop.game.physiscs import Physics
from gym_treechop.game.structures import Vec3

PHYSICS_RUNS = 100_000
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


BLOCK_ATTACK_RUNS = 100_000
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


TREE_CHOP_ENV_RUNS = 1_000
# 100_000 TREE_CHOP_ENV_RUNS took about 125 seconds on my 7300 HQ laptop before Numba optimizations.
# OLD:                10_000/ 40s -> 250 steps/s
# NEW (1.1.2020): -> 100_000/125s -> 800 steps/s
#
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
    print(f"TreeChop - {TREE_CHOP_ENV_RUNS / elapsed} steps/second")


VEC3_ROTATE_RUNS = 1_000_000


# 1_000_000 runs took about 15s
# PYTHON:       1_000_000/15s -> 66_666  steps/s
# NUMBA:        1_000_000/ 8s -> 125_000 steps/s
def benchmark_Vec3Rotate():
    print(f"Running for {VEC3_ROTATE_RUNS} ticks.")

    startTime = time()
    v = Vec3(1, 1, 1)
    v.rotate(0, 0)  # Make numba to JIT compile the function.
    for tick in range(VEC3_ROTATE_RUNS):
        if not tick % 10_000:
            print(f"Benchmark Vec3 Rotate: {tick}/{VEC3_ROTATE_RUNS} - {(tick / VEC3_ROTATE_RUNS * 100):.2f}%")
        v = v.rotate(random(), random())

    elapsed = time() - startTime
    print(f"Vec3 Rotate - {elapsed} seconds")


GAME_GET_BLOCK_DISTANCE = 1_000_000


# 1_000_000 runs took about 15s
# PYTHON:       100_000/   16s -> 6_250  steps/s
# NUMBA:        1_000_000/ 14s -> 72_000 steps/s
def benchmark_gameGetBlockDistance():
    game = Game()
    print(f"Running for {GAME_GET_BLOCK_DISTANCE} ticks.")

    startTime = time()
    game.getBlockDistance()  # Make numba to JIT compile the function.
    for tick in range(GAME_GET_BLOCK_DISTANCE):
        if not tick % 10_000:
            print(f"Benchmark getBlockDistance: "
                  f"{tick}/{GAME_GET_BLOCK_DISTANCE} - "
                  f"{(tick / GAME_GET_BLOCK_DISTANCE * 100):.2f}%")
        game.getBlockDistance(position=game.player.position, vector=Vec3(random(), random(), random()), maxDistance=8)

    elapsed = time() - startTime
    print(f"getBlockDistance - {elapsed} seconds")


ENV_GET_OBSERVATION = 10_000


# 1_000_000 runs took about 15s
# PYTHON:       100_000/   16s -> 6_250  steps/s
# NUMBA:        1_000_000/ 14s -> 72_000 steps/s
def benchmark_env_getObservation():
    env = TreeChopEnv()
    print(f"Running for {ENV_GET_OBSERVATION} ticks.")

    startTime = time()
    env._getObservation()  # Make numba to JIT compile the function.
    for tick in range(ENV_GET_OBSERVATION):
        if not tick % 1_000:
            print(f"Benchmark env_getObservtion: "
                  f"{tick}/{ENV_GET_OBSERVATION} - "
                  f"{(tick / ENV_GET_OBSERVATION * 100):.2f}%")
        env._getObservation()

    elapsed = time() - startTime
    print(f"env_getObservtion - {elapsed} seconds")


def main():
    # benchmark_physics()
    # benchmark_blockAttack()
    # benchmark_TreeChopEnv()
    # benchmark_Vec3Rotate()
    # benchmark_gameGetBlockDistance()
    benchmark_env_getObservation()


if __name__ == '__main__':
    main()
