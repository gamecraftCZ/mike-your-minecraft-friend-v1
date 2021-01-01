from gym_treechop.TreeChopEnv import TreeChopEnv


def main():
    env = TreeChopEnv(maxGameLengthSteps=600)

    for i in range(600):
        action = env.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        input(f"obs: {obs}")


if __name__ == '__main__':
    main()
