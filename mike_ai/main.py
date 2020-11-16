from time import sleep

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from gym_treechop.TreeChopEnv import TreeChopEnv


def main():
    env = TreeChopEnv()
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10)

    print("#########################################")
    print("################ TEST: ##################")
    print("#########################################")
    obs = env.reset()
    for i in range(2000):  # 2000ticks = 100seconds game
        # action, _states = model.predict(obs)
        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        if round(rewards):
            print(f"Reward: {rewards}, done: {done}, info: {info}")
        env.render()
        sleep(0.1)

    env.close()


if __name__ == '__main__':
    main()
