import os
import sys

# Has to be here so the os PATH is correct for the imports
sys.path.append(os.getcwd())

from gym_treechop.TreeChopEnv import TreeChopEnv, REWARDS

from time import time

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpLstmPolicy, FeedForwardPolicy, LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from tensorflow.python.client import device_lib


# Custom MLP policy of three layers of size 128 each - NOT USED
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")


class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)


def main():
    print("####################################")
    print("Compute Devices:")
    print(device_lib.list_local_devices())
    print("####################################")

    env = TreeChopEnv()
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    env = DummyVecEnv([lambda: env])
    # env = make_vec_env('CartPole-v1', n_envs=4)

    # TODO Try make_vec_env for faster training
    # n_cpu = 2
    # env = SubprocVecEnv([lambda: env for i in range(n_cpu)])

    model = PPO2(
        policy=MlpLstmPolicy,
        env=env,
        # n_steps=512,
        nminibatches=1,
        learning_rate=lambda f: f * 2.5e-4,  # 2.5e-4,  #
        tensorboard_log="./ppo2_tensorboard/",
        verbose=1
    )

    TIMESTAMPS = 200_000  # _000
    # model.load("trained_PPO31.model")
    # model = PPO2.load("trained_PPO31.model", env)
    model.learn(total_timesteps=TIMESTAMPS)
    model.save(f"trained_{int(time())}_{TIMESTAMPS}.model")

    print("#########################################")
    print("################ TEST: ##################")
    print("#########################################")
    import matplotlib.pyplot as plt

    input("Press any key to start...")
    obs = env.reset()
    cumulativeReward = 0
    plt.axis([0, 500, -20, 150])
    # toPlotY = []
    try:
        for i in range(500):  # 10 = 1 second in game
            action, _states = model.predict(obs)
            # action = env.action_space.sample()

            obs, rewards, done, info = env.step(action)
            cumulativeReward += rewards
            if abs(rewards) > abs(REWARDS.TICK_PASSED):
                print(f"Reward: {rewards}, done: {done}, info: {info}")
            print("Cumulative reward: ", cumulativeReward)

            env.render()

            if done:
                obs = env.reset()
                cumulativeReward = 0

            # Plotting reward each 10 ticks
            # if i % 10 == 0:
            plt.scatter(i, cumulativeReward)

            # toPlotY.append(cumulativeReward)
            # plt.plot(toPlotY, linestyle='solid', color='blue')

            # sleep(0.05)
            plt.pause(0.05)
    except:
        pass

    plt.show()
    env.close()


if __name__ == '__main__':
    main()
