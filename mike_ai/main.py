import os
import sys
from time import time

from stable_baselines.common.policies import MlpLstmPolicy

# Has to be here so the os PATH is correct for the imports
from gym_treechop.game.physiscs import Physics

sys.path.append(os.getcwd())

from gym_treechop.TreeChopEnv import TreeChopEnv

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import CheckpointCallback

from tensorflow.python.client import device_lib


def main():
    print("####################################")
    print("Compute Devices:")
    print(device_lib.list_local_devices())
    print("####################################")

    env = TreeChopEnv(maxGameLengthSteps=60)
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
        learning_rate=lambda f: (f + 0.3) ** 3 * 2.5e-3,  # 2.5e-4,  #
        tensorboard_log="./ppo2_tensorboard/",
        verbose=1,
    )

    # model = SAC(
    #     policy=LnMlpPolicy,
    #     env=env,
    #     # nminibatches=1,
    #     # learning_rate=7.5e-3,
    #     tensorboard_log="./look_tensorboard/",
    #     verbose=1,
    # )

    # model = A2C(
    #     policy=MlpLstmPolicy,
    #     env=env,
    #     # nminibatches=1,
    #     # learning_rate=7.5e-3,
    #     tensorboard_log="./a2c_tensorboard/",
    #     verbose=1,
    # )

    checkpoint_callback = CheckpointCallback(save_freq=1_000, save_path='./model_checkpoints/')

    TIMESTAMPS = 200_000_000  # _000
    # model = SAC.load("rl_model_635000_steps.zip", env)
    # model = SAC.load("model_checkpoints/rl_model_205000_steps.zip", env)
    model.tensorboard_log = "./hhhhhhh_tensorboard/"
    model.learn(total_timesteps=TIMESTAMPS, callback=[checkpoint_callback, ])
    model.save(f"trained_{int(time())}_{TIMESTAMPS}.zip")

    print("#########################################")
    print("################ TEST: ##################")
    print("#########################################")
    import matplotlib.pyplot as plt

    input("Press any key to start...")
    env = TreeChopEnv(maxGameLengthSteps=160, endAfterOneBlock=False, fixedTreeHeight=6)

    cumulativeReward = 0
    plt.axis([0, 5000, -20, 650])
    print("Started")
    # toPlotY = []
    try:
        obs = env.reset()
        for i in range(5000):  # 10 = 1 second in game
            action, _states = model.predict(obs)
            # print("ACTION: ", action)

            # action = env.action_space.sample()

            obs, rewards, done, info = env.step(action)
            cumulativeReward += rewards

            env.render()

            if done:
                # Chop the block looking at, until chopped or look changed to another block.
                # Physics
                for ii in range(int(1 / 0.1)):  # 0.1*10 = 1tick
                    Physics.step(env.game, 0.1)

                a = env.game.getBlockInFrontOfPlayer()
                b = env.game.getNextWoodBlock()
                c = env.game.attackBlock(0.1)
                # while rawEnv.game.getBlockInFrontOfPlayer()[1] == rawEnv.game.getNextWoodBlock() \
                #         and not rawEnv.game.attackBlock(0.1):
                while a[1] == b and not c:
                    env.render()
                    for ii in range(int(1 / 0.1)):  # 0.1*10 = 1tick
                        Physics.step(env.game, 0.1)
                    a = env.game.getBlockInFrontOfPlayer()
                    b = env.game.getNextWoodBlock()
                    c = env.game.attackBlock(0.1)

                if env.game.getWoodLeft():
                    env.state["done"] = False
                    if env._isDone():
                        env.render()
                        input("Continue...")
                        obs = env.reset()
                else:
                    env.render()
                    input("Continue...")
                    obs = env.reset()

            # Plotting reward
            plt.scatter(i, cumulativeReward)

            # print("obs: ", obs)
            # input("Press any key to continue...")
            plt.pause(0.05)
            # sleep(0.05)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        pass
    except Exception as e:
        print("EXCEPTION!")
        raise e

    print("End")
    plt.show()
    env.close()


if __name__ == '__main__':
    main()
