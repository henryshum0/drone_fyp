import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.GateRL import GateRLEnv
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gymnasium.utils.env_checker import check_env

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_EPISODE_LEN_SEC = 20
DEFAULT_PYB_FREQ = 500
DEFAULT_CTRL_FREQ = 500
DEFAULT_NETWORK_FREQ = 100
DEFAULT_EPISODE = 10000
DEFAULT_N_ENVS = 5

filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'gate-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
if not os.path.exists(filename):
    os.makedirs(filename+'/')

def run():

    # env = GateRLEnv(pyb_freq=DEFAULT_PYB_FREQ,
    #                       ctrl_freq=DEFAULT_CTRL_FREQ,
    #                       episode_len_sec=DEFAULT_EPISODE_LEN_SEC,
    #                       gui=False,)
    # check_env(env, warn=True)

    monitor_dir = filename+'/train/'
    train_env = make_vec_env(GateRLEnv,
                             env_kwargs=dict(pyb_freq=DEFAULT_PYB_FREQ,
                                             ctrl_freq=DEFAULT_CTRL_FREQ,
                                             episode_len_sec=DEFAULT_EPISODE_LEN_SEC,
                                             gui=False),
                                             n_envs=DEFAULT_N_ENVS,
                                             seed=1,
                                             monitor_dir=monitor_dir,
                             )
    eval_env = GateRLEnv(pyb_freq=DEFAULT_PYB_FREQ,
                         ctrl_freq=DEFAULT_CTRL_FREQ,
                         episode_len_sec=DEFAULT_EPISODE_LEN_SEC,
                         gui=True,
                         debug=False,
                        )
    
    
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    policy_kwargs = dict(
        net_arch=[512, 512, 256, 128],
        log_std_init=-.5,
    )

    model=PPO('MlpPolicy',
              train_env,
              verbose=1,
              policy_kwargs=policy_kwargs,
              device='cpu',
              n_steps=DEFAULT_NETWORK_FREQ*4,
              batch_size=int(DEFAULT_NETWORK_FREQ / 2),
              )
    eval_callback = EvalCallback(eval_env=eval_env,
                                 best_model_save_path=filename+'/best_model/',
                                 log_path=filename+'/logs/',
                                 eval_freq=10000,
                                 deterministic=True,
                                 render=False,
                                 verbose=1,
                                 n_eval_episodes=5,
                                 )
    
    model.learn(total_timesteps=DEFAULT_EPISODE*DEFAULT_NETWORK_FREQ*DEFAULT_EPISODE_LEN_SEC,
                callback=eval_callback,
                log_interval=100)
    results = load_results(monitor_dir)
    model.save(filename+'/final_model/model.zip')
    print("saved final model to:", filename+'/final_model/model.zip')
    
    with np.load(filename + "/logs/evaluations.npz") as data:
        timesteps = data['timesteps']
        results = data['results'].mean(axis=1)
        print("Data from evaluations.npz")
        for j in range(timesteps.shape[0]):
            print(f"{timesteps[j]},{results[j]}")
        plt.plot(timesteps, results, marker='o', linestyle='-', markersize=4)
        plt.xlabel('Training Steps')
        plt.ylabel('Episode Reward')
        plt.grid(True, alpha=0.6)
        plt.show()

    if os.path.isfile(filename+'/best_model/best_model.zip'):
        path = filename+'/best_model/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    test_env = GateRLEnv(pyb_freq=DEFAULT_PYB_FREQ,
                         ctrl_freq=DEFAULT_CTRL_FREQ,
                         network_freq=DEFAULT_NETWORK_FREQ,
                         episode_len_sec=DEFAULT_EPISODE_LEN_SEC,
                         gui=True,
                         record=True,)
    test_env_nogui = GateRLEnv(pyb_freq=DEFAULT_PYB_FREQ,
                               ctrl_freq=DEFAULT_CTRL_FREQ,
                               network_freq=DEFAULT_NETWORK_FREQ,
                               episode_len_sec=DEFAULT_EPISODE_LEN_SEC,
                               gui=False,)
    
    logger = Logger(logging_freq_hz=DEFAULT_NETWORK_FREQ,
                    num_drones=1,
                    output_folder=filename+'/test/',
                    colab=False
                    )
    
    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.NETWORK_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)

        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            control = np.zeros(12)
            control[0:4] = action
            logger.log(drone=0,
                       timestamp=i/test_env.NETWORK_FREQ,
                       state=test_env._getDroneStateVector(0).squeeze(),
                        control=control
                       )
        test_env.render()
        print("\nTerminated: ", terminated)
        sync(i, start, test_env.NETWORK_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=1)
    test_env.close()

    logger.plot()

def test(path:str):
    if not os.path.isfile(path):
        print("[ERROR]: no model under the specified path", filename)
        exit()

    model = PPO.load(path)

    test_env = GateRLEnv(pyb_freq=DEFAULT_PYB_FREQ,
                         ctrl_freq=DEFAULT_CTRL_FREQ,
                         network_freq=DEFAULT_NETWORK_FREQ,
                         episode_len_sec=DEFAULT_EPISODE_LEN_SEC,
                         gui=True,
                         record=True,)
    test_env_nogui = GateRLEnv(pyb_freq=DEFAULT_PYB_FREQ,
                               ctrl_freq=DEFAULT_CTRL_FREQ,
                               network_freq=DEFAULT_NETWORK_FREQ,
                               episode_len_sec=DEFAULT_EPISODE_LEN_SEC,
                               gui=False,)
    
    logger = Logger(logging_freq_hz=DEFAULT_NETWORK_FREQ,
                    num_drones=1,
                    output_folder=filename+'/test/',
                    colab=False
                    )
    
    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.NETWORK_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)

        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            control = np.zeros(12)
            control[0:4] = action
            logger.log(drone=0,
                       timestamp=i/test_env.NETWORK_FREQ,
                       state=test_env._getDroneStateVector(0).squeeze(),
                        control=control
                       )
        # test_env.render()
        print("\nTerminated: ", terminated)
        sync(i, start, test_env.NETWORK_TIMESTEP)
        if terminated:
            
            obs, info = test_env.reset(seed=1)
    test_env.close()

    logger.plot()
if __name__ == "__main__":
    # test("/home/henryshum0/gym-pybullet-drones/gym_pybullet_drones/examples/results/gate-01.11.2026_00.01.00/best_model/best_model.zip")
    run()