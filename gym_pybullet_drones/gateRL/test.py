import os
import time
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from gym_pybullet_drones.gateRL.gateRLEnv import GateRLEnv
from gym_pybullet_drones.gateRL.waypoints import waypoints1, waypoints_figure8
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.gateRL.train import filename, DEFAULT_PYB_FREQ, DEFAULT_CTRL_FREQ,  DEFAULT_N_ENVS

def test(model_path):
    eval_env = GateRLEnv(
                        waypoints=waypoints_figure8,
                        pyb_freq=DEFAULT_PYB_FREQ,
                         ctrl_freq=DEFAULT_CTRL_FREQ,
                         episode_len_sec=5,
                         gui=True,
                         debug=True,
                         debug_pause=True,
                         train=False,
                        )
    eval_env = Monitor(eval_env, filename+'/eval/')
    model = PPO.load(model_path)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    
if __name__ == "__main__":
    model = "/home/henryshum0/drone_fyp/gym_pybullet_drones/gateRL/results/gate-03.01.2026_13.13.53/best_model/best_model.zip"
    test(model)