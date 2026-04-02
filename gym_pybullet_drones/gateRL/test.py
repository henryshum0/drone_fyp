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
from gym_pybullet_drones.gateRL.waypoints.test_templates import *
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, EnvStateType
from gym_pybullet_drones.gateRL.train import filename, DEFAULT_PYB_FREQ, DEFAULT_CTRL_FREQ,  DEFAULT_N_ENVS, REWARD_WEIGHTS

def test(model_path):
    env_state_kwargs = dict(
            waypoints_templates=[TestTemplate1(), TestTemplate2(), TestTemplate3(), TestTemplate4()],
            K=120,
            low=-20,
            high=20,
            T=0.075
    )
    eval_env = GateRLEnv(
                        reward_weights=REWARD_WEIGHTS,
                        pyb_freq=200,
                         ctrl_freq=200,
                         episode_len_sec=20,
                         gui=True,
                         debug=True,
                         debug_pause=True,
                         train=False,
                         use_reward_shaping=True,
                         env_state_manager_kwargs=env_state_kwargs,
                         env_state_type=EnvStateType.ENV_STATE1
                        )
    eval_env = Monitor(eval_env, filename+'/eval/')
    model = PPO.load(model_path)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    
if __name__ == "__main__":
    model = "/home/henryshum0/drone_fyp/gym_pybullet_drones/gateRL/results/gate-03.19.2026_23.10.45/checkpoints/ppo_checkpoint_232000000_steps.zip"
    test(model)