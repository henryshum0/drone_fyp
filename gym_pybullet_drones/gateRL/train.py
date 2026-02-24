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
from gym_pybullet_drones.gateRL.procedualLearning import ProcedualLearning
from gym_pybullet_drones.gateRL.waypoints import waypoints1
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_EPISODE_LEN_SEC = 1
N_STEPS = 2000
DEFAULT_PYB_FREQ = 500
DEFAULT_CTRL_FREQ = 500
DEFAULT_NETWORK_FREQ = 100
DEFAULT_EPISODE = 100000
DEFAULT_N_ENVS = 1

filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'gate-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
if not os.path.exists(filename):
    os.makedirs(filename+'/')

def run():


    monitor_dir = filename+'/train/'
    procedual_learning_callback = ProcedualLearning(waypoints=waypoints1,
                                  buffer_size=100000,
                                  n_moderate=100,
                                  K=5,
                                  low=0.,
                                  high=0.2,
                                  verbose=0,
                                  )
    train_env = make_vec_env(GateRLEnv,
                             env_kwargs=dict(waypoints=waypoints1,
                                             procedual_learning=procedual_learning_callback,
                                             pyb_freq=DEFAULT_PYB_FREQ,
                                             ctrl_freq=DEFAULT_CTRL_FREQ,
                                             episode_len_sec=DEFAULT_EPISODE_LEN_SEC,
                                             gui=False,
                                             debug=False,),
                                             n_envs=DEFAULT_N_ENVS,
                                             seed=1,
                                             monitor_dir=monitor_dir,    
                             )

    eval_env = GateRLEnv(
                        waypoints=waypoints1,
                        pyb_freq=DEFAULT_PYB_FREQ,
                         ctrl_freq=DEFAULT_CTRL_FREQ,
                         episode_len_sec=DEFAULT_EPISODE_LEN_SEC,
                         gui=False,
                         debug=False,
                         train=False,
                        )
    eval_env = Monitor(eval_env, filename+'/eval/')
    
    
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    policy_kwargs = dict(
        net_arch=[512, 512, 256, 128],
    )

    model=PPO('MlpPolicy',
              train_env,
              verbose=2,
              policy_kwargs=policy_kwargs,
              device='cuda',
              n_steps=N_STEPS,
              batch_size=int(DEFAULT_NETWORK_FREQ),
              
              )
    eval_callback = EvalCallback(eval_env=eval_env,
                                 best_model_save_path=filename+'/best_model/',
                                 log_path=filename+'/logs/',
                                 eval_freq=5000,
                                 deterministic=True,
                                 render=False,
                                 verbose=1,
                                 n_eval_episodes=5,
                                 )
    
    
    model.learn(total_timesteps=DEFAULT_EPISODE*DEFAULT_NETWORK_FREQ*DEFAULT_EPISODE_LEN_SEC,
                callback=[procedual_learning_callback, eval_callback],
                log_interval=100)
    results = load_results(monitor_dir)
    model.save(filename+'/final_model/model.zip')
    print("saved final model to:", filename+'/final_model/model.zip')



if __name__ == "__main__":
    run()