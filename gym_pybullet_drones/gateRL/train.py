import os
import shutil
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
from gym_pybullet_drones.gateRL.waypoints import waypoints1, waypoints_figure8
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_OUTPUT_FOLDER = 'results'
MAX_EPISODE_LEN_SEC = 20
INITIAL_EPISODE_LEN_SEC = 4
N_STEPS = 2048
BATCH_SIZE = 256
DEFAULT_PYB_FREQ = 500
DEFAULT_CTRL_FREQ = 500
DEFAULT_NETWORK_FREQ = 100
DEFAULT_EPISODE = 30000
DEFAULT_N_ENVS = 100
USE_REWARD_SHAPING = False
USE_TENSORBOARD = True
LOAD_MODEL = False
LOAD_MODEL_PATH = "/home/henryshum0/drone_fyp/gym_pybullet_drones/gateRL/results/gate-03.02.2026_20.48.55/final_model/model.zip"

filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'gate-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
if not os.path.exists(filename):
    os.makedirs(filename+'/')


def save_training_file_snapshot(run_dir):
    """Copy key training/source files into the run folder for reproducibility."""
    gate_rl_dir = os.path.dirname(__file__)
    pkg_root_dir = os.path.abspath(os.path.join(gate_rl_dir, ".."))
    snapshot_dir = os.path.join(run_dir, "source_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)

    files_to_copy = [
        os.path.join(gate_rl_dir, "train.py"),
        os.path.join(gate_rl_dir, "gateRLEnv.py"),
        os.path.join(gate_rl_dir, "procedualLearning.py"),
        os.path.join(pkg_root_dir, "control", "CustomCTBRControl.py"),
        os.path.join(gate_rl_dir, "waypoints.py"),
    ]

    copied = []
    for src_path in files_to_copy:
        if not os.path.isfile(src_path):
            print(f"[WARN] Snapshot source file not found: {src_path}")
            continue
        dst_path = os.path.join(snapshot_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        copied.append(dst_path)

    print(f"[INFO] Saved {len(copied)} source file(s) to: {snapshot_dir}")
    for f in copied:
        print(f"       - {f}")

def run():

    save_training_file_snapshot(filename)

    monitor_dir = filename+'/train/'
    procedual_learning_callback = ProcedualLearning(waypoints=waypoints_figure8,
                                  exp_buffer_size=500000,
                                  init_buffer_size=50000,
                                  low=-3,
                                  high=3,
                                  verbose=1,
                                  p_init=1,
                                  K_init = 30,
                                  step_K=10,
                                  K_max=200,
                                  K_schedule_base=0.95,
                                  K_schedule_start_updates=20,
                                  initial_episode_len = INITIAL_EPISODE_LEN_SEC,
                                  episode_len_update_rollout_interval=1,
                                  episode_len_update_close_ratio=0.7,
                                  max_episode_len_sec=MAX_EPISODE_LEN_SEC,
                                  episode_len_step=.5,
                                  delta_t = 1/DEFAULT_NETWORK_FREQ,
                                  )
    train_env = make_vec_env(GateRLEnv,
                             env_kwargs=dict(waypoints=waypoints_figure8,
                                             procedual_learning=procedual_learning_callback,
                                             pyb_freq=DEFAULT_PYB_FREQ,
                                             ctrl_freq=DEFAULT_CTRL_FREQ,
                                             episode_len_sec=MAX_EPISODE_LEN_SEC,
                                             use_reward_shaping=USE_REWARD_SHAPING,
                                             gui=False,
                                             debug=False,
                                             debug_pause=False,),
                                             n_envs=DEFAULT_N_ENVS,
                                             seed=1,
                                             monitor_dir=monitor_dir,    
                             )

    eval_env = GateRLEnv(
                         waypoints=waypoints_figure8,
                         pyb_freq=DEFAULT_PYB_FREQ,
                         ctrl_freq=DEFAULT_CTRL_FREQ,
                         episode_len_sec=MAX_EPISODE_LEN_SEC,
                         use_reward_shaping=USE_REWARD_SHAPING,
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
    if LOAD_MODEL:
        model = PPO.load(LOAD_MODEL_PATH, env=train_env, device='cuda')
        print(f"Loaded model from {LOAD_MODEL_PATH}")
    else:

        model=PPO('MlpPolicy',
                train_env,
                verbose=2,
                policy_kwargs=policy_kwargs,
                device='cuda',
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                tensorboard_log=filename+'/tensorboard/' if USE_TENSORBOARD else None,
                seed=1,
                ent_coef= 0.01,
                )
    eval_callback = EvalCallback(eval_env=eval_env,
                                 best_model_save_path=filename+'/best_model/',
                                 log_path=filename+'/logs/',
                                 deterministic=True,
                                 render=False,
                                 verbose=1,
                                 n_eval_episodes=5,
                                 )
    
    
    model.learn(total_timesteps=DEFAULT_EPISODE*DEFAULT_NETWORK_FREQ*MAX_EPISODE_LEN_SEC,
                callback=[procedual_learning_callback, eval_callback],
                log_interval=100)
    results = load_results(monitor_dir)
    model.save(filename+'/final_model/model.zip')
    print("saved final model to:", filename+'/final_model/model.zip')



if __name__ == "__main__":
    run()