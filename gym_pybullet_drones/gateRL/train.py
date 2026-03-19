import os
import shutil
import time
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from gym_pybullet_drones.gateRL.gateRLEnv import GateRLEnv
from gym_pybullet_drones.gateRL.procedualLearning import ProcedualLearning
from gym_pybullet_drones.utils.enums import DroneModel, EnvStateType, ObservationType, ActionType
from gym_pybullet_drones.gateRL.waypoints.easy_templates import *
from gym_pybullet_drones.gateRL.waypoints.hard_templates import *
from gym_pybullet_drones.gateRL.waypoints.test_templates import *
from gym_pybullet_drones.gateRL.waypoints.train_templates import *

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_OUTPUT_FOLDER = 'results'
K_INIT = 50
K_STEP = 10
K_MAX = 200
K_SCHEDULE_BASE = 1.25
K_SCHEDULE_START_UPDATES = 3
FLAT_LOW = -20
FLAT_HIGH = 20
HARD_TEMPLATE_MAX_PCT = 0.0
HARD_TEMPLATE_MIN_PCT = 0.0
MAX_EPISODE_LEN_SEC = 4
INITIAL_EPISODE_LEN_SEC = 4
N_STEPS = 2048
BATCH_SIZE = 256
DEFAULT_PYB_FREQ = 200
DEFAULT_CTRL_FREQ = 200
DEFAULT_NETWORK_FREQ = 100
DEFAULT_EPISODE = 1500
DEFAULT_N_ENVS = 100
DEBUG=True
USE_REWARD_SHAPING = False
USE_TENSORBOARD = False
train_templates = [
    UpDownTemplate(),
    DownDownTemplate(),
    DownUpTemplate(),
    FrontBackTemplate(),
    FrontFrontTemplate(),
    SideSideTemplate1(),
    SideSideTemplate2(),
]
omni_template = [OmniTemplate()]
test_templates = [
    TestTemplate1(),
    TestTemplate2(),
    TestTemplate3(),
    TestTemplate4(),
]
REWARD_WEIGHTS = {
    'aero': 500,
    'pa': 1,
    'theta_error': 1,
    'aero_shaped': 1,
    'pa_shaped': 1,
    'theta_error_shaped': 1,
    'act': -0,
    'act_change': -1.5,
    'yaw': -5,
    'time_penalty': -0,
    'out_of_bound_penalty': -0,
    'timeout_penalty': -0,
}
ENV_STATE_TYPE = EnvStateType.ENV_STATE1
ENV_STATE_KWARGS = dict(
    waypoints_templates=omni_template,
    K=K_INIT,
    low=FLAT_LOW,
    high=FLAT_HIGH,
    history_p=0.3,
    T = 0.075,
)
LOAD_MODEL = True
LOAD_MODEL_PATH = "/home/henryshum0/drone_fyp/gym_pybullet_drones/gateRL/results/gate-03.18.2026_01.10.09/best_model/best_model.zip"

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
    procedual_learning_callback = ProcedualLearning(
                                  dt=1/DEFAULT_NETWORK_FREQ,            
                                  verbose=1,
                                  K_init = K_INIT,
                                  step_K=K_STEP,
                                  K_max=K_MAX,
                                  K_schedule_base=K_SCHEDULE_BASE,
                                  K_schedule_start_updates=K_SCHEDULE_START_UPDATES,
                                  initial_episode_len = INITIAL_EPISODE_LEN_SEC,
                                  episode_len_update_rollout_interval=1,
                                  episode_len_update_close_ratio=0.7,
                                  max_episode_len_sec=MAX_EPISODE_LEN_SEC,
                                  episode_len_step=.5,
                                  hard_template_max_pct=HARD_TEMPLATE_MAX_PCT,
                                  hard_template_min_pct=HARD_TEMPLATE_MIN_PCT,
                                  )
    train_env = make_vec_env(GateRLEnv,
                             env_kwargs=dict(
                                             env_state_type=ENV_STATE_TYPE,
                                             env_state_manager_kwargs=ENV_STATE_KWARGS,
                                             reward_weights=REWARD_WEIGHTS,
                                             pyb_freq=DEFAULT_PYB_FREQ,
                                             ctrl_freq=DEFAULT_CTRL_FREQ,
                                             episode_len_sec=MAX_EPISODE_LEN_SEC,
                                             use_reward_shaping=USE_REWARD_SHAPING,
                                             gui=DEBUG,
                                             debug=DEBUG,
                                             debug_pause=DEBUG,
                                             train=True,
                                             ),
                                             n_envs=DEFAULT_N_ENVS if not DEBUG else 1,
                                             seed=1,
                                             monitor_dir=monitor_dir,    
                             )

    eval_env = GateRLEnv(
                         env_state_type=ENV_STATE_TYPE,
                         env_state_manager_kwargs=ENV_STATE_KWARGS,
                         reward_weights=REWARD_WEIGHTS,
                         pyb_freq=DEFAULT_PYB_FREQ,
                         ctrl_freq=DEFAULT_CTRL_FREQ,
                         episode_len_sec=15,
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
                ent_coef= 0.008,
                )

    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=filename+'/checkpoints/',
                                         name_prefix='ppo_checkpoint')
    eval_callback = EvalCallback(eval_env=eval_env,
                                 best_model_save_path=filename+'/best_model/',
                                 log_path=filename+'/logs/',
                                 deterministic=True,
                                 render=False,
                                 verbose=1,
                                 n_eval_episodes=20,
                                 )
    
    
    model.learn(total_timesteps=N_STEPS*DEFAULT_N_ENVS*DEFAULT_EPISODE,
                callback=[procedual_learning_callback, eval_callback, checkpoint_callback],
                log_interval=100)
    results = load_results(monitor_dir)
    model.save(filename+'/final_model/model.zip')
    print("saved final model to:", filename+'/final_model/model.zip')



if __name__ == "__main__":
    run()