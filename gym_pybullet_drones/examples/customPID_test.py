from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.GateRL import GateRLEnv
from gym_pybullet_drones.control.CustomCTBRControl import CTBRPIDControl
from gym_pybullet_drones.control.CTBRControl import CTBRControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from transforms3d.quaternions import rotate_vector, qconjugate
import numpy as np
import time
import csv

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 500
DEFAULT_CONTROL_FREQ_HZ = 500
DEFAULT_NETWORK_FREQ = 100
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'results'

def run(
        drone=DEFAULT_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        num_drones=1,
        ):
    
    INIT_XYZ = np.array([[.3*i, .3*i, .1] for i in range(1,num_drones+1)])
    INIT_RPY = np.array([[.0, .0, .0] for _ in range(num_drones)])
    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     pyb_freq=simulation_freq_hz,
                     initial_xyzs=INIT_XYZ,
                     initial_rpys=INIT_RPY,
                     ctrl_freq=control_freq_hz,
                    #  network_freq=DEFAULT_NETWORK_FREQ,
                     gui=gui,
                     )
    PYB_CLIENT = env.getPyBulletClient()
    
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=False
                    )
    
    desired_ctrl = CTBRControl(drone_model=drone,)
    custom_ctrl = CTBRPIDControl(drone_model=drone, ctrl_freq=control_freq_hz)
    
    desired_action = np.zeros((1, 4))
    action = np.zeros((1,4))
    
    with open("../assets/beta-traj.csv", mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        trajectory1 = iter([{
            "pos": np.array([
                float(row["p_x"]),
                float(row["p_y"]),
                float(row["p_z"]),
            ]),
            "vel": np.array([
                float(row["v_x"]),
                float(row["v_y"]),
                float(row["v_z"]),
            ]),
        } for row in csv_reader])
    
    ARM_TIME = 1.
    TRAJ_TIME = 1.5
    START = time.time()
    for i in range(0, int(duration_sec * control_freq_hz)):
        t = i/env.CTRL_FREQ
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        if t > TRAJ_TIME:
            try:
                target = next(trajectory1)
                desired_action[0:] = desired_ctrl.computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=env._getDroneStateVector(0),
                    target_pos=target["pos"] + [INIT_XYZ[0][0], INIT_XYZ[0][1], 0.0],
                    target_vel=target["vel"],
                )
                drone_ori = obs[0][3:7]
                drone_ori_wxyz = np.zeros(4)
                drone_ori_wxyz[0] = drone_ori[3]
                drone_ori_wxyz[1:4] = drone_ori[0:3]
                cur_body_rate = rotate_vector(obs[0][13:16], drone_ori_wxyz)
                action[0:] = custom_ctrl.computeControl(control_timestep=env.CTRL_TIMESTEP,
                                                    thrust=desired_action[0, 0],
                                                    cur_body_rate=cur_body_rate,
                                                    target_body_rate=desired_action[0, 1:4],
                                                    )
                print("\ndesired_action: ", desired_action)
                print("\nrpm: ", action)
            except Exception as e:
                print("Error:", e)
                break
        logger.log(drone=0,
                   timestamp=t,
                   state=obs[0],)
        
        env.render()
        
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)
    
    env.close()
    logger.save()
    
    if plot:
        logger.plot()
        
if __name__ == "__main__":
    run()