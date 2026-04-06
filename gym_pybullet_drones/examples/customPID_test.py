from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.CustomCTBRControl import CTBRPIDControl
from gym_pybullet_drones.control.CTBRControl import CTBRControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from transforms3d.quaternions import rotate_vector, qconjugate
import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 200
DEFAULT_CONTROL_FREQ_HZ = 200
DEFAULT_SETPOINT_FREQ_HZ = 200
DEFAULT_DURATION_SEC = 20
DEFAULT_LAG = 0.075
DEFAULT_OUTPUT_FOLDER = 'results'
NO_GRAVITY = False
USE_DEFAULT_CSV = False
NON_DEFAULT_CSV_PATH = "/home/henryshum0/drone_fyp/gym_pybullet_drones/gateRL/mocap_flight-04p-ellipse.csv"
CAMERA_FPS = 30
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
RENDER_FREQ_HZ = 10

def run(
        drone=DEFAULT_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        setpoint_freq_hz=DEFAULT_SETPOINT_FREQ_HZ,
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
                     no_gravity=NO_GRAVITY,
                     obstacles=True,
                     camera_enabled=False,
                     camera_fps=CAMERA_FPS,
                     camera_drone_id=0,
                     camera_record=False,
                     camera_width=CAMERA_WIDTH,
                     camera_height=CAMERA_HEIGHT,
                     )
    PYB_CLIENT = env.getPyBulletClient()
    
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=False
                    )
    
    desired_ctrl = CTBRControl(drone_model=drone,)
    custom_ctrl = CTBRPIDControl(drone_model=drone, ctrl_freq=control_freq_hz)

    if control_freq_hz % setpoint_freq_hz != 0:
        raise ValueError("control_freq_hz must be divisible by setpoint_freq_hz")
    steps_per_setpoint = control_freq_hz // setpoint_freq_hz
    
    desired_action = np.zeros((1, 4))
    action = np.zeros((1,4))
    rate_data = init_rate_tracking_data()
    
    if USE_DEFAULT_CSV:
        with open("/home/henryshum0/drone_fyp/gym_pybullet_drones/assets/beta-traj.csv", mode='r') as csv_file:
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
    else:
        with open(NON_DEFAULT_CSV_PATH, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            trajectory1 = iter([{
                "pos": np.array([
                    float(row["drone_x"]),
                    float(row["drone_y"]),
                    float(row["drone_z"]),
                ]),
                # "vel": np.array([
                #     float(row["v_x"]),
                #     float(row["v_y"]),
                #     float(row["v_z"]),
                # ]),
            } for row in csv_reader])

    

    START = time.time()
    
    dt = 1 / control_freq_hz
    t = 0.0
    sim_step_idx = 0
    render_every_n = max(1, int(control_freq_hz / RENDER_FREQ_HZ))
    while True:
        if t >= duration_sec:
            print(f"Reached desired duration of {duration_sec} seconds. Ending simulation.")
            break
        try:
            target = next(trajectory1)
            # Update target setpoint at a lower rate (e.g. 100Hz),
            # while body-rate controller still runs every control tick (e.g. 500Hz).
                # Decimate trajectory source: keep only every `setpoint_csv_step`-th CSV sample.
            if USE_DEFAULT_CSV:
                    desired_action[0:] = desired_ctrl.computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=env._getDroneStateVector(0),
                    target_pos=target["pos"],
                    target_vel=target["vel"],
                )
                
            else:
                desired_action[0:] = desired_ctrl.computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=env._getDroneStateVector(0),
                    target_pos=target["pos"] + [INIT_XYZ[0][0], INIT_XYZ[0][1], 0.0],
                    # target_vel=target["vel"],
                )
        except Exception as e:
            print("Error:", e)
            break

        for i in range(steps_per_setpoint):
            t += dt
            sim_step_idx += 1
            #### Step the simulation ###################################
            obs, reward, terminated, truncated, info = env.step(action)

            drone_ori = obs[0][3:7]
            drone_ori_wxyz = np.zeros(4)
            drone_ori_wxyz[0] = drone_ori[3]
            drone_ori_wxyz[1:4] = drone_ori[0:3]
            # p.getBaseVelocity() angular velocity is in world frame.
            # Convert world -> body using conjugate quaternion.
            cur_body_rate = rotate_vector(obs[0][13:16], qconjugate(drone_ori_wxyz))
            target_body_rate = np.zeros(3)
            motor_rpm = action[0].copy()
            _, imu_gyro_noisy_all = env.getIMUReadings(noisy=True)
            imu_gyro_noisy = imu_gyro_noisy_all[0]

            target_body_rate = desired_action[0, 1:4]
            action[0:] = custom_ctrl.compute_delayed_control(control_timestep=env.CTRL_TIMESTEP,
                                                thrust=desired_action[0, 0],
                                                cur_body_rate=cur_body_rate,
                                                target_body_rate=target_body_rate,
                                                T=DEFAULT_LAG,
                                                )
            motor_rpm = action[0].copy()
            # print("\ndesired_action: ", desired_action)
            # print("\nrpm: ", action)
            append_rate_tracking_data(
                rate_data,
                t,
                target_body_rate,
                cur_body_rate,
                motor_rpm,
                imu_gyro_noisy,
            )

            logger.log(drone=0,
                    timestamp=t,
                    state=obs[0],)
            if gui and (sim_step_idx % render_every_n == 0):
                env.render()
            
    logger.save()
    plot_rate_tracking(
        rate_data,
        output_folder,
        max_rpm=getattr(env, "MAX_RPM", None),
        show_plot=plot,
    )
    plot_imu_diagnostics(rate_data, output_folder, show_plot=plot)
    env.close()
    # if plot:
    #     logger.plot()


def init_rate_tracking_data():
    return {
        "t": [],
        "cmd": [[], [], []],  # p, q, r
        "act": [[], [], []],  # p, q, r
        "rpm": [[], [], [], []],  # m1, m2, m3, m4
        "imu_gyro_noisy": [[], [], []],
    }


def append_rate_tracking_data(
    rate_data,
    t,
    commanded_body_rate,
    actual_body_rate,
    motor_rpm,
    imu_gyro_noisy,
):
    rate_data["t"].append(float(t))
    for axis in range(3):
        rate_data["cmd"][axis].append(float(commanded_body_rate[axis]))
        rate_data["act"][axis].append(float(actual_body_rate[axis]))
        rate_data["imu_gyro_noisy"][axis].append(float(imu_gyro_noisy[axis]))
    for motor_idx in range(4):
        rate_data["rpm"][motor_idx].append(float(motor_rpm[motor_idx]))


def plot_imu_diagnostics(
    rate_data,
    output_folder,
    filename="custom_pid_imu_diagnostics.png",
    show_plot=True,
):
    if len(rate_data["t"]) == 0:
        print("[customPID_test] No IMU data collected; skipping IMU plot.")
        return

    t = np.array(rate_data["t"])
    fig, gyro_axis = plt.subplots(1, 1, figsize=(11, 4.5), sharex=True)
    axis_short = ["x", "y", "z"]

    for idx in range(3):
        gyro_axis.plot(
            t,
            rate_data["act"][idx],
            linewidth=1.3,
            label=f"Body rate {axis_short[idx]} (from state)",
        )
        gyro_axis.plot(
            t,
            rate_data["imu_gyro_noisy"][idx],
            linewidth=1.0,
            alpha=0.7,
            linestyle=":",
            label=f"IMU gyro noisy {axis_short[idx]}",
        )

    gyro_axis.set_ylabel("Gyro [rad/s]")
    gyro_axis.set_xlabel("Time [s]")
    gyro_axis.set_title("State Body Rate vs IMU Gyro Noisy")
    gyro_axis.grid(True, alpha=0.3)
    gyro_axis.legend(loc="upper right", ncol=2)

    fig.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, filename)
    fig.savefig(save_path, dpi=160)
    print(f"[customPID_test] Saved IMU diagnostics plot to: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_rate_tracking(
    rate_data,
    output_folder,
    filename="custom_pid_rate_tracking.png",
    max_rpm=None,
    show_plot=True,
):
    if len(rate_data["t"]) == 0:
        print("[customPID_test] No angular-rate data collected; skipping plot.")
        return

    t = np.array(rate_data["t"])
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    labels = ["p (roll rate)", "q (pitch rate)", "r (yaw rate)"]

    for idx, axis in enumerate(axes[:3]):
        axis.plot(t, rate_data["cmd"][idx], label="Commanded", linewidth=1.4)
        axis.plot(t, rate_data["act"][idx], label="Actual", linewidth=1.2, alpha=0.85)
        axis.set_ylabel(f"{labels[idx]} [rad/s]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    # Body-rate tracking error subplot
    err_axis = axes[3]
    axis_short = ["p", "q", "r"]
    for idx in range(3):
        cmd = np.array(rate_data["cmd"][idx])
        act = np.array(rate_data["act"][idx])
        err = cmd - act
        err_axis.plot(t, err, linewidth=1.1, label=f"e_{axis_short[idx]} = cmd-act")
    err_axis.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    err_axis.set_ylabel("Error [rad/s]")
    err_axis.set_title("Body-rate Tracking Error")
    err_axis.grid(True, alpha=0.3)
    err_axis.legend(loc="upper right", ncol=3)

    # Motor RPM subplot for saturation checks
    rpm_axis = axes[4]
    for motor_idx in range(4):
        rpm_axis.plot(t, rate_data["rpm"][motor_idx], linewidth=1.1, label=f"Motor {motor_idx + 1}")

    if max_rpm is not None:
        rpm_axis.axhline(max_rpm, color="red", linestyle="--", linewidth=1.2, label="MAX_RPM")
        rpm_axis.axhline(0.0, color="black", linestyle=":", linewidth=0.8)

    rpm_axis.set_ylabel("RPM")
    rpm_axis.set_title("Motor RPM Commands (Saturation Check)")
    rpm_axis.grid(True, alpha=0.3)
    rpm_axis.legend(loc="upper right", ncol=3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Commanded vs Actual Angular Rate + Tracking Error + Motor RPM")
    fig.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, filename)
    fig.savefig(save_path, dpi=160)
    print(f"[customPID_test] Saved combined rate/RPM plot to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


        
if __name__ == "__main__":
    run()