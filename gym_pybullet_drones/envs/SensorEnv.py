import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.sensors.camera import CameraSensor
from gym_pybullet_drones.sensors.imu import IMU
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class SensorEnv(BaseAviary):
    """BaseAviary subclass exposing IMU and camera sensor streams."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB_GND_DRAG_DW,
        pyb_freq: int = 240,
        ctrl_freq: int = 240,
        gui=False,
        record=False,
        obstacles=False,
        user_debug_gui=True,
        output_folder="results",
        camera_enabled: bool = True,
        use_egl_renderer: bool = False,
        camera_fps: int = 30,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fov_deg: float = 90.0,
        camera_near: float = 0.03,
        camera_far: float = 1000.0,
        imu_enabled: bool = True,
        imu_accel_noise_std_range=(0.02, 0.02),
        imu_gyro_noise_std_range=(0.005, 0.005),
        imu_accel_bias_std_range=(0.0, 0.0),
        imu_gyro_bias_std_range=(0.0, 0.0),
    ):
        self.CAMERA_ENABLED = bool(camera_enabled)
        self.CAMERA_FPS = int(camera_fps)
        self.CAMERA_DRONE_ID = 0
        self.CAMERA_WIDTH = int(camera_width)
        self.CAMERA_HEIGHT = int(camera_height)
        self.CAMERA_FOV_DEG = float(camera_fov_deg)
        self.CAMERA_NEAR = float(camera_near)
        self.CAMERA_FAR = float(camera_far)

        self.IMU_ENABLED = bool(imu_enabled)
        self.IMU_ACCEL_NOISE_STD_RANGE = tuple(imu_accel_noise_std_range)
        self.IMU_GYRO_NOISE_STD_RANGE = tuple(imu_gyro_noise_std_range)
        self.IMU_ACCEL_BIAS_STD_RANGE = tuple(imu_accel_bias_std_range)
        self.IMU_GYRO_BIAS_STD_RANGE = tuple(imu_gyro_bias_std_range)

        if int(num_drones) != 1:
            raise ValueError("SensorEnv supports only a single drone (num_drones must be 1)")

        if self.CAMERA_FPS <= 0:
            raise ValueError("camera_fps must be positive")
        if self.CAMERA_WIDTH <= 0 or self.CAMERA_HEIGHT <= 0:
            raise ValueError("camera_width and camera_height must be positive")

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            use_egl_renderer=use_egl_renderer,
            obstacles=obstacles,
            user_debug_gui=user_debug_gui,
            output_folder=output_folder,
            compute_returns_per_step=False,
        )

        self.camera = None
        self.imu = None
        self.imu_acc_actual = np.zeros((3,), dtype=np.float32)
        self.imu_gyro_actual = np.zeros((3,), dtype=np.float32)
        self.imu_acc_noisy = np.zeros((3,), dtype=np.float32)
        self.imu_gyro_noisy = np.zeros((3,), dtype=np.float32)

        self._init_sensors()

    def _init_sensors(self):
        if self.CAMERA_ENABLED:
            if self.CAMERA_DRONE_ID < 0 or self.CAMERA_DRONE_ID >= self.NUM_DRONES:
                raise ValueError(
                    f"camera_drone_id must be in [0, {self.NUM_DRONES - 1}], got {self.CAMERA_DRONE_ID}"
                )
            fx = 0.5 * self.CAMERA_WIDTH / np.tan(0.5 * np.deg2rad(self.CAMERA_FOV_DEG))
            fy = fx
            cx = 0.5 * self.CAMERA_WIDTH
            cy = 0.5 * self.CAMERA_HEIGHT
            self.camera = CameraSensor(
                width=self.CAMERA_WIDTH,
                height=self.CAMERA_HEIGHT,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                near=self.CAMERA_NEAR,
                far=self.CAMERA_FAR,
                client_id=self.CLIENT,
                pyb_freq=self.PYB_FREQ,
                fps=self.CAMERA_FPS,
            )

        if self.IMU_ENABLED:
            self.imu = IMU(
                freq=self.CTRL_FREQ,
                pyb_freq=self.PYB_FREQ,
                client_id=self.CLIENT,
                accel_noise_std_range=self.IMU_ACCEL_NOISE_STD_RANGE,
                gyro_noise_std_range=self.IMU_GYRO_NOISE_STD_RANGE,
                accel_bias_std_range=self.IMU_ACCEL_BIAS_STD_RANGE,
                gyro_bias_std_range=self.IMU_GYRO_BIAS_STD_RANGE,
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._init_sensors()
        self._update_sensors()
        return self._computeObs(), self._computeInfo()

    def step(self, action):
        super().step(action)
        self._update_sensors()
        return self._computeObs(), self._computeReward(), self._computeTerminated(), self._computeTruncated(), self._computeInfo()

    def getIMUReadings(self, noisy: bool = True):
        if noisy:
            return self.imu_acc_noisy.copy(), self.imu_gyro_noisy.copy()
        return self.imu_acc_actual.copy(), self.imu_gyro_actual.copy()

    def getCameraFrames(self):
        if self.camera is None:
            return None, None
        rgb, depth = self.camera.get_frames()
        return rgb, depth
    
    def getSensorInfo(self):
        info = self._computeInfo()
        if self.camera is not None:
            info["camera_new_frame"] = bool(getattr(self.camera, "new_frame_captured", False))
        return info

    def _update_sensors(self):
        if self.IMU_ENABLED and self.imu is not None:
            self.imu.update_from_kinematics(
                vel_world=self.vel[0],
                ang_vel_world=self.ang_v[0],
                quat_xyzw=self.quat[0],
                step_counter=self.step_counter,
            )
            self.imu_acc_actual, self.imu_gyro_actual = self.imu.get_actual()
            self.imu_acc_noisy, self.imu_gyro_noisy = self.imu.get_noisy()
            self.imu_acc_actual = self.imu_acc_actual.astype(np.float32)
            self.imu_gyro_actual = self.imu_gyro_actual.astype(np.float32)
            self.imu_acc_noisy = self.imu_acc_noisy.astype(np.float32)
            self.imu_gyro_noisy = self.imu_gyro_noisy.astype(np.float32)

        if self.CAMERA_ENABLED and self.camera is not None:
            pos = self.pos[self.CAMERA_DRONE_ID]
            quat = self.quat[self.CAMERA_DRONE_ID]
            rot_mat = np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
            cam_pos = pos + (rot_mat @ np.array([self.L, 0.0, 0.0], dtype=float))
            
            self.camera.update(cam_pos, quat, self.step_counter)

    def _actionSpace(self):
        low = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _observationSpace(self):
        obs_dim = 26
        low = np.array([-np.inf] * obs_dim, dtype=np.float32)
        high = np.array([np.inf] * obs_dim, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        states = self._getDroneStateVector(0).astype(np.float32)
        if self.IMU_ENABLED:
            gyro = self.imu_gyro_noisy
            acc = self.imu_acc_noisy
        else:
            gyro = np.zeros((3,), dtype=np.float32)
            acc = np.zeros((3,), dtype=np.float32)
        return np.hstack((states, gyro, acc)).astype(np.float32)

    def _preprocessAction(self, action):
        act = np.asarray(action, dtype=np.float32)
        if act.shape == (1, 4):
            act = act[0]
        if act.shape != (4,):
            raise ValueError(f"Expected action shape (4,) or (1,4), got {act.shape}")
        act = np.clip(act, -1.0, 1.0)
        return self._normalizedActionToRPM(act).reshape(1, 4)

    def _computeReward(self):
        return -1.0

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        info = {
            "imu_enabled": self.IMU_ENABLED,
            "camera_enabled": self.CAMERA_ENABLED,
            "imu_acc_noisy": self.imu_acc_noisy.copy(),
            "imu_gyro_noisy": self.imu_gyro_noisy.copy(),
        }
        if self.camera is not None:
            info["camera_new_frame"] = bool(getattr(self.camera, "new_frame_captured", False))
        return info
