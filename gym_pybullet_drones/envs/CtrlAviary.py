import os
from datetime import datetime

import numpy as np
import pybullet as p
from gymnasium import spaces
from PIL import Image

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.sensors.camera import CameraSensor
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class CtrlAviary(BaseAviary):
    """Multi-drone environment class for control applications."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB_GND_DRAG_DW,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 no_gravity: bool=False,
                 output_folder='results',
                 camera_enabled: bool=False,
                 camera_fps: int=30,
                 camera_drone_id: int=0,
                 camera_record: bool=False,
                 camera_far: float=1000.0,
                 camera_width: int=320,
                 camera_height: int=240,
                 ):
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        no_gravity : bool, optional
            If True, sets gravity to zero in this environment.
        camera_enabled : bool, optional
            If True, enables onboard camera capture.
        camera_fps : int, optional
            Camera update rate in FPS.
        camera_drone_id : int, optional
            Index of the drone used for camera capture.
        camera_record : bool, optional
            If True, save captured camera frames to disk.
        camera_far : float, optional
            Far clipping plane for the onboard camera.
        camera_width : int, optional
            Camera image width in pixels.
        camera_height : int, optional
            Camera image height in pixels.
        camera_renderer : int | None, optional
            PyBullet camera renderer. If None, picks OpenGL in GUI and TinyRenderer in DIRECT.

        """
        self.NO_GRAVITY = no_gravity
        self.CAMERA_ENABLED = bool(camera_enabled)
        self.CAMERA_FPS = int(camera_fps)
        self.CAMERA_DRONE_ID = int(camera_drone_id)
        self.CAMERA_RECORD = bool(camera_record)
        self.CAMERA_FAR = float(camera_far)
        self.CAMERA_WIDTH = int(camera_width)
        self.CAMERA_HEIGHT = int(camera_height)

        if self.CAMERA_FPS <= 0:
            raise ValueError("camera_fps must be positive")
        if self.CAMERA_WIDTH <= 0 or self.CAMERA_HEIGHT <= 0:
            raise ValueError("camera_width and camera_height must be positive")

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder,
                         ground_plane=False,
                         )

        self.camera = None
        self._camera_frame_id = 0
        self._camera_record_path = None
        if self.CAMERA_ENABLED:
            if self.CAMERA_DRONE_ID < 0 or self.CAMERA_DRONE_ID >= self.NUM_DRONES:
                raise ValueError(
                    f"camera_drone_id must be in [0, {self.NUM_DRONES - 1}], got {self.CAMERA_DRONE_ID}"
                )

            cam_w = float(self.CAMERA_WIDTH)
            cam_h = float(self.CAMERA_HEIGHT)
            fov_rad = np.deg2rad(90.0)
            fx = 0.5 * cam_w / np.tan(0.5 * fov_rad)
            fy = fx
            cx = 0.5 * cam_w
            cy = 0.5 * cam_h

            self.camera = CameraSensor(
                width=int(cam_w),
                height=int(cam_h),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                near=0.03,
                far=self.CAMERA_FAR,
                client_id=self.CLIENT,
                control_freq=self.CTRL_FREQ,
                fps=self.CAMERA_FPS,
            )
            if self.CAMERA_RECORD:
                self._camera_record_path = os.path.join(
                    self.OUTPUT_FOLDER,
                    "drone_camera_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
                )
                os.makedirs(self._camera_record_path, exist_ok=True)

    ################################################################################

    def _housekeeping(self):
        """Housekeeping + optional zero-gravity override."""
        super()._housekeeping()
        if self.NO_GRAVITY:
            p.setGravity(0, 0, 0, physicsClientId=self.CLIENT)

    ################################################################################

    def step(self,
             action
             ):
        """Advances simulation and updates onboard camera at the requested FPS."""
        obs, reward, terminated, truncated, info = super().step(action)

        if self.camera is not None:
            pos = self.pos[self.CAMERA_DRONE_ID]
            quat = self.quat[self.CAMERA_DRONE_ID]
            rot_mat = np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
            cam_pos = pos + (rot_mat @ np.array([self.L, 0.0, 0.0], dtype=float))
            self.camera.update(cam_pos, quat)
            if self.CAMERA_RECORD and self._camera_record_path is not None and self.camera.new_frame_captured:
                rgb = self.camera.get_rgb()
                frame_path = os.path.join(self._camera_record_path, f"frame_{self._camera_frame_id:06d}.png")
                Image.fromarray(rgb, mode="RGB").save(frame_path)
                self._camera_frame_id += 1

        return obs, reward, terminated, truncated, info

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
