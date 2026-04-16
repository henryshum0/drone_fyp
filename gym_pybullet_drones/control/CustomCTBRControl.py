import numpy as np
import pybullet as p
import xml.etree.ElementTree as etxml
import pkg_resources


from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class PID_K():
    def __init__(self):
        self.ROLL_RATE_KP = 0.2
        self.ROLL_RATE_KI = 0.0
        self.ROLL_RATE_KD = 0.005
        self.ROLL_RATE_INTEGRATION_LIMIT = 33.3
        self.PITCH_RATE_KP = 0.2
        self.PITCH_RATE_KI = 0.0
        self.PITCH_RATE_KD = 0.005
        self.PITCH_RATE_INTEGRATION_LIMIT = 33.3
        self.YAW_RATE_KP = 0.2
        self.YAW_RATE_KI = 0.0
        self.YAW_RATE_KD = 0.00004
        self.YAW_RATE_INTEGRATION_LIMIT = 166.7
        self.CRAZYFLIE_CTRL_FREQ = 500 # for reference
        self.LOW_PASS_CUTOFF = 100


class CTBRPIDControl(BaseControl):

    def __init__(self, drone_model, ctrl_freq, g = 9.8, pid_k:PID_K=None):
        self.ctrl_freq = ctrl_freq
        self.PID_K = pid_k if pid_k is not None else PID_K()
        super().__init__(drone_model, g)
        # if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
        #     print("[ERROR] in CTBRPIDControl.__init__(), CTBRPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
        #     exit()

        self.KP = np.array([self.PID_K.ROLL_RATE_KP,
                            self.PID_K.PITCH_RATE_KP,
                            self.PID_K.YAW_RATE_KP])
        self.KI = np.array([self.PID_K.ROLL_RATE_KI,
                            self.PID_K.PITCH_RATE_KI,
                            self.PID_K.YAW_RATE_KI])
        self.KD = np.array([self.PID_K.ROLL_RATE_KD,
                            self.PID_K.PITCH_RATE_KD,
                            self.PID_K.YAW_RATE_KD])
        self.INTEGRATION_LIMIT = np.array([self.PID_K.ROLL_RATE_INTEGRATION_LIMIT,
                                           self.PID_K.PITCH_RATE_INTEGRATION_LIMIT,
                                           self.PID_K.YAW_RATE_INTEGRATION_LIMIT])

        # Keep controller conversion consistent with the active URDF.
        self.mass = self._getURDFParameter('m')
        self.L = self._getURDFParameter('arm')
        self.KF = self._getURDFParameter('kf')
        self.KM = self._getURDFParameter('km')
        self.THRUST2WEIGHT_RATIO = self._getURDFParameter('thrust2weight')
        # Physical motor limits from URDF.
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.mass * g) / (4.0 * self.KF))
        self.MAX_FORCE_PER_MOTOR = self.KF * self.MAX_RPM**2
        self.ARM_X = self.L / np.sqrt(2)
        self.ARM_Y = self.L / np.sqrt(2)
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_X_TORQUE = 2.0 * self.ARM_Y * self.MAX_FORCE_PER_MOTOR
            self.MAX_Y_TORQUE = 2.0 * self.ARM_X * self.MAX_FORCE_PER_MOTOR
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MAX_X_TORQUE = self.L * self.MAX_FORCE_PER_MOTOR
            self.MAX_Y_TORQUE = self.L * self.MAX_FORCE_PER_MOTOR
        elif self.DRONE_MODEL == DroneModel.RACE:
            self.ARM_X, self.ARM_Y = self._get_race_arm_components()
            self.MAX_X_TORQUE = 2.0 * self.ARM_Y * self.MAX_FORCE_PER_MOTOR
            self.MAX_Y_TORQUE = 2.0 * self.ARM_X * self.MAX_FORCE_PER_MOTOR
        else:
            raise ValueError("[ERROR] in CTBRPIDControl.__init__(), unsupported drone model")
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)

        # Allocation matrix A maps motor forces f=[f1,f2,f3,f4] to wrench w=[T,tau_x,tau_y,tau_z]:
        #   w = A @ f
        c = self.KM / self.KF
        if self.DRONE_MODEL == DroneModel.CF2X:
            l = self.L / np.sqrt(2)
            self.ALLOCATION_MATRIX = np.array([
                [1.0, 1.0, 1.0, 1.0],
                [-l,  -l,   l,   l],
                [-l,   l,   l,  -l],
                [-c,   c,  -c,   c],
            ], dtype=float)

        elif self.DRONE_MODEL == DroneModel.RACE:
            self.ALLOCATION_MATRIX = np.array([
                [1.0, 1.0, 1.0, 1.0],
                [self.ARM_Y,  self.ARM_Y,   -self.ARM_Y,   -self.ARM_Y],
                [-self.ARM_X,   self.ARM_X,   self.ARM_X,  -self.ARM_X],
                [c,   -c,  c,   -c],
            ], dtype=float)
    
        elif self.DRONE_MODEL == DroneModel.CF2P:
            l = self.L
            self.ALLOCATION_MATRIX = np.array([
                [1.0, 1.0, 1.0, 1.0],
                [0.0,   l, 0.0,  -l],
                [ -l, 0.0,   l, 0.0],
                [-c,   c,  -c,   c],
            ], dtype=float)
        else:
            raise ValueError("[ERROR] in CTBRPIDControl.__init__(), unsupported drone model")

        self.ALLOCATION_MATRIX_INV = np.linalg.inv(self.ALLOCATION_MATRIX)
            
        self.reset()

    def _get_race_arm_components(self):
        urdf = self.DRONE_MODEL.value + ".urdf"
        path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/' + urdf)
        urdf_tree = etxml.parse(path).getroot()
        abs_x = []
        abs_y = []
        for link in urdf_tree.findall('link'):
            name = link.attrib.get('name', '')
            if not (name.startswith('prop') and name.endswith('_link')):
                continue
            origin = link.find('./inertial/origin')
            if origin is None:
                continue
            xyz = [float(v) for v in origin.attrib.get('xyz', '0 0 0').split(' ')]
            if len(xyz) >= 2:
                abs_x.append(abs(xyz[0]))
                abs_y.append(abs(xyz[1]))

        if len(abs_x) == 0 or len(abs_y) == 0:
            default_arm = self.L / np.sqrt(2)
            return default_arm, default_arm
        return float(np.max(abs_x)), float(np.max(abs_y))

    def reset(self):
        super().reset()
        self.prev_rpm = np.zeros(4)
        self.prev_error = np.zeros(3)
        self.prev_body_rate = np.zeros(3)
        self.integral_error = np.zeros(3)
        self.lpdf2 = lpf2(self.ctrl_freq, self.PID_K.LOW_PASS_CUTOFF) # second order low pass filter for derivative term
        
    def computeControl(self, 
                       control_timestep,
                       thrust,
                       cur_body_rate,
                       target_body_rate
                       ):
        """
        custom PID controller for body rate control
        
        :param control_timestep: The time step at which control is computed
        :param thrust: desired thrust 
        :param cur_body_rate: measured body rate in rad/s
        :param target_body_rate: desired body rate in rad/s
        """
        desired_torque = self.getDesiredTorque(control_timestep, cur_body_rate, target_body_rate)
        # Keep torques within physical bounds.
        desired_torque[0] = np.clip(desired_torque[0], -self.MAX_X_TORQUE, self.MAX_X_TORQUE)
        desired_torque[1] = np.clip(desired_torque[1], -self.MAX_Y_TORQUE, self.MAX_Y_TORQUE)
        desired_torque[2] = np.clip(desired_torque[2], -self.MAX_Z_TORQUE, self.MAX_Z_TORQUE)

        # `thrust` is expected as mass-normalized acceleration [m/s^2]
        # (as returned by CTBRControl), convert to total force [N].
        if self.DRONE_MODEL in [DroneModel.CF2P, DroneModel.CF2X, DroneModel.RACE]:
            if thrust < 0:
                thrust = 0
            total_thrust = float(thrust) * self.mass
            total_thrust = np.clip(total_thrust, 0.0, 4.0 * self.MAX_FORCE_PER_MOTOR)

            wrench = np.array([
                total_thrust,
                desired_torque[0],
                desired_torque[1],
                desired_torque[2],
            ], dtype=float)

            motor_forces = self.ALLOCATION_MATRIX_INV @ wrench
            motor_forces = np.clip(motor_forces, 0.0, self.MAX_FORCE_PER_MOTOR)
            rpm = np.sqrt(motor_forces / self.KF)
            rpm = np.clip(rpm, 0.0, self.MAX_RPM)
            return rpm
        else:
            print("[ERROR] in CTBRPIDControl.computeControl(), CTBRPIDControl requires DroneModel.CF2X, DroneModel.CF2P, or DroneModel.RACE")
            exit()

    def compute_delayed_control(self,control_timestep,thrust,cur_body_rate,target_body_rate,T,):
        rpm = self.computeControl(control_timestep, thrust, cur_body_rate, target_body_rate)
        delayed_rpm = delay_response(self.prev_rpm, rpm, T=T, dt=control_timestep)
        self.prev_rpm = delayed_rpm.copy()
        return delayed_rpm


    def getDesiredTorque(self, control_timestep, cur_body_rate, target_body_rate):
        desired_torque = np.zeros(3)
        
        # Body-rate error is not an angle; do not wrap it by +/-pi.
        error = target_body_rate - cur_body_rate
        desired_torque += self.KP * error

        # D-term on measurement: provides damping and avoids setpoint-derivative kick.
        if control_timestep > 0.0:
            body_rate_derivative = (cur_body_rate - self.prev_body_rate) / control_timestep
        else:
            body_rate_derivative = np.zeros(3)
        derivative = self.lpdf2.lpf2Apply(body_rate_derivative)
        if (not np.isfinite(derivative).all()):
            derivative = np.zeros(3)
        desired_torque -= self.KD * derivative
        
        self.integral_error = self.integral_error + error * control_timestep
        # Anti-windup: limit integral state.
        self.integral_error = np.clip(self.integral_error, -self.INTEGRATION_LIMIT, self.INTEGRATION_LIMIT)
        desired_torque += self.KI * self.integral_error
        
        self.prev_error = error
        self.prev_body_rate = cur_body_rate
        
        return desired_torque

    def lowPassFilter(self, control_timestep, input, last_output):
        rc = 1 / (2 * np.pi * self.PID_K.LOW_PASS_CUTOFF)
        alpha = control_timestep / (rc + control_timestep)
        return last_output + alpha * (input - last_output)
    
    
class lpf2():
    def __init__(self, sample_freq, cutoff_freq):
        fr = sample_freq / cutoff_freq
        ohm = np.tan(np.pi / fr)
        c = 1.0 + 2.0 * np.cos(np.pi / 4.0) * ohm + ohm * ohm
        self.b0 = ohm * ohm / c
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0
        self.a1 = 2.0 * (ohm * ohm - 1.0) / c
        self.a2 = (1.0 - 2.0 * np.cos(np.pi / 4.0) * ohm + ohm * ohm) / c
        self.delay_element_1 = 0.0
        self.delay_element_2 = 0.0

    def lpf2Apply(self, sample):
        element_0 = sample - self.a1 * self.delay_element_1 - self.a2 * self.delay_element_2
        if (not np.isfinite(element_0).all()):
            element_0 = sample
        output = self.b0 * element_0 + self.b1 * self.delay_element_1 + self.b2 * self.delay_element_2
        self.delay_element_2 = self.delay_element_1
        self.delay_element_1 = element_0
        return output
    
def delay_response(y_t, u, T, dt):
    # First-order actuator lag: y[k+1] = u[k] + (y[k]-u[k]) * exp(-dt/T)
    y_t1 = u + (y_t - u) * np.exp(-dt / T)
    return y_t1