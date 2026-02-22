import numpy as np
import pybullet as p


from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

# default values for CrazyFlie2 
ROLL_RATE_KP = 250.0 * 180 / np.pi
ROLL_RATE_KI = 500.0 * 180 / np.pi
ROLL_RATE_KD = 2.5 * 180 / np.pi
ROLL_RATE_INTEGRATION_LIMIT = 33.3 * 180 / np.pi
PITCH_RATE_KP = 250.0 * 180 / np.pi
PITCH_RATE_KI = 500.0 * 180 / np.pi
PITCH_RATE_KD = 2.5 * 180 / np.pi
PITCH_RATE_INTEGRATION_LIMIT = 33.3 * 180 / np.pi
YAW_RATE_KP = 120.0 * 180 / np.pi
YAW_RATE_KI = 16.7 * 180 / np.pi
YAW_RATE_KD = 0.0 * 180 / np.pi
YAW_RATE_INTEGRATION_LIMIT = 166.7 * 180 / np.pi
CRAZYFLIE_CTRL_FREQ = 500 # for reference
LOW_PASS_CUTOFF = 30.0

class CTBRPIDControl(BaseControl):

    def __init__(self, drone_model, ctrl_freq, g = 9.8):
        self.ctrl_freq = ctrl_freq
        super().__init__(drone_model, g)
        # if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
        #     print("[ERROR] in CTBRPIDControl.__init__(), CTBRPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
        #     exit()

        
        self.KP = np.array([ROLL_RATE_KP, 
                            PITCH_RATE_KP, 
                            YAW_RATE_KP])
        self.KI = np.array([ROLL_RATE_KI, 
                            PITCH_RATE_KI,
                            YAW_RATE_KI])
        self.KD = np.array([ROLL_RATE_KD, 
                            PITCH_RATE_KD, 
                            YAW_RATE_KD])
        self.INTEGRATION_LIMIT = np.array([ROLL_RATE_INTEGRATION_LIMIT, 
                                           PITCH_RATE_INTEGRATION_LIMIT,
                                           YAW_RATE_INTEGRATION_LIMIT])

        self.mass = 0.027 # cf2x and cf2p mass in kg   
        self.KF = 3.16e-10 # cf2x and cf2p thrust coefficient
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
    
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
            
        self.reset()

    def reset(self):
        super().reset()
        self.prev_error = np.zeros(3)
        self.prev_body_rate = np.zeros(3)
        self.integral_error = np.zeros(3)
        self.lpdf2 = lpf2(self.ctrl_freq, LOW_PASS_CUTOFF) # second order low pass filter for derivative term
        
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
        desired_torque = np.clip(desired_torque, -3200, 3200)
        # `thrust` can be either a PWM baseline (as in DSLPIDControl) or a
        # normalized/acceleration value (as returned by CTBRControl). If it's
        # small (well below PWM ranges) convert it to the PWM baseline using
        # the same conversion DSLPIDControl uses: rpm = sqrt(F/(4*Kf));
        # pwm = (rpm - const)/scale. If it already looks like PWM, leave it.
        if self.DRONE_MODEL == DroneModel.CF2P or self.DRONE_MODEL == DroneModel.CF2X:
            if thrust < 0:
                thrust = 0
            if np.isscalar(thrust) and abs(thrust) < (self.MIN_PWM * 0.5):
                # treat `thrust` as acceleration (m/s^2) -> convert to force (N)
                scalar_thrust = max(0.0, thrust * self.mass)
                rpm_baseline = np.sqrt(scalar_thrust / (4.0 * self.KF)) 
                pwm_baseline = (rpm_baseline - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

            else:
                pwm_baseline = thrust
            pwm = pwm_baseline + np.dot(self.MIXER_MATRIX, desired_torque)
            pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
            return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        else:
            print("[ERROR] in CTBRPIDControl.computeControl(), CTBRPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()



    def getDesiredTorque(self, control_timestep, cur_body_rate, target_body_rate):
        desired_torque = np.zeros(3)
        
        error = target_body_rate - cur_body_rate
        if (error[2] > np.pi):
            error[2] -= 2*np.pi
        elif (error[2] < -np.pi):
            error[2] += 2*np.pi
        desired_torque += self.KP * error
        
        delta = - (error - self.prev_error)
        # if (delta[2] > np.pi):
        #     delta[2] -= 2*np.pi
        # elif (delta[2] < -np.pi):
        #     delta[2] += 2*np.pi
        derivative = self.lpdf2.lpf2Apply(delta / control_timestep)
        if (np.isnan(derivative).any()):
            derivative = np.zeros(3)
        desired_torque += self.KD * derivative
        
        self.integral_error = self.integral_error + error * control_timestep
        self.integral_error = np.clip(self.integral_error, -self.INTEGRATION_LIMIT, self.INTEGRATION_LIMIT)
        desired_torque += self.KI * self.integral_error
        
        self.prev_error = error
        self.prev_body_rate = cur_body_rate
        
        return desired_torque

    def lowPassFilter(self, control_timestep, input, last_output):
        rc = 1 / (2 * np.pi * LOW_PASS_CUTOFF)
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