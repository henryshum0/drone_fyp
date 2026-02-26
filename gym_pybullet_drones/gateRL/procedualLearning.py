from copy import deepcopy
import random
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch
from transforms3d.euler import euler2quat, mat2euler
from transforms3d.quaternions import rotate_vector, qconjugate

from gym_pybullet_drones.gateRL.waypoints import waypoints1
class ProcedualLearning(BaseCallback):
    def __init__(self, 
                 waypoints, 
                 exp_buffer_size = 1000000,
                 init_buffer_size = 10000,
                 adap_buffer_size = 500,
                 p_init = 0.7,
                 p_adap = 0.2,
                 n_moderate=100,
                 top=0.3,

                 low=0.,
                 high=0.1,
                 verbose=0,
                 K_init=1,
                 K_step=1,
                 K_max=50,
                 K_schedule_base=1.5,
                 K_schedule_start_updates=10,
                 max_episode_len_sec=20,
                 initial_episode_len = 0.5,
                 delta_t = 0.01,
                 ):
        
        super(ProcedualLearning, self).__init__(verbose)
        

        # Constants
        self.DELTA_T = delta_t
        # buffer params
        self.experience_buffer_size = exp_buffer_size
        self.init_buffer_size = init_buffer_size
        self.adaptive_buffer_size = adap_buffer_size  # determines the number of init state sampled per flat state
        self.initial_state_buffer = []
        self.experience_buffer = []
        self.adaptive_buffer = []
        self.trj_buffer = []
        self.P_INIT = p_init
        self.P_ADAP = p_adap
        self.P_BUFF = 1 - p_init - p_adap
        

        # schedules K
        self.K_init = K_init
        self.K_step = K_step
        self.K = K_init
        self.K_max = K_max
        self.K_schedule_base = K_schedule_base
        self.K_schedule_start_updates = K_schedule_start_updates

        # flat state expansion params
        self.low = low
        self.high = high
        self.n_moderate = n_moderate
        self.TOP = top

        # waypoints and spawns
        self.waypoint_xyzs = waypoints["pos"]
        self.waypoint_rpys = waypoints["rpy"]
        self.waypoint_quats = np.array([euler2quat(*rpy) for rpy in self.waypoint_rpys])
        self.predefined_spawns = waypoints["spawn"]
        
        # Track K scheduling
        self.n_K_updates = 0
        self.last_K_update_step = 0
        
        # Track episode length scheduling
        self.MAX_EPISODE_LEN_SEC = max_episode_len_sec
        self.INITIAL_EPISODE_LEN_SEC = initial_episode_len
        self.accumulated_dt = 0.0
        self.waypoint_travel_time = None  # Will be calculated in _on_training_start
        

        self._init_buffers()
        
    def _on_step(self):
        # In SB3 OnPolicyAlgorithm.collect_rollouts(), callback locals are updated
        # right after `env.step()`, but before `self._last_obs = new_obs`.
        # So `infos` corresponds to the post-step transition and is naturally aligned
        # with `new_obs` (s_{t+1}), not with `self.model._last_obs` (s_t).
        new_obs = self.locals.get("new_obs", None)

        # Compute V(new_obs) so that (obs, value, next_waypoints) are aligned.
        values_np = None
        if new_obs is not None:
            with torch.no_grad():
                obs_tensor, _ = self.model.policy.obs_to_tensor(new_obs)
                values = self.model.policy.predict_values(obs_tensor)
            values_np = values.detach().cpu().numpy()
            infos = self.locals.get("infos", [])
            next_waypoints = tuple(info.get("next_waypoints", None) for info in infos)
            accelerations = tuple(info.get("acc", None) for info in infos)
            assert all([nw is not None for nw in next_waypoints]), "next_waypoints should be provided in info"
            assert new_obs.shape[0] == values_np.shape[0] == len(next_waypoints) == len(accelerations), "Batch size of new_obs, values, next_waypoints and accelerations should be the same"
            for single_obs, single_value, single_next_waypoints, acc in zip(new_obs, values_np, next_waypoints, accelerations):
                self.trj_buffer.append(
                    {
                        "obs": single_obs,
                        "value": single_value,
                        "next_waypoints": tuple(single_next_waypoints),
                        "acc": acc,
                    }
                )
        return True
    
    def _on_training_start(self):

        print(f"Initbuffer size: {len(self.initial_state_buffer)}, "
              f"Adap buffer size: {len(self.adaptive_buffer)}, " 
              f"Exp buffer size: {len(self.experience_buffer)}")
        
        # Set initial episode length for all environments
        self.training_env.env_method("set_episode_len", self.INITIAL_EPISODE_LEN_SEC)
        
        # Verify the setting worked
        current_episode_lens = self.training_env.get_attr("episode_len_sec")
        if self.verbose > 0:
            print(f"[ProcedualLearning] Initial episode length set to: {current_episode_lens[0]:.3f}s")
        
        # Calculate max distance between consecutive waypoints
        max_dist = 0.0
        for i in range(len(self.waypoint_xyzs) - 1):
            dist = np.linalg.norm(self.waypoint_xyzs[i+1] - self.waypoint_xyzs[i])
            max_dist = max(max_dist, dist)
        
        # Estimate travel time with safety margin
        # Assuming typical cruise velocity of 5 m/s (from _flat_from_waypoint)
        # Add 50% safety margin for acceleration/deceleration
        typical_velocity = 5.0  # m/s
        self.waypoint_travel_time = (max_dist / typical_velocity) * 1.5
        
        if self.verbose > 0:
            print(f"[ProcedualLearning] Max waypoint distance: {max_dist:.2f}m, "
                  f"Estimated travel time: {self.waypoint_travel_time:.2f}s")
        
        return True
    
        
    def _on_rollout_end(self):
        # Update K schedule based on policy updates
        self.schedule_K()
        self._fill_init_state_buffer()
        self._fill_adap_buffer()
        self._fill_exp_buffer()

        assert self.adaptive_buffer != [], "Adaptive buffer should not be empty after rollout end"
        self.trj_buffer = []
        return True
    
    def sample_spawn(self, network_step_counter, training=True):
        prb = np.random.uniform(0, 1)
        if training:
            if prb < self.P_ADAP:
                assert len(self.adaptive_buffer) > 0, "Adaptive buffer is empty, cannot sample spawn"
                spawn = random.choice(self.adaptive_buffer)
            elif prb < self.P_ADAP + self.P_BUFF:
                assert len(self.experience_buffer) > 0, "Experience buffer is empty, cannot sample spawn"
                spawn = random.choice(self.experience_buffer)
            else:
                # predefined initial states
                spawn = random.choice(self.initial_state_buffer)
        else:
            spawn = self.predefined_spawns[0]

        return spawn    

    def _init_buffers(self):
        # init initial state buffer with waypoints
        self._fill_init_state_buffer()
        self.experience_buffer = deepcopy(self.initial_state_buffer)
        self.adaptive_buffer = deepcopy(self.initial_state_buffer)
        self.trj_buffer = []

    def _fill_init_state_buffer(self):
        n_spawns_per_waypoint = self.init_buffer_size // len(self.waypoint_xyzs)
        for idx in range(len(self.waypoint_xyzs)):
            flat = self._flat_from_waypoint(idx, self.waypoint_xyzs, self.waypoint_rpys)
            for _ in range(n_spawns_per_waypoint):
                expanded_flat = self.expand_flat(flat)
                spawn = self._spawn_from_flat(expanded_flat)
                self.initial_state_buffer.append(spawn)

    def _fill_adap_buffer(self):
        self.adaptive_buffer = []
        n_samples = self.adaptive_buffer_size // self.n_moderate
        for moderate in self._get_moderate_trj():
            flat = self._flat_from_trj_sample(moderate)
            for _ in range(n_samples):
                expanded_flat = self.expand_flat(flat)
                spawn = self._spawn_from_flat(expanded_flat)
                self.adaptive_buffer.append(spawn)

    def _fill_exp_buffer(self):
        self.experience_buffer.extend(self.adaptive_buffer)
        if len(self.experience_buffer) > self.experience_buffer_size:
            self.experience_buffer = self.experience_buffer[-self.experience_buffer_size:]
            
    def _get_moderate_trj(self):
        sorted_list = sorted(self.trj_buffer, key=lambda x: x["value"])
        top = int(len(sorted_list) * self.TOP)
        if top + self.n_moderate > len(sorted_list):
            moderates = sorted_list[top:]
        else:
            moderates = sorted_list[top:top+self.n_moderate]
        assert len(moderates) != 0, "Moderate trajectories should not be empty"
        return moderates
    
    def schedule_K(self):
        """
        Update K based on the number of policy updates with exponentially decreasing update frequency.
        
        The interval between K updates grows exponentially: 
        update_interval = K_schedule_base^(n_K_updates) * K_schedule_start_updates
        
        This allows K to increase gradually (making the task harder) while the update rate 
        decreases exponentially to stabilize training at harder settings.
        """
        if self.K >= self.K_max:
            return  # K has reached maximum, no further updates needed
        
        # Calculate the required steps since last update using exponential growth
        required_steps_since_last_update = int(
            self.K_schedule_start_updates * (self.K_schedule_base ** self.n_K_updates)
        )
        
        # Check if enough policy updates have occurred
        steps_since_last_update = self.num_timesteps - self.last_K_update_step
        
        if steps_since_last_update >= required_steps_since_last_update:
            # Update K (increase by K_step to make task progressively harder)
            old_K = self.K
            self.K = min(self.K + self.K_step, self.K_max)
            
            # compensate for increased difficulty by increasing episode length
            current_episode_len = self.training_env.get_attr("episode_len_sec")[0]
            if current_episode_len < self.MAX_EPISODE_LEN_SEC:
                new_len = min(current_episode_len + self.DELTA_T * self.K_step, self.MAX_EPISODE_LEN_SEC)
                self.training_env.env_method("set_episode_len", new_len)
            self.accumulated_dt += self.DELTA_T * self.K_step
            
            # Update tracking variables
            self.last_K_update_step = self.num_timesteps
            self.n_K_updates += 1
            
            if self.verbose > 0:
                print(f"[ProcedualLearning] K updated from {old_K} to {self.K} "
                      f"at timestep {self.num_timesteps} "
                      f"(update #{self.n_K_updates}, "
                      f"next update in ~{int(self.K_schedule_start_updates * (self.K_schedule_base ** self.n_K_updates))} steps)")
            
            # Check if episode length needs updating
            self.schedule_episode_length()
        
        return self.K
    
    def schedule_episode_length(self):
        """
        Update episode length when accumulated dt reaches the estimated waypoint travel time.
        
        This ensures the drone has sufficient time to reach waypoints as K increases difficulty.
        Each K update accumulates DELTA_T, and when the threshold is reached, episode length
        increases by the accumulated amount.
        """
        # Only update episode length when accumulated time is significant
        if self.accumulated_dt >= self.waypoint_travel_time :
            current_episode_len = self.training_env.get_attr("episode_len_sec")[0]
            if current_episode_len + self.accumulated_dt > self.MAX_EPISODE_LEN_SEC:
                new_episode_len = self.MAX_EPISODE_LEN_SEC
            else:
                new_episode_len = current_episode_len + self.accumulated_dt
            
            # Update episode length for all environments
            self.training_env.env_method("set_episode_len", new_episode_len)
            
            if self.verbose > 0:
                print(f"[ProcedualLearning] Episode length updated: "
                      f"{current_episode_len:.3f}s -> {new_episode_len:.3f}s "
                      f"(+{self.accumulated_dt:.3f}s)")
            
            # Reset accumulator
            self.accumulated_dt = 0.0
        else:
            if self.verbose > 0:
                print(f"[ProcedualLearning] Episode length accumulator: "
                      f"{self.accumulated_dt:.3f}s / {self.waypoint_travel_time:.2f}s "
                      f"(threshold not reached yet)")
                
    def _spawn_from_flat(self, flat):
        p, v, a, next_waypoints = flat
        rpy = self._get_spawn_rpy(v, a)
        spawn = {"pos": p, "vel": v, "acc": a, "rpy": rpy, "next_waypoints": next_waypoints}
        return spawn

    def _get_spawn_rpy(self, v, a):
        z = (a - np.array([0, 0, -9.81])) / np.linalg.norm(a - np.array([0, 0, -9.81]))
        x = (v - np.dot(v, z) * z) / np.linalg.norm(v - np.dot(v, z) * z)
        y = np.cross(z, x)
        R = np.vstack((x, y, z)).T
        return mat2euler(R)
    
    def _flat_from_trj_sample(self, trj_sample):
        p = trj_sample["obs"][0, 14:17]
        v_b = trj_sample["obs"][0, 21:24]
        quat = trj_sample["obs"][0, 17:21]
        v = local_to_world(v_b, quat)
        a = trj_sample["acc"]
        next_waypoints = trj_sample["next_waypoints"]
        return p, v, a, next_waypoints
    
    def _flat_from_waypoint(self, idx, waypoint_xyzs, waypoint_rpys):
        if idx <= len(waypoint_xyzs) - 2:
            next_waypoints = (idx, idx+1)
        else:    
            next_waypoints = (idx, 0)
        p = waypoint_xyzs[idx]
        a = np.array([0, 0, 0])
        v = np.array([5, 0, 0])
        a = local_to_world(a, waypoint_rpys[idx])
        v = local_to_world(v, waypoint_rpys[idx])
        return p, v, a, next_waypoints
        
    def expand_flat(self, flat):
        p, v, a, next_waypoints = flat
        for _ in range(self.K):
            j = np.random.uniform(low=self.low, high=self.high, size=(3,))
            a = a - j * self.DELTA_T
            v = v - a * self.DELTA_T - j * self.DELTA_T**2 / 2
            p = p - v * self.DELTA_T - a * self.DELTA_T**2 / 2 - j * self.DELTA_T**3 / 6
        return p, v, a, next_waypoints

def world_to_local(vec, ori):
    assert len(vec) == 3 and (len(ori) == 3 or len(ori) == 4)
    if len(ori) == 3:
        q = euler2quat(*ori)
    else:
        q = ori
    return rotate_vector(vec, qconjugate(q))

def local_to_world(vec, ori):
    assert len(vec) == 3 and (len(ori) == 3 or len(ori) == 4)
    if len(ori) == 3:
        q = euler2quat(*ori)
    else:
        q = ori
    return rotate_vector(vec, q)
            
            
if __name__ == "__main__":
    callback = ProcedualLearning( waypoints=waypoints1, buffer_size=1000)
    for spawn in callback.experience_buffer:
        print(spawn, "\n")