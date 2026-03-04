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
                 init_buffer_size = 100000,
                 p_init = 0.7,
                 low=0.,
                 high=0.1,
                 verbose=0,
                 K_init=10,
                 step_K=10,
                 K_max=500,
                 K_schedule_base=1.1,
                 K_schedule_start_updates=1,
                 episode_len_update_rollout_interval = 1,
                 episode_len_update_close_ratio = 0.9,
                 max_episode_len_sec=20,
                 initial_episode_len = 0.5,
                 episode_len_step = 0.1,
                 delta_t = 0.01,
                 ):
        
        super(ProcedualLearning, self).__init__(verbose)
        

        # Constants
        self.DELTA_T = delta_t
        # buffer params
        self.experience_buffer_size = exp_buffer_size
        self.init_buffer_size = init_buffer_size
        self.initial_state_buffer_easy = []
        self.initial_state_buffer_hard = []
        self.experience_buffer = []
        self.P_INIT = p_init
        self.P_BUFF = 1 - p_init 
        

        # schedules K
        self.K_init = K_init
        self.step_K = step_K
        self.K = K_init
        self.K_max = K_max
        self.K_schedule_base = K_schedule_base
        self.K_schedule_start_updates = K_schedule_start_updates


        # flat state expansion params
        self.low = low
        self.high = high

        # waypoints and spawns
        self.waypoint_xyzs = waypoints["pos"]
        self.waypoint_rpys = waypoints["rpy"]
        self.waypoint_quats = np.array([euler2quat(*rpy) for rpy in self.waypoint_rpys])
        self.predefined_spawns = waypoints["spawn"]
        
        # Track K scheduling
        self.n_K_updates = 0
        self.last_K_update_rollout = 0
        self.n_rollouts = 0

        # Track per-rollout trajectories and cumulative rewards
        self.current_rollout_trajectories = []
        self.completed_rollout_trajectories = []
        self.ordered_rollout_trajectories = []
        
        # Track episode length scheduling
        self.MAX_EPISODE_LEN_SEC = max_episode_len_sec
        self.INITIAL_EPISODE_LEN_SEC = initial_episode_len
        self.EPISODE_LEN_STEP = episode_len_step
        self.EPISODE_LEN_UPDATE_ROLLOUT_INTERVAL = episode_len_update_rollout_interval
        self.EPISODE_LEN_UPDATE_CLOSE_RATIO = float(episode_len_update_close_ratio)
        self.last_episode_len_update = 0

        self._init_buffers()
        
    def _on_step(self):
        # In SB3 OnPolicyAlgorithm.collect_rollouts(), callback locals are updated
        # right after `env.step()`, but before `self._last_obs = new_obs`.
        # So `infos` corresponds to the post-step transition and is naturally aligned
        # with `new_obs` (s_{t+1}), not with `self.model._last_obs` (s_t).
        new_obs = self.locals.get("new_obs", None)

        # Compute V(new_obs) so that (obs, value, next_waypoints) are aligned.
        if new_obs is not None:

            infos = self.locals.get("infos", [])
            rewards = self.locals.get("rewards", None)
            dones = self.locals.get("dones", None)
            next_waypoints = tuple(info.get("next_waypoints", None) for info in infos)
            accelerations = tuple(info.get("acc", None) for info in infos)
            positions = tuple(info.get("pos", None) for info in infos)
            velocities = tuple(info.get("vel", None) for info in infos)
            assert all([nw is not None for nw in next_waypoints]), "next_waypoints should be provided in info"
            assert new_obs.shape[0] == len(next_waypoints) == len(accelerations), "Batch size of new_obs, values, next_waypoints and accelerations should be the same"

            if len(self.current_rollout_trajectories) == 0:
                self.current_rollout_trajectories = [
                    {"cum_reward": 0.0, "samples": []}
                    for _ in range(new_obs.shape[0])
                ]

            if rewards is None:
                rewards = np.zeros(new_obs.shape[0], dtype=float)
            if dones is None:
                dones = np.zeros(new_obs.shape[0], dtype=bool)



            for env_idx, (single_next_waypoints, pos, v, acc, rew, done) in enumerate(
                zip(next_waypoints, positions, velocities, accelerations, rewards, dones)
            ):
                traj = self.current_rollout_trajectories[env_idx]
                traj["cum_reward"] += float(rew)
                traj["samples"].append(
                    {
                        "next_waypoints": tuple(single_next_waypoints),
                        "pos": pos,
                        "vel": v,
                        "acc": acc,
                        "reward": float(rew),
                    }
                )

                if done:
                    self.completed_rollout_trajectories.append(deepcopy(traj))
                    self.current_rollout_trajectories[env_idx] = {"cum_reward": 0.0, "samples": []}
        return True
    
    def _on_training_start(self):

        print(f"Initbuffer (easy) size: {len(self.initial_state_buffer_easy)}, "
              f"Initbuffer (hard) size: {len(self.initial_state_buffer_hard)}, "
              f"Exp buffer size: {len(self.experience_buffer)}")

        k_update_rollout_steps = self.get_K_update_rollout_steps()
        print(f"[ProcedualLearning] K will be updated at rollout(s): {k_update_rollout_steps}")
        
        # Set initial episode length for all environments
        self.training_env.env_method("set_episode_len", self.INITIAL_EPISODE_LEN_SEC)
        
        # Verify the setting worked
        current_episode_lens = self.training_env.get_attr("episode_len_sec")
        if self.verbose > 0:
            print(f"[ProcedualLearning] Initial episode length set to: {current_episode_lens[0]:.3f}s")
        
        
        return True

    def get_K_update_rollout_steps(self):
        """Return rollout indices where `K` is expected to be updated.

        The schedule mirrors `schedule_K()` exactly:
        - update interval for the nth K-update is
          int(K_schedule_start_updates * K_schedule_base**n)
        - one K-update increments K by `step_K` until `K_max`.
        """
        rollout_steps = []
        simulated_k = self.K_init
        simulated_n_k_updates = 0
        simulated_last_k_update_rollout = 0

        while simulated_k < self.K_max:
            required_rollouts_since_last_update = int(
                self.K_schedule_start_updates * (self.K_schedule_base ** simulated_n_k_updates)
            )
            required_rollouts_since_last_update = max(1, required_rollouts_since_last_update)

            next_update_rollout = simulated_last_k_update_rollout + required_rollouts_since_last_update
            rollout_steps.append(next_update_rollout)

            simulated_k = min(simulated_k + self.step_K, self.K_max)
            simulated_last_k_update_rollout = next_update_rollout
            simulated_n_k_updates += 1

        return rollout_steps
    
        
    def _on_rollout_end(self):
        # Track rollouts and update K schedule
        self.n_rollouts += 1

        # Include unfinished trajectories from this rollout and order by cumulative reward
        for traj in self.current_rollout_trajectories:
            if len(traj["samples"]) > 0:
                self.completed_rollout_trajectories.append(deepcopy(traj))
        self.ordered_rollout_trajectories = self.order_rollout_trajectories_by_cum_reward()
        self.current_rollout_trajectories = []

        upper_quartile_episode_len_sec = self.get_upper_quartile_rollout_episode_len_sec()

        self.schedule_episode_length(upper_quartile_episode_len_sec)
        self.schedule_K()
        self.initial_state_buffer_easy = []
        self.initial_state_buffer_hard = []
        self._fill_init_state_buffers()
        self._fill_exp_buffer()

        if self.verbose > 0:
            print(f"[ProcedualLearning] Rollout #{self.n_rollouts} ended. "
                  f"Current K: {self.K}, Exp buffer size: {len(self.experience_buffer)}, "
                  f"Completed trajectories in this rollout: {len(self.completed_rollout_trajectories)}")

        self.completed_rollout_trajectories = []
        return True
    
    def sample_spawn(self, network_step_counter, training=True):
        prb = np.random.uniform(0, 1)
        if training:
            if prb <self.P_BUFF:
                assert len(self.experience_buffer) > 0, "Experience buffer is empty, cannot sample spawn"
                spawn = random.choice(self.experience_buffer)
            else:
                # predefined initial states
                prb = np.random.uniform(0, 1)
                if prb < .8:
                    spawn = random.choice(self.initial_state_buffer_easy)
                else:
                    spawn = random.choice(self.initial_state_buffer_hard)
        else:
            spawn = self.predefined_spawns[0]

        return spawn    

    def _init_buffers(self):
        # init initial state buffer with waypoints
        self._fill_init_state_buffers()
        self.experience_buffer = deepcopy(self.initial_state_buffer_easy)

    def _fill_init_state_buffers(self):
        n_spawns_per_waypoint = self.init_buffer_size // len(self.waypoint_xyzs)
        for idx in range(len(self.waypoint_xyzs)):
            flat = self._flat_from_waypoint(idx, self.waypoint_xyzs, self.waypoint_rpys)
            for _ in range(n_spawns_per_waypoint):
                expanded_flat = self.expand_flat(flat)
                spawn = self._spawn_from_flat(expanded_flat)
                self.initial_state_buffer_easy.append(spawn)
                expanded_flat_hard = self.expand_flat(flat, K=self.K + 30)
                spawn_hard = self._spawn_from_flat(expanded_flat_hard)
                self.initial_state_buffer_hard.append(spawn_hard)

    def _fill_exp_buffer(self):
        if len(self.ordered_rollout_trajectories) == 0:
            return

        n_trajectories = len(self.ordered_rollout_trajectories)
        start_idx = int(n_trajectories * 0.1)
        end_idx = int(n_trajectories * 0.3)
        if end_idx <= start_idx:
            end_idx = min(n_trajectories, start_idx + 1)

        selected_trajectories = self.ordered_rollout_trajectories[start_idx:end_idx]
        new_spawns = []

        for trajectory in selected_trajectories:
            samples = trajectory.get("samples", [])
            if len(samples) == 0:
                continue

            n_top_samples = max(1, int(np.ceil(len(samples) * 0.2)))
            top_samples = samples[:n_top_samples]

            for sample in top_samples:
                flat = self._flat_from_trj_sample(sample)
                spawn = self._spawn_from_flat(flat)
                new_spawns.append(spawn)

        if len(new_spawns) == 0:
            return

        self.experience_buffer.extend(new_spawns)
        if len(self.experience_buffer) > self.experience_buffer_size:
            self.experience_buffer = self.experience_buffer[-self.experience_buffer_size:]

        if self.verbose > 0:
            print(
                f"[ProcedualLearning] Exp buffer +{len(new_spawns)} from trajectories "
                f"[{start_idx}:{end_idx}] of {n_trajectories}"
            )


    def order_rollout_trajectories_by_cum_reward(self, descending=True):
        """
        Order trajectories collected in the current rollout by cumulative reward.

        Returns:
            list[dict]: trajectory dicts sorted by `cum_reward`.
        """
        if len(self.completed_rollout_trajectories) == 0:
            return []
        return sorted(
            self.completed_rollout_trajectories,
            key=lambda trajectory: trajectory["cum_reward"],
            reverse=descending,
        )

    def get_upper_quartile_rollout_episode_len_sec(self):
        """Return upper quartile (75th percentile) episode length in seconds for the latest rollout."""
        if len(self.completed_rollout_trajectories) == 0:
            return 0.0

        episode_lens = [
            len(traj.get("samples", [])) * self.DELTA_T
            for traj in self.completed_rollout_trajectories
            if len(traj.get("samples", [])) > 0
        ]
        if len(episode_lens) == 0:
            return 0.0
        return float(np.percentile(episode_lens, 75))
    
    def schedule_K(self):
        """Update K based on number of rollouts with exponentially increasing intervals."""
        if self.K >= self.K_max:
            return self.K

        required_rollouts_since_last_update = int(
            self.K_schedule_start_updates * (self.K_schedule_base ** self.n_K_updates)
        )

        rollouts_since_last_update = self.n_rollouts - self.last_K_update_rollout
        if rollouts_since_last_update < required_rollouts_since_last_update:
            return self.K

        old_K = self.K
        self.K = min(self.K + self.step_K, self.K_max)

        self.last_K_update_rollout = self.n_rollouts
        self.n_K_updates += 1

        if self.verbose > 0:
            next_interval = int(
                self.K_schedule_start_updates * (self.K_schedule_base ** self.n_K_updates)
            )
            print(
                f"[ProcedualLearning] K updated {old_K} -> {self.K} at rollout #{self.n_rollouts} "
                f"(timestep {self.num_timesteps}, next in ~{next_interval} rollouts)"
            )

        return self.K

    
    def schedule_episode_length(self, upper_quartile_episode_len_sec):
        """
        Update episode length based on upper quartile episode length observed in rollout trajectories.

        Episode length is increased only when rollouts are consistently close to the current
        horizon, and only every `EPISODE_LEN_UPDATE_ROLLOUT_INTERVAL` rollouts (cooldown).
        """
        if upper_quartile_episode_len_sec <= 0.0:
            return

        if self.n_rollouts - self.last_episode_len_update < self.EPISODE_LEN_UPDATE_ROLLOUT_INTERVAL:
            return

        current_episode_len = self.training_env.get_attr("episode_len_sec")[0]
        if current_episode_len >= self.MAX_EPISODE_LEN_SEC:
            return

        # Increase only when upper quartile rollout episode length is sufficiently close
        # to the current episode length.
        if upper_quartile_episode_len_sec < self.EPISODE_LEN_UPDATE_CLOSE_RATIO * current_episode_len:
            return

        new_episode_len = min(
            self.MAX_EPISODE_LEN_SEC,
            max(current_episode_len + self.EPISODE_LEN_STEP, upper_quartile_episode_len_sec + self.EPISODE_LEN_STEP),
        )

        if new_episode_len <= current_episode_len:
            return

        self.training_env.env_method("set_episode_len", new_episode_len)
        self.last_episode_len_update = self.n_rollouts

        if self.verbose > 0:
            print(
                f"[ProcedualLearning] Episode length updated from rollout upper quartile: "
                f"{current_episode_len:.3f}s -> {new_episode_len:.3f}s "
                f"(q75={upper_quartile_episode_len_sec:.3f}s)"
            )

    def _spawn_from_flat(self, flat):
        p, v, a, next_waypoints = flat
        rpy = self._get_spawn_rpy(v, a)
        spawn = {"pos": p, "vel": v, "acc": a, "rpy": rpy, "next_waypoints": next_waypoints}
        return spawn

    def _get_spawn_rpy(self, v, a):
        z = (a - np.array([0, 0, -9.81])) / (np.linalg.norm(a - np.array([0, 0, -9.81])))
        x = (v - np.dot(v, z) * z) / (np.linalg.norm(v - np.dot(v, z) * z))
        y = np.cross(z, x)
        R = np.vstack((x, y, z)).T
        return mat2euler(R)
    
    def _flat_from_trj_sample(self, trj_sample):
        p = trj_sample["pos"]
        v = trj_sample["vel"]
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
        v = np.array([1, 0, 0])
        a = local_to_world(a, waypoint_rpys[idx])
        v = local_to_world(v, waypoint_rpys[idx])
        return p, v, a, next_waypoints
        
    def expand_flat(self, flat, K=None):
        if K is None:
            K = self.K
        p, v, a, next_waypoints = flat
        for _ in range(K):
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