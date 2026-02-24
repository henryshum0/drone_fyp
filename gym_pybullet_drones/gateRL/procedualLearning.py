import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch
from transforms3d.euler import euler2quat, mat2euler
from transforms3d.quaternions import rotate_vector, qconjugate

from gym_pybullet_drones.gateRL.waypoints import waypoints1
from gym_pybullet_drones.gateRL.gateRLEnv import GateRLEnv
class ProcedualLearning(BaseCallback):
    def __init__(self, 
                 waypoints, 
                 buffer_size, 
                 n_moderate=100,
                 dt=0.01,
                 K=10,
                 low=0.,
                 high=0.1,

                 verbose=0):
        
        super(ProcedualLearning, self).__init__(verbose)
        self.buffer_size = buffer_size
        self.DELTA_T = dt
        self.K = K
        self.low = low
        self.high = high
        self.n_moderate = n_moderate
        self.waypoint_xyzs = waypoints["pos"]
        self.waypoint_rpys = waypoints["rpy"]
        self.waypoint_quats = np.array([euler2quat(*rpy) for rpy in self.waypoint_rpys])
        self.experience_buffer = []
        self.adaptive_buffer = []
        self._init_experience_buffer()
        self.trj_buffer = []
        
        
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
            accelerations = tuple(info.get("acceleration", None) for info in infos)
            assert all([nw is not None for nw in next_waypoints]), "next_waypoints should be provided in info"
            assert new_obs.shape[0] == values_np.shape[0] == len(next_waypoints) == len(accelerations), "Batch size of new_obs, values, next_waypoints and accelerations should be the same"
            for idx, (single_obs, single_value, single_next_waypoints, acc) in enumerate(
                zip(new_obs, values_np, next_waypoints, accelerations)
            ):
                self.trj_buffer.append(
                    {
                        "obs": single_obs,
                        "value": single_value,
                        "next_waypoints": tuple(single_next_waypoints),
                        "acc": acc,
                        "id": idx,
                    }
                )
        return True
    
    def _on_training_start(self):
        vec_env = self.training_env  # provided by BaseCallback
        vec_env.env_method("update_experience_buffer", self.experience_buffer)
        vec_env.env_method("update_adaptive_buffer", self.experience_buffer)
        self.trj_buffer = []
        self.adaptive_buffer = []
        return True
        
    def _on_rollout_end(self):
        # sort and get the moderate value rollouts        
        moderates = self._get_moderate_trj()
        for moderate in moderates:
            spawn = expand_using_trj_sample(moderate, K=self.K, low=self.low, high=self.high, dt=self.DELTA_T)
            self._experience_buffer_add(spawn)
            self._adaptive_buffer_add(spawn)

        # update env experience buffer for all sub-envs
        assert self.adaptive_buffer != [], "Adaptive buffer should not be empty after rollout end"
        vec_env = self.training_env
        vec_env.env_method("update_experience_buffer", self.experience_buffer)
        vec_env.env_method("update_adaptive_buffer", self.adaptive_buffer)
        self.trj_buffer = []
        self.adaptive_buffer = []
        return True

    def _init_experience_buffer(self):
        # define the goal state to be p, q, v
        for i in range(len(self.waypoint_xyzs)):
            spawn = expand_using_waypoints(i, self.waypoint_xyzs, self.waypoint_rpys, K=self.K, low=self.low, high=self.high, dt=self.DELTA_T)
            self._experience_buffer_add(spawn)
            self._adaptive_buffer_add(spawn)

    def _experience_buffer_add(self, spawn):
        # add spawns to experience buffer
        # spawn to be (p, v, a, rpy, next_waypoints)
        self.experience_buffer.append(spawn)
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
            
    def _adaptive_buffer_add(self, spawn):
        self.adaptive_buffer.append(spawn)
        if len(self.adaptive_buffer) > self.buffer_size:
            self.adaptive_buffer.pop(0)
            
    def _get_moderate_trj(self):
        sorted_list = sorted(self.trj_buffer, key=lambda x: x["value"])
        top = int(len(sorted_list) * 0.1)
        if top + self.n_moderate > len(sorted_list):
            moderates = sorted_list[top:]
        else:
            moderates = sorted_list[top:top+self.n_moderate]
        assert len(moderates) != 0, "Moderate trajectories should not be empty"
        return moderates
    
    def schedule_K(self,):
        # scheduler K based on the mean rewards of evaluations and the min number of steps between each updates of K
        pass

def expand_using_trj_sample(trj_sample, K=5, low=0., high=0.2, dt=0.01):
    p = trj_sample["obs"][0, 14:17]
    v_b = trj_sample["obs"][0, 21:24]

    # get acceleration
    quat = trj_sample["obs"][0, 17:21]
    v = local_to_world(v_b, quat)
    a = trj_sample["acc"]

    # expand flat state to get init states
    flat = expand_flat(p, v, a, K=K, low=low, high=high, dt=dt)
    rpy = get_spawn_rpy(*flat)
    spawn = {
        "pos": flat[0],
        "vel": flat[1],
        "acc": flat[2],
        "rpy": rpy,
        "next_waypoints": trj_sample["next_waypoints"],
    }
    return spawn

def expand_using_waypoints(idx, waypoint_xyzs, waypoint_rpys, K=5, low=0., high=0.2, dt=0.01):
    if idx <= len(waypoint_xyzs) - 2:
        next_waypoints = (idx, idx+1)
    else:    
        next_waypoints = (idx, -1)
    a = np.array([0, 0, 0])
    v = np.array([5, 0, 0])
    a = local_to_world(a, waypoint_rpys[idx])
    v = local_to_world(v, waypoint_rpys[idx])
    p, v, a = expand_flat(waypoint_xyzs[idx], v, a, K, low, high, dt)
    rpy = get_spawn_rpy(v, a)
    spawn = {"pos": p, "vel": v, "acc": a, "rpy": rpy, "next_waypoints": next_waypoints}
    return spawn
        
def get_spawn_rpy(v, a):
    z = (a - np.array([0, 0, -9.81])) / np.linalg.norm(a - np.array([0, 0, -9.81]))
    x = (v - np.dot(v, z) * z) / np.linalg.norm(v - np.dot(v, z) * z)
    y = np.cross(z, x)
    R = np.vstack((x, y, z)).T
    return mat2euler(R)
        
def expand_flat(p, v, a, K, low, high, dt):
    for _ in range(K):
        j = np.random.uniform(low=low, high=high, size=(3,))
        a = a - j * dt
        v = v - a * dt - j * dt**2 / 2
        p = p - v * dt - a * dt**2 / 2 - j * dt**3 / 6
    return p, v, a

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