import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch
from transforms3d.euler import euler2quat, mat2euler
from transforms3d.quaternions import rotate_vector, qconjugate

from gym_pybullet_drones.gateRL.waypoints import waypoints1

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
        self.steps = 0
        self.DELTA_T = dt
        self.K = K
        self.low = low
        self.high = high
        self.n_moderate = n_moderate
        self.waypoints = waypoints
        self.experience_buffer = self._init_experience_buffer()
        self.trj_buffer = []
        
        
    def _on_rollout_start(self):
        self.trj_buffer = []
        return True
        
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
                obs_tensor, _ = self.model.policy.obs_to_tensor(new_obs, self.model.device)
                values = self.model.policy.predict_values(obs_tensor)
            values_np = values.detach().cpu().numpy().squeeze()

            infos = self.locals.get("infos", [])
            next_waypoints = [info.get("next_waypoints", None) for info in infos]
            assert all([nw is not None for nw in next_waypoints]), "next_waypoints should be provided in info"
            assert len(new_obs) == len(values_np) == len(next_waypoints), "Length of new_obs, values, and next_waypoints should be the same"
            for single_obs, single_value, single_next_waypoints in zip(new_obs, values_np, next_waypoints):
                self.trj_buffer.append(
                    {"obs": single_obs, 
                     "value": single_value, 
                     "next_waypoints": single_next_waypoints
                     }
                )
        return True
    
    def _on_rollout_end(self):
        # Placeholder for computing moderate difficulty spawns.
        return
    
    def get_flat(self, obs, values):
        pass
        
    def _init_experience_buffer(self):
        # define the goal state to be p, q, v
        buffer = []
        waypoints_xyz = self.waypoints[0]
        waypoints_rpy = self.waypoints[1]
        
        for i, (xyz, rpy) in enumerate(zip(waypoints_xyz, waypoints_rpy)):
            if i < len(waypoints_xyz) - 2:
                next_waypoints = [i+1, i+2]
            elif i == len(waypoints_xyz) - 2:
                next_waypoints = [i+1, -1]
            else:    
                next_waypoints = [-1, -1]
            a = np.array([0, 0, 0])
            v = np.array([5, 0, 0])
            a = local_to_world(a, rpy)
            v = local_to_world(v, rpy)
            p, v, a = self._expand_flat(xyz, v, a)
            rpy = self._get_spawn_rpy(p, v, a)
            spawn = {"pos": p, "vel": v, "acc": a, "rpy": rpy, "next_waypoints": next_waypoints}
            buffer.append(spawn)
        return buffer
            
    def _flat_from_drone_state(self, pos, vel, acc):
        # drone state is p, v, a
        pass
    
    def _expand_flat(self, p, v, a):
        for _ in range(self.K):
            j = np.random.uniform(low=self.low, high=self.high, size=(3,))
            a = a - j * self.DELTA_T
            v = v - a * self.DELTA_T - j * self.DELTA_T**2 / 2
            p = p - v * self.DELTA_T - a * self.DELTA_T**2 / 2 - j * self.DELTA_T**3 / 6
        return p, v, a
    
    def _get_spawn_rpy(self, p, v, a):
        z = (a - np.array([0, 0, -9.81])) / np.linalg.norm(a - np.array([0, 0, -9.81]))
        x = (v - np.dot(v, z) * z) / np.linalg.norm(v - np.dot(v, z) * z)
        y = np.cross(z, x)
        R = np.vstack((x, y, z)).T
        return mat2euler(R)
    
    def _buffer_add(self, spawn):
        # spawn to be (p, v, a, rpy, next_waypoints)
        self.experience_buffer.append(spawn)
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
            


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
    print(callback.experience_buffer)