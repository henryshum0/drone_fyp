from copy import deepcopy
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ProcedualLearning(BaseCallback):
    def __init__(self, 
                 verbose=0,
                 dt=0.01,
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
                 hard_template_min_pct=50.0,
                 hard_template_max_pct=50.0,
                 hard_template_curve=1.0,
                 ):
        
        super(ProcedualLearning, self).__init__(verbose)
        

        # schedules K
        self.K_init = K_init
        self.step_K = step_K
        self.K = K_init
        self.K_max = K_max
        self.K_schedule_base = K_schedule_base
        self.K_schedule_start_updates = K_schedule_start_updates

        # Track K scheduling
        self.n_K_updates = 0
        self.last_K_update_rollout = 0
        self.n_rollouts = 0

        # Track episode length scheduling
        self.DT = dt
        self.MAX_EPISODE_LEN_SEC = max_episode_len_sec
        self.INITIAL_EPISODE_LEN_SEC = initial_episode_len
        self.EPISODE_LEN_STEP = episode_len_step
        self.EPISODE_LEN_UPDATE_ROLLOUT_INTERVAL = episode_len_update_rollout_interval
        self.EPISODE_LEN_UPDATE_CLOSE_RATIO = float(episode_len_update_close_ratio)
        self.last_episode_len_update = 0
        self.ep_len_buffer = []
        self.all_ep_len = []

        # Track hard template scheduling
        self.HARD_TEMPLATE_MIN_PCT = float(hard_template_min_pct)
        self.HARD_TEMPLATE_MAX_PCT = float(hard_template_max_pct)
        self.HARD_TEMPLATE_CURVE = float(hard_template_curve)
        
    def _on_step(self):
        dones = self.locals.get("dones", None)

        if len(self.ep_len_buffer) == 0:
            self.ep_len_buffer = [0 for _ in range(len(dones))]
        for env_idx, done in enumerate(dones):
            self.ep_len_buffer[env_idx] += self.DT
            if done:
                self.all_ep_len.append(deepcopy(self.ep_len_buffer[env_idx]))
                self.ep_len_buffer[env_idx] = 0
        return True
    
    def _on_training_start(self):
        # set env K and ep len
        self.training_env.env_method("set_K", self.K_init)

        k_update_rollout_steps = self.get_K_update_rollout_steps()
        print(f"[ProcedualLearning] K will be updated at rollout(s): {k_update_rollout_steps}")
        
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

        upper_quartile_episode_len_sec = self.get_upper_quartile_rollout_episode_len_sec()
        self.schedule_K()
        if self.verbose > 0:
            print(f"[ProcedualLearning] Rollout #{self.n_rollouts} ended. "
                  f"Current K: {self.K}"
                  f"upper quartile episode length: {upper_quartile_episode_len_sec:.3f}s")
        self.all_ep_len = []
        return True


    def get_upper_quartile_rollout_episode_len_sec(self):
        """Return upper quartile (75th percentile) episode length in seconds for the latest rollout."""
        if len(self.all_ep_len) == 0:
            return 0.0
        episode_lens = np.array(self.all_ep_len)
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

        self.training_env.env_method("set_K", self.K)
        if self.verbose > 0:
            next_interval = int(
                self.K_schedule_start_updates * (self.K_schedule_base ** self.n_K_updates)
            )
            print(
                f"[ProcedualLearning] K updated {old_K} -> {self.K} at rollout #{self.n_rollouts} "
                f"(timestep {self.num_timesteps}, next in ~{next_interval} rollouts)"
            )