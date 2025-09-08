import numpy as np
import gym
from gym import spaces

class CRNOMAEnv(gym.Env):
    def __init__(self, time_slots=250):
        super(CRNOMAEnv, self).__init__()

        self.num_SUs = 3
        self.users = [f'SU{i+1}' for i in range(self.num_SUs)]
        self.T = time_slots
        self.time_slot = 0

        self.E_tx = 0.03
        self.B_max = 0.5
        self.B_init = self.B_max
        self.P_PU = 1
        self.P_SU = 0.03
        self.lambda_arrival = 0.8
        self.T_min = 0.3
        self.Q_max = 2

        self.distances = {
            'PU': 0.5,
            'SU1': 0.7,
            'SU2': 1.0,
            'SU3': 1.2
        }
        self.distances_SU_to_PU = {
            'SU1': 1.0,
            'SU2': 1.2,
            'SU3': 1.4
        }

        # ✅ Continuous action vector for DDPG
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_SUs,), dtype=np.float32)

        # ✅ Observation: [AoI, queue length, battery, channel gain] for each SU
        low = np.array([0, 0, 0, 0] * self.num_SUs, dtype=np.float32)
        high = np.array([1, 1, 1, 1] * self.num_SUs, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    def compute_large_scale_gain(self, d, PL_exponent=3.2, PL_scaling=3):
        return 1 / (d ** PL_exponent * 10 ** PL_scaling)

    def compute_small_scale_gain(self):
        g = np.random.randn() + 1j * np.random.randn()
        return np.abs(g) ** 2

    def model_energy_harvest(self, P_PU, h_gain, P_max=0.04, a=8e5, b=5e-7):
        P_in = P_PU * h_gain
        return P_max / (1 + np.exp(-a * (P_in - b)))


    def reset(self):
        self.AoI = {u: 0 for u in self.users}
        self.Q = {u: [] for u in self.users}
        self.B = {u: self.B_init for u in self.users}
        self.time_slot = 0
        self.packet_count = {u: 0 for u in self.users}
        self.total_dropped_packets = 0
        return self._get_obs()

    def _get_obs(self):
        state = []
        for u in self.users:
            h_bs = self.compute_large_scale_gain(self.distances[u]) * self.compute_small_scale_gain()
            state += [
                min(self.AoI[u], 20) / 20,
                len(self.Q[u]) / self.Q_max,
                self.B[u] / self.B_max,
                h_bs / 1e-3
            ]
        return np.array(state, dtype=np.float32)

    def _PU_SINR_ok(self, su, h_pu, h_su_to_pu, noise=1e-9, SINR_thresh=1):
        numerator = self.P_PU * h_pu
        denominator = self.P_SU * h_su_to_pu + noise
        SINR = numerator / denominator
        return SINR >= SINR_thresh

    def get_valid_actions(self):
        return [i for i, u in enumerate(self.users) if len(self.Q[u]) > 0 and self.B[u] >= self.E_tx]

    def step(self, action):
        self.time_slot += 1
        done = self.time_slot >= self.T
        reward = 0

        # ✅ Convert continuous action vector to discrete index
        selected_index = int(np.argmax(action))
        selected_index = np.clip(selected_index, 0, self.num_SUs - 1)
        selected_su = self.users[selected_index]

        h_pu = self.compute_large_scale_gain(self.distances['PU']) * self.compute_small_scale_gain()
        h_su_to_pu = {
            u: self.compute_large_scale_gain(self.distances_SU_to_PU[u]) * self.compute_small_scale_gain()
            for u in self.users
        }

        # ✅ Packet arrival and queue update
        dropped_packets = 0
        for u in self.users:
            arrivals = np.random.poisson(self.lambda_arrival)
            for _ in range(arrivals):
                self.Q[u].append(self.time_slot)
                if len(self.Q[u]) > self.Q_max:
                    self.Q[u].pop(0)
                    dropped_packets += 1
        self.total_dropped_packets += dropped_packets

        valid_actions = self.get_valid_actions()
        if len(valid_actions) == 0:
            reward -= 0.5
        if selected_index not in valid_actions:
            reward -= 2

        aoi_before = {u: self.AoI[u] for u in self.users}

        # ✅ Update states and calculate reward
        for u in self.users:
            if u == selected_su and selected_index in valid_actions and self._PU_SINR_ok(u, h_pu, h_su_to_pu[u]):
                self.packet_count[u] += 1
                gen_time = self.Q[u].pop(0)
                self.AoI[u] = self.time_slot - gen_time
                self.B[u] -= self.E_tx
                reward += 2
            else:
                if len(self.Q[u]) > 0:
                    self.AoI[u] += 1
                harvested = self.model_energy_harvest(self.P_PU, h_su_to_pu[u])
                self.B[u] = min(self.B[u] + harvested, self.B_max)

        for u in self.users:
            self.AoI[u] = min(self.AoI[u], 20)

        # ✅ AoI-based shaping
        if self.time_slot > 10:
            aoi_reduction = sum([
                max(0, min(aoi_before[u], 20) - min(self.AoI[u], 20))
                for u in self.users
            ])
            aoi_increase = sum([
                max(0, np.log1p(self.AoI[u] - aoi_before[u]))
                for u in self.users if self.AoI[u] > aoi_before[u]
            ])
            aoi_steady = sum([
                1 for u in self.users if self.AoI[u] == aoi_before[u]
            ])

            reward += 5 * np.log1p(aoi_reduction)
            reward -= 2.5 * np.log1p(aoi_increase)
            reward += 0.5 * aoi_steady

            aoi_values = np.array([self.AoI[u] for u in self.users])
            aoi_std = np.std(aoi_values)
            reward -= 2 * aoi_std

        max_aoi_user = max(self.users, key=lambda u: self.AoI[u])
        if selected_su == max_aoi_user and selected_index in valid_actions:
            reward += 2.5

        reward -= 2 * dropped_packets

        if self.time_slot > self.T * 0.4:
            for u in self.users:
                avg_tp = self.packet_count[u] / (self.time_slot + 1)
                if avg_tp < self.T_min:
                    reward -= 0.2

        reward = np.clip(reward, -10, 10)
        obs = self._get_obs()
        return obs, reward, done, {"dropped_packets": dropped_packets}

    def render(self, mode='human'):
        print(f"Time Slot {self.time_slot}")
        for u in self.users:
            print(f"{u}: AoI={self.AoI[u]}, Q={len(self.Q[u])}, B={self.B[u]:.3f}")
        print(f"Dropped packets so far: {self.total_dropped_packets}")
