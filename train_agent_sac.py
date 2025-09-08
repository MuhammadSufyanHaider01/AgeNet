from environment import CRNOMAEnv
from SAC_Agent import SACAgent
import numpy as np
import matplotlib.pyplot as plt
import math


# ✅ Initialize Environment
env = CRNOMAEnv(time_slots=500)

# ✅ Environment Details
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# ✅ Initialize SAC Agent
agent = SACAgent(state_size, action_size)

# ✅ Hyperparameters
episodes = 100
B = 1e6   # Bandwidth (Hz)
N0 = 1e-9  # Noise power (Watts)

# ✅ Tracking Metrics
reward_history = []
drop_history = []
aoi_avg_history = []
throughput_avg_history = []

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    total_drops = 0
    cumulative_aoi_sum = 0
    cumulative_throughput = 0

    print(f"\n=== Episode {e + 1} ===")

    for t in range(env.T):
        valid_actions = env.get_valid_actions()

        if len(valid_actions) == 0:
            action = np.random.choice(range(action_size))
        else:
            action, _ = agent.get_action(state, valid_actions)

        next_state, reward, done, info = env.step(action)

        agent.store_transition(state, action, reward, next_state, done)
        agent.update()

        state = next_state
        total_reward += reward
        total_drops += info['dropped_packets']

        # ✅ AoI sum for this time slot
        sum_aoi = sum([env.AoI[u] for u in env.users])
        cumulative_aoi_sum += sum_aoi

        # ✅ Throughput calculation
        selected_su = env.users[action]

        h_pu = env.compute_large_scale_gain(env.distances['PU']) * env.compute_small_scale_gain()
        h_su_to_pu = env.compute_large_scale_gain(env.distances_SU_to_PU[selected_su]) * env.compute_small_scale_gain()

        SINR = (env.P_PU * h_pu) / (env.P_SU * h_su_to_pu + N0)

        if (len(env.Q[selected_su]) > 0) and (env.B[selected_su] >= env.E_tx) and env._PU_SINR_ok(
                selected_su, h_pu, h_su_to_pu):
            throughput = B * math.log2(1 + SINR)  # in bits/sec
        else:
            throughput = 0

        cumulative_throughput += throughput

        # ✅ Logging
        aoi_list = [env.AoI[u] for u in env.users]
        battery_list = [round(env.B[u], 3) for u in env.users]
        queue_list = [len(env.Q[u]) for u in env.users]

        status_line = (f"[Ep {e + 1} | TS {t + 1}] "
                       f"AoI={aoi_list} | "
                       f"Battery={battery_list} | "
                       f"Queue={queue_list} | "
                       f"Action: {env.users[action]} | "
                       f"Rwd: {reward:.2f} | "
                       f"Drop: {info['dropped_packets']} | "
                       f"Thpt: {throughput/1e6:.3f} Mbps")
        print(status_line)

        if done:
            break

    avg_sum_aoi = cumulative_aoi_sum / env.T
    avg_throughput = cumulative_throughput / env.T

    aoi_avg_history.append(avg_sum_aoi)
    throughput_avg_history.append(avg_throughput)
    reward_history.append(total_reward)
    drop_history.append(total_drops)

    print(f"=== Episode {e + 1} Summary ===")
    print(f"Total Reward: {total_reward:.2f} | "
          f"Total Dropped Packets: {total_drops} | "
          f"Avg Sum of AoI: {avg_sum_aoi:.2f} | "
          f"Avg Throughput: {avg_throughput/1e6:.3f} Mbps")
    print("=" * 90)


# ✅ Save Actor Model
agent.actor.save("sac_cr_noma_actor.h5")
print("Actor model saved as sac_cr_noma_actor.h5")


# ✅ Plot Average Sum of AoI vs Episodes
plt.figure(figsize=(8, 6))
plt.plot(range(1, episodes + 1), aoi_avg_history, marker='o', linestyle='-', color='blue')
plt.title('Average Sum of AoI vs Episodes (SAC)')
plt.xlabel('Episodes')
plt.ylabel('Average Sum of AoI')
plt.grid(True)
y_min = np.floor(min(aoi_avg_history) * 2) / 2
y_max = np.ceil(max(aoi_avg_history) * 2) / 2
plt.yticks(np.arange(y_min, y_max + 0.5, 0.5))
plt.tight_layout()
plt.savefig('Avg_AoI_vs_Episodes_SAC.png')
# np.save("sac_aoi_curve.npy", aoi_avg_history)
plt.show()


