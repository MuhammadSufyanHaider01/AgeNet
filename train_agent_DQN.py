from env_test import CRNOMAEnv
from DQN_Agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import math


# ✅ Initialize Environment
env = CRNOMAEnv(time_slots=100)

# ✅ Environment Details
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# ✅ Initialize DQN Agent
agent = DQNAgent(state_size, action_size)

# ✅ Hyperparameters
episodes = 100
B = 1e6  # Bandwidth (1 MHz)
N0 = 1e-9  # Noise power

# ✅ Tracking
reward_history = []
drop_history = []
aoi_avg_history = []
throughput_avg_history = []

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    total_drops = 0
    cumulative_aoi_sum = 0
    cumulative_throughput = 0  # ✅ Throughput tracker

    print(f"\n=== Episode {e + 1} ===")

    for t in range(env.T):
        valid_actions = env.get_valid_actions()

        if len(valid_actions) == 0:
            action = np.random.choice(range(action_size))
        else:
            action = agent.act(state, valid_actions)

        next_state, reward, done, info = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state

        total_reward += reward
        total_drops += info['dropped_packets']

        # ✅ AoI sum for this time slot
        sum_aoi = sum([env.AoI[u] for u in env.users])
        cumulative_aoi_sum += sum_aoi

        # ✅ Throughput calculation for this time slot
        selected_su = env.users[action]

        h_pu = env.compute_large_scale_gain(env.distances['PU']) * env.compute_small_scale_gain()
        h_su_to_pu = env.compute_large_scale_gain(env.distances_SU_to_PU[selected_su]) * env.compute_small_scale_gain()

        SINR = (env.P_PU * h_pu) / (env.P_SU * h_su_to_pu + N0)

        if (len(env.Q[selected_su]) > 0) and (env.B[selected_su] >= env.E_tx) and env._PU_SINR_ok(
                selected_su, h_pu, h_su_to_pu):
            throughput = B * math.log2(1 + SINR)  # in bits per second
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
                       )
        print(status_line)

        if done:
            agent.update_target_model()
            break

    avg_sum_aoi = cumulative_aoi_sum / env.T
    avg_throughput = cumulative_throughput / env.T  # ✅ Average throughput for this episode

    aoi_avg_history.append(avg_sum_aoi)
    throughput_avg_history.append(avg_throughput)
    reward_history.append(total_reward)
    drop_history.append(total_drops)

    print(f"=== Episode {e + 1} Summary ===")
    print(f"Total Reward: {total_reward:.2f} | "
          f"Total Dropped Packets: {total_drops} | "
          f"Avg Sum of AoI: {avg_sum_aoi:.2f} | "
          f"Avg Throughput: {avg_throughput/1e6:.3f} Mbps | "
          f"Epsilon: {agent.epsilon:.3f}")
    print("=" * 90)

# ✅ Save model
agent.model.save("dqn_cr_noma_model.h5")
print("Model saved as dqn_cr_noma_model.h5")


# ✅ Plot Average Sum of AoI vs Episodes
plt.figure(figsize=(8, 6))
plt.plot(range(1, episodes + 1), aoi_avg_history, marker='o', linestyle='-', color='b')
plt.title('Average Sum of AoI vs Episodes (DQN)')
plt.xlabel('Episodes')
plt.ylabel('Average Sum of AoI')
plt.grid(True)
plt.tight_layout()
plt.savefig('Avg_AoI_vs_Episodes_DQN.png')
plt.show()

# ✅ Plot Average Throughput vs Episodes
plt.figure(figsize=(8, 6))
plt.plot(range(1, episodes + 1), [t / 1e6 for t in throughput_avg_history], marker='s', linestyle='-', color='green')
plt.title('Average Throughput vs Episodes (DQN)')
plt.xlabel('Episodes')
plt.ylabel('Throughput (Mbps)')
plt.grid(True)
plt.tight_layout()
plt.savefig('Avg_Throughput_vs_Episodes_DQN.png')
plt.show()
