import numpy as np
import matplotlib.pyplot as plt
import math
from plots import CRNOMAEnv
from DDPG_Agent import DDPGAgent

# ✅ Initialize Environment
env = CRNOMAEnv(time_slots=500)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = 1.0  # Because your action_space is Box(0,1,...)

agent = DDPGAgent(state_dim, action_dim, action_bound)

# ✅ Training Hyperparameters
episodes = 100
B = 1e6     # Bandwidth = 1 MHz
N0 = 1e-9   # Noise power

# ✅ Tracking Metrics
reward_history = []
drop_history = []
aoi_avg_history = []
throughput_avg_history = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    total_drops = 0
    cumulative_aoi_sum = 0
    cumulative_throughput = 0

    print(f"\n=== Episode {ep + 1} ===")

    for t in range(env.T):
        action = agent.get_action(state)

        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, float(done))

        state = next_state
        total_reward += reward
        total_drops += info["dropped_packets"]

        # ✅ AoI and Throughput calculations
        sum_aoi = sum([env.AoI[u] for u in env.users])
        cumulative_aoi_sum += sum_aoi

        selected_su = env.users[int(np.argmax(action))]

        h_pu = env.compute_large_scale_gain(env.distances['PU']) * env.compute_small_scale_gain()
        h_su_to_pu = env.compute_large_scale_gain(env.distances_SU_to_PU[selected_su]) * env.compute_small_scale_gain()

        SINR = (env.P_PU * h_pu) / (env.P_SU * h_su_to_pu + N0)

        if (len(env.Q[selected_su]) > 0 and
            env.B[selected_su] >= env.E_tx and
            env._PU_SINR_ok(selected_su, h_pu, h_su_to_pu)):
            throughput = B * math.log2(1 + SINR)
        else:
            throughput = 0

        cumulative_throughput += throughput

        # ✅ Logging
        aoi_list = [env.AoI[u] for u in env.users]
        battery_list = [round(env.B[u], 3) for u in env.users]
        queue_list = [len(env.Q[u]) for u in env.users]

        print(f"[Ep {ep + 1} | TS {t + 1}] "
              f"AoI={aoi_list} | "
              f"Battery={battery_list} | "
              f"Queue={queue_list} | "
              f"Action: {env.users[int(np.argmax(action))]} | "
              f"Rwd: {reward:.2f} | Drop: {info['dropped_packets']}")

        # ✅ Train at each step
        agent.train()

        if done:
            break

    avg_sum_aoi = cumulative_aoi_sum / env.T
    avg_throughput = cumulative_throughput / env.T

    reward_history.append(total_reward)
    drop_history.append(total_drops)
    aoi_avg_history.append(avg_sum_aoi)
    throughput_avg_history.append(avg_throughput)

    print(f"=== Episode {ep + 1} Summary ===")
    print(f"Total Reward: {total_reward:.2f} | "
          f"Total Dropped Packets: {total_drops} | "
          f"Avg Sum of AoI: {avg_sum_aoi:.2f} | "
          f"Avg Throughput: {avg_throughput / 1e6:.3f} Mbps")
    print("=" * 90)

# ✅ Save Models
agent.actor.save("ddpg_cr_noma_actor.h5")
agent.critic.save("ddpg_cr_noma_critic.h5")
print("Models saved.")

# ✅ Plot AoI
plt.figure(figsize=(8, 6))
plt.plot(range(1, episodes + 1), aoi_avg_history, marker='o', linestyle='-', color='b')
plt.title('Average Sum of AoI vs Episodes (DDPG)')
plt.xlabel('Episodes')
plt.ylabel('Average Sum of AoI')
# plt.yticks(5,41,1.5)
plt.grid(True)
plt.tight_layout()
plt.savefig('Avg_AoI_vs_Episodes_DDPG.png')
np.save("ddpg_aoi_curve.npy", aoi_avg_history)
plt.show()
