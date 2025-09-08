import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import csv

# ------------------------------
# 1. System Initialization
# ------------------------------
time_slot_duration = 1
T = 500  # time slots per episode
episodes = 100
num_SUs = 3

users = ['SU1', 'SU2', 'SU3']

# Energy values (mJ)
E_tx = 0.03
B_max = 0.5
B_init = B_max
P_PU = 1  # PU transmit power in mW
P_SU = 0.03
Q_max = 2  # Buffer size

# Arrival rate
lambda_arrival = 0.9

# Distances from BS (m)
distances = {'PU': 0.5, 'SU1': 0.7, 'SU2': 1.0, 'SU3': 1.2}
distances_SU_to_PU = {'SU1': 1.0, 'SU2': 1.2, 'SU3': 1.4}

# ------------------------------
# 2. Channel Gain Model
# ------------------------------
def compute_large_scale_gain(d, PL_exponent=3.2, PL_scaling=3):
    return 1 / (d ** PL_exponent * 10 ** PL_scaling)

def compute_small_scale_gain():
    g = np.random.randn() + 1j * np.random.randn()
    return np.abs(g)**2

def model_energy_harvest(P_PU, h_gain, P_max=0.04, a=8e5, b=5e-7):
    P_in = P_PU * h_gain
    return P_max / (1 + np.exp(-a * (P_in - b)))

# ------------------------------
# 2.5 PU SINR Check
# ------------------------------
def _PU_SINR_ok(su, h_pu, h_su_to_pu, SINR_thresh=1, noise=1e-9):
    numerator = P_PU * h_pu
    denominator = P_SU * h_su_to_pu[su] + noise
    SINR = numerator / denominator
    return SINR >= SINR_thresh

# ------------------------------
# 3. Main Simulation Loop over Episodes
# ------------------------------
aoi_avg_history = []
energy_harvested_history = []

for ep in range(episodes):
    AoI = {u: 1 for u in users}
    Q = {u: [] for u in users}  # store arrival times
    B = {u: B_init for u in users}
    sum_aoi_episode = 0
    total_energy_harvested = 0

    for t in range(T):
        h_PU = compute_large_scale_gain(distances['PU']) * compute_small_scale_gain()
        h_SU = {u: compute_large_scale_gain(distances[u]) * compute_small_scale_gain() for u in users}
        h_SU_to_PU = {u: compute_large_scale_gain(distances_SU_to_PU[u]) * compute_small_scale_gain() for u in users}

        # Packet arrivals: store generation time
        arrivals = {u: np.random.poisson(lambda_arrival) for u in users}
        for u in users:
            for _ in range(arrivals[u]):
                Q[u].append(t)
                if len(Q[u]) > Q_max:
                    Q[u].pop(0)

        # Exhaustive search for best action
        best_action = None
        best_reward = -np.inf

        for u in users:
            if B[u] < E_tx or len(Q[u]) == 0 or not _PU_SINR_ok(u, h_PU, h_SU_to_PU):
                continue

            # Clone system state
            AoI_temp = AoI.copy()
            Q_temp = {k: Q[k][:] for k in Q}
            B_temp = B.copy()

            # Apply action
            AoI_temp[u] = t - Q_temp[u][0]
            Q_temp[u].pop(0)
            B_temp[u] -= E_tx

            for j in users:
                if j != u:
                    if len(Q_temp[j]) > 0:
                        AoI_temp[j] += 1
                    B_temp[j] = min(B_temp[j] + model_energy_harvest(P_PU, h_SU_to_PU[j]), B_max)

            reward = -sum(AoI_temp.values())
            if reward > best_reward:
                best_reward = reward
                best_action = u

        # Apply best action
        for u in users:
            if u == best_action:
                AoI[u] = t - Q[u][0]
                Q[u].pop(0)
                B[u] -= E_tx
            else:
                if len(Q[u]) > 0:
                    AoI[u] += 1
                eh = model_energy_harvest(P_PU, h_SU_to_PU[u])
                B[u] = min(B[u] + eh, B_max)
                total_energy_harvested += eh

        sum_aoi_episode += sum(AoI[u] for u in users)

    avg_aoi_episode = sum_aoi_episode / T
    aoi_avg_history.append(avg_aoi_episode)
    energy_harvested_history.append(total_energy_harvested)

# ------------------------------
# 4. Plot Average AoI vs Episodes
# ------------------------------
plt.figure(figsize=(8, 6))
smoothed_aoi = gaussian_filter1d(aoi_avg_history, sigma=2)

plt.plot(range(1, episodes + 1), smoothed_aoi, linestyle='-', color='blue')
plt.title('Average Sum of AoI vs Episodes (Exhaustive Search with Accurate AoI)', fontsize=14)
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Average Sum of AoI', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(4, 10)
plt.yticks(np.arange(0, 10.5, 0.4))
plt.tight_layout()
plt.savefig('Avg_AoI_vs_Episodes_ExhaustiveSearch_AoI.png')
plt.show()

# ------------------------------
# 5. Save AoI and Energy to CSV
# ------------------------------
csv_filename = "exhaustive_search_results.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Average_Sum_AoI", "Total_Energy_Harvested"])

    for ep in range(episodes):
        writer.writerow([ep + 1, aoi_avg_history[ep], energy_harvested_history[ep]])

print(f"Results saved to {csv_filename}")
