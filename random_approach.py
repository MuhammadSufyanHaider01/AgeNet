import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ------------------------------
# 1. System Initialization
# ------------------------------
time_slot_duration = 1
T = 1000  # time slots per episode
episodes = 100
num_SUs = 3
Qmax = 2

users = ['SU1', 'SU2', 'SU3']

# Energy values (mJ)
E_tx = 0.03
B_max = 0.5
B_init = B_max
eta = 0.7
P_PU = 1  # PU transmit power in mW

# Arrival rate
lambda_arrival = 0.3

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

def model_energy_harvest( P_PU, h_gain, E_max=0.04, k=1000):
        P_recv = P_PU * h_gain
        return E_max * (1 - np.exp(-k * P_recv))


# ------------------------------
# 3. Main Simulation Loop over Episodes
# ------------------------------
aoi_avg_history = []

for ep in range(episodes):
    # Reset environment for each episode
    AoI = {u: 1 for u in users}
    Q = {u: [] for u in users}  # Store generation time of each packet
    B = {u: B_init for u in users}

    sum_aoi_episode = 0

    for t in range(T):
        h_PU = compute_large_scale_gain(distances['PU']) * compute_small_scale_gain()
        h_SU = {u: compute_large_scale_gain(distances[u]) * compute_small_scale_gain() for u in users}
        h_SU_to_PU = {u: compute_large_scale_gain(distances_SU_to_PU[u]) * compute_small_scale_gain() for u in users}

        # Packet arrivals: store generation time (t)
        arrivals = {u: np.random.poisson(lambda_arrival) for u in users}
        for u in users:
            for _ in range(arrivals[u]):
                Q[u].append(t)
                if len(Q[u]) > Qmax:
                    Q[u].pop(0)  # Drop oldest packet if buffer full

        # Randomly select one SU to transmit
        selected_SU = np.random.choice(users)

        # Transmission & harvesting logic
        for u in users:
            if u == selected_SU and B[u] >= E_tx and len(Q[u]) > 0:
                gen_time = Q[u].pop(0)
                AoI[u] = t - gen_time   # Proper AoI update (t - g)
                B[u] -= E_tx
            else:
                if len(Q[u])>0: # If there are packets in the queue
                    AoI[u] += 1
                harvested = model_energy_harvest(P_PU, h_SU_to_PU[u])
                B[u] = min(B[u] + harvested, B_max)

        sum_aoi_episode += sum(AoI[u] for u in users)

    avg_aoi_episode = sum_aoi_episode / T
    aoi_avg_history.append(avg_aoi_episode)

# ------------------------------
# 4. Plot Average AoI vs Episodes
# ------------------------------
plt.figure(figsize=(8, 6))
smoothed_aoi = gaussian_filter1d(aoi_avg_history, sigma=2)

plt.plot(range(1, episodes + 1), smoothed_aoi, linestyle='-', color='blue')
plt.title('Average Sum of AoI vs Episodes (Random Selection)', fontsize=14)
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Average Sum of AoI', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Custom Y-axis scale (optional)
plt.ylim(6, 10)
plt.yticks(np.arange(6, 12.5, 0.5))

plt.tight_layout()
plt.savefig('Avg_AoI_vs_Episodes_Random.png')
# np.save("random_aoi_curve.npy", aoi_avg_history)
plt.show()
