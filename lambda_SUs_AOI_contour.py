import numpy as np
import matplotlib.pyplot as plt
import csv
from tensorflow.keras.models import load_model # type: ignore
from environment_SU_var import CRNOMAEnv  # Make sure this points to your environment file

# -----------------------------
# CONFIGURATION
# -----------------------------
lambda_vals = np.round(np.arange(0.2, 0.91, 0.05), 2)
SU_vals = np.array([2, 3, 4, 5])
T = 1000  # Time steps per episode

CSV_OUTPUT = "aoi_contour_data.csv"

model_paths = {
    2: "sac_cr_noma_actor_SU2_Q2.h5",
    3: "sac_cr_noma_actor_SU3_Q2.h5",
    4: "sac_cr_noma_actor_SU4_Q2.h5",
    5: "sac_cr_noma_actor_SU5_Q2.h5"
}

aoi_matrix = np.zeros((len(SU_vals), len(lambda_vals)))

# -----------------------------
# CSV HEADER
# -----------------------------
with open(CSV_OUTPUT, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['SUs', 'Lambda', 'Avg_Sum_AoI'])

# -----------------------------
# MAIN LOOP
# -----------------------------
for i, su_count in enumerate(SU_vals):
    print(f"\n🧠 Loading model for {su_count} SUs...")
    model = load_model(model_paths[su_count])

    for j, lamb in enumerate(lambda_vals):
        print(f"\n🔁 Running for λ = {lamb}, SUs = {su_count}")

        env = CRNOMAEnv(time_slots=T, num_SUs=su_count, Q_max=2)  # Assuming Q_max is 2 for all
        env.num_SUs = su_count
        env.users = [f'SU{k+1}' for k in range(su_count)]
        env.lambda_arrival = lamb
        env.reset()

        aoi_acc = {u: 0 for u in env.users}
        state = env._get_obs()

        for t in range(T):
            probs = model.predict(state.reshape(1, -1), verbose=0)[0]
            action = np.argmax(probs)
            next_state, reward, done, info = env.step(action)

            # Logging per step
            aoi_list = [env.AoI[u] for u in env.users]
            battery_list = [round(env.B[u], 3) for u in env.users]
            queue_list = [len(env.Q[u]) for u in env.users]

            print(f"[λ={lamb}, SUs={su_count} | TS={t+1}] "
                  f"AoI={aoi_list} | Battery={battery_list} | "
                  f"Queue={queue_list} | Action={env.users[action]} | "
                  f"Drop={info['dropped_packets']}")

            for u in env.users:
                aoi_acc[u] += env.AoI[u]

            state = next_state

        avg_sum_aoi = sum(aoi_acc.values()) / T
        aoi_matrix[i, j] = avg_sum_aoi

        print(f"✅ DONE: λ = {lamb}, SUs = {su_count}, Avg Sum AoI = {avg_sum_aoi:.2f}")

        # CSV logging
        with open(CSV_OUTPUT, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([su_count, lamb, round(avg_sum_aoi, 2)])

# Contour Plot
X, Y = np.meshgrid(lambda_vals, SU_vals)

plt.figure(figsize=(9, 6))

filled = plt.contourf(X, Y, aoi_matrix, levels=20, cmap='viridis')
contours = plt.contour(X, Y, aoi_matrix, colors='black', linewidths=0.8)
plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

cbar = plt.colorbar(filled)
cbar.set_label('Age of Information', fontsize=12)

plt.title('Contour Plot of AOI vs λ and Number of SUs', fontsize=15, fontweight='bold')
plt.xlabel('Arrival Rate λ', fontsize=12)
plt.ylabel('Number of SUs', fontsize=12)

plt.xticks(lambda_vals, rotation=45)
plt.yticks(SU_vals)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("contour_aoi_lambda_sus.png")
plt.show()
