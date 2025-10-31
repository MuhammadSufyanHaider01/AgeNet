import numpy as np
import matplotlib.pyplot as plt
import csv
from environment_SU_var import CRNOMAEnv
from tensorflow.keras.models import load_model # type: ignore
import math

su_values = [2, 3, 4, 5]
qmax_values = [2, 4]
time_slots = 1000

aoi_results = {q: [] for q in qmax_values}
eh_results = {q: [] for q in qmax_values}
csv_rows = [["SU", "Qmax", "Avg_AoI", "Total_EH"]]

for Q_max in qmax_values:
    print(f"\n### Evaluating for Q_max = {Q_max} ###\n")
    for num_su in su_values:
        env = CRNOMAEnv(time_slots=time_slots, num_SUs=num_su, Q_max=Q_max)
        actor = load_model(f"sac_cr_noma_actor_SU{num_su}_Q{Q_max}.h5")

        total_aoi, total_eh = 0, 0
        state = env.reset()

        for t in range(time_slots):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                action = np.random.choice(env.action_space.n)
            else:
                probs = actor.predict(state.reshape(1, -1), verbose=0)[0]
                mask = np.zeros_like(probs)
                mask[valid_actions] = 1
                masked_probs = probs * mask
                masked_probs /= masked_probs.sum()
                action = np.random.choice(env.action_space.n, p=masked_probs)

            next_state, _, done, info = env.step(action)

            total_aoi += sum(env.AoI.values())
            eh_step = sum([
                env.model_energy_harvest(env.P_PU,
                    env.compute_large_scale_gain(env.distances_SU_to_PU[u]) * env.compute_small_scale_gain()
                ) for u in env.users
            ])
            total_eh += eh_step

            print(f"[SU={num_su}, Q={Q_max}, TS={t+1}] "
                  f"Action: {env.users[action]}, "
                  f"AoI: {[env.AoI[u] for u in env.users]}, "
                  f"Battery: {[round(env.B[u], 3) for u in env.users]}, "
                  f"EH Step: {eh_step:.5f}")

            state = next_state
            if done:
                break

        avg_aoi = total_aoi / time_slots
        aoi_results[Q_max].append(avg_aoi)
        eh_results[Q_max].append(total_eh)
        csv_rows.append([num_su, Q_max, avg_aoi, total_eh])

        print(f"\n→ SU={num_su}, Q_max={Q_max}: Avg AoI={avg_aoi:.2f}, Total EH={total_eh:.3f}\n")

# 🔽 Save to CSV for MATLAB
with open("sac_eval_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

# 📊 Plotting
x = np.arange(len(su_values))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))
bar1 = ax1.bar(x - width/2, aoi_results[2], width, label='AoI (Q=2)', color='skyblue')
bar2 = ax1.bar(x + width/2, aoi_results[4], width, label='AoI (Q=4)', color='orange')

ax1.set_xlabel('Number of Secondary Users (SUs)')
ax1.set_ylabel('Average Sum AoI', color='blue')
ax1.set_xticks(x)
ax1.set_xticklabels(su_values)
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
bar3 = ax2.bar(x - width/2, eh_results[2], width, label='EH (Q=2)', color='green', alpha=0.3)
bar4 = ax2.bar(x + width/2, eh_results[4], width, label='EH (Q=4)', color='red', alpha=0.3)

ax2.set_ylabel('Total Energy Harvested (Joules)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

lines = [bar1, bar2, bar3, bar4]
labels = [bar.get_label() for bar in lines]
ax1.legend(lines, labels, loc='upper left')

plt.title("AoI and Energy Harvested vs Number of SUs (Q=2 vs Q=4)")
plt.tight_layout()
plt.savefig("Dual_Y_BarGraph_AoI_EH_vs_SUs.png")
plt.show()
