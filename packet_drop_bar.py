import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from environment_SU_var import CRNOMAEnv  # Replace with the actual module if needed

arrival_rates = [0.2, 0.3, 0.4, 0.5,0.6, 0.7]
num_SUs_list = [2, 3]
Q_max_values = [2, 4]
time_slots = 1000

results = []

# Mapping of actor file names
actor_paths = {
    (2, 2): 'sac_cr_noma_actor_SU2_Q2.h5',
    (2, 4): 'sac_cr_noma_actor_SU2_Q4.h5',
    (3, 2): 'sac_cr_noma_actor_SU3_Q2.h5',
    (3, 4): 'sac_cr_noma_actor_SU3_Q4.h5'
}

for num_SUs in num_SUs_list:
    for Q_max in Q_max_values:
        actor_path = actor_paths[(num_SUs, Q_max)]
        print(f"\n=== Loading Model: {actor_path} for SU={num_SUs}, Q_max={Q_max} ===")
        actor_model = load_model(actor_path)

        for lam in arrival_rates:
            print(f"\n--- Running Simulation: λ={lam}, SU={num_SUs}, Q={Q_max} ---")
            env = CRNOMAEnv(time_slots=time_slots, num_SUs=num_SUs, Q_max=Q_max)
            env.lambda_arrival = lam
            obs = env.reset()
            done = False
            time_step = 0
            total_dropped = 0

            while not done:
                obs_reshaped = obs.reshape(1, -1)
                action_probs = actor_model(obs_reshaped)
                action = np.argmax(action_probs)

                obs, reward, done, info = env.step(action)
                dropped = info.get("dropped_packets", 0)
                total_dropped += dropped

                print(f"[λ={lam} | SU={num_SUs} | Q={Q_max}] Time Slot {time_step+1}/{time_slots} → "
                      f"Action: {action}, Dropped: {dropped}, Total Dropped: {total_dropped}, Reward: {reward:.2f}")
                time_step += 1

            total_arrived = lam * num_SUs * time_slots
            drop_prob = total_dropped / total_arrived
            print(f"→ Finished λ={lam}, SU={num_SUs}, Q_max={Q_max}: Drop Probability = {drop_prob:.4f}")

            results.append({
                'Lambda': lam,
                'Num_SUs': num_SUs,
                'Q_max': Q_max,
                'Drop_Probability': drop_prob
            })

# Save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("sac_packet_drop_results.csv", index=False)
print("\n✅ Results saved to 'sac_packet_drop_results.csv'")
