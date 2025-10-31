import numpy as np
import pandas as pd
import tensorflow as tf
from environment_SU_var import CRNOMAEnv
from SAC_Agent_SU_var import SACAgent

# Parameters
T = 1000
arrival_rates = np.arange(0.1, 0.71, 0.1)
Q_values = [2, 4]
output_rows = []
B = 1e6  # Hz
N0 = 1e-9  # Noise power

# Load actor models
actor_models = {
    2: tf.keras.models.load_model("sac_cr_noma_actor_SU3_Q2.h5"),
    4: tf.keras.models.load_model("sac_cr_noma_actor_SU3_Q4.h5")
}

for Q in Q_values:
    for lam in arrival_rates:
        print(f"\n=== Running for Qmax={Q}, λ={lam:.1f} ===")

        # Initialize environment
        env = CRNOMAEnv(time_slots=T, num_SUs=3, Q_max=Q)
        env.lambda_arrival = lam

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = SACAgent(state_size, action_size)
        agent.actor = actor_models[Q]  # Replace with loaded model

        state = env.reset()
        total_aoi_sum = 0
        total_energy_harvested = 0

        for t in range(env.T):
            valid_actions = env.get_valid_actions()

            state_input = state.reshape(1, -1)
            probs = agent.actor(state_input).numpy()[0]

            if valid_actions:
                mask = np.zeros_like(probs)
                mask[valid_actions] = 1.0
                masked_probs = probs * mask

                if masked_probs.sum() == 0 or np.isnan(masked_probs).any():
                    masked_probs = mask / mask.sum()
                else:
                    masked_probs = masked_probs / masked_probs.sum()

                action = np.random.choice(env.num_SUs, p=masked_probs)
            else:
                action = np.random.choice(env.num_SUs)

            # 🟢 Step and use true harvested energy from info dict
            next_state, reward, done, info = env.step(action)

            # ✅ Use accurate EH from environment
            total_energy_harvested += info.get("harvested_energy", 0)

            # ✅ Track AoI
            total_aoi_sum += sum([env.AoI[u] for u in env.users])

            # Logging
            print(f"[t={t + 1}] λ={lam:.1f}, Q={Q}, AoI={env.AoI}, EH+{info.get('harvested_energy', 0):.4f}, Rwd={reward:.2f}")

            state = next_state
            if done:
                break

        avg_sum_aoi = total_aoi_sum / T
        output_rows.append({
            "Lambda": lam,
            "Qmax": Q,
            "Avg_AoI": avg_sum_aoi,
            "Total_EH": total_energy_harvested
        })

# Save results
df = pd.DataFrame(output_rows)
df.to_csv("lambda_vs_aoi_eh_.csv", index=False)
print("\n✅ Evaluation complete. Results saved to lambda_vs_aoi_eh.csv")
