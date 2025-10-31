import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from environment_SU_var import CRNOMAEnv  # your environment file

# SAC models and corresponding buffer sizes
models_info = {
    2: "sac_cr_noma_actor_SU3_Q2.h5",
    4: "sac_cr_noma_actor_SU3_Q4.h5",
    6: "sac_cr_noma_actor_SU3_Q6.h5"
}

episodes = 100
time_slots = 250
num_SUs = 3
lambda_arrival = 0.3   # arrival rate for all SUs

# Results storage
selection_results = []
queue_distribution_results = []

for Q_val, model_path in models_info.items():
    print(f"\n=== Running Evaluation for Q_max={Q_val} with model={model_path} ===")
    env = CRNOMAEnv(time_slots=time_slots, num_SUs=num_SUs, Q_max=Q_val)
    env.lambda_arrival = lambda_arrival   # ✅ set arrival rate
    model = load_model(model_path)

    # Counters
    counts_selected = {u: 0 for u in env.users}
    queue_counts = {u: np.zeros(Q_val + 1, dtype=int) for u in env.users}
    total_slots = episodes * time_slots

    for ep in range(episodes):
        state = env.reset()
        env.lambda_arrival = lambda_arrival  # enforce every reset
        for t in range(env.T):
            state_input = state.reshape(1, -1)
            action_probs = model(state_input).numpy()[0]
            action = np.argmax(action_probs)  # greedy choice

            # Step environment
            next_state, reward, done, info = env.step(action)
            selected_su = env.users[action]
            counts_selected[selected_su] += 1

            # Record queue lengths for all SUs
            for u in env.users:
                q_len = len(env.Q[u])
                queue_counts[u][q_len] += 1

            state = next_state
            if done:
                break

    # --- Aggregate selection fractions ---
    row_selection = {"Q_max": Q_val}
    for u in env.users:
        pct = (counts_selected[u] / total_slots) * 100
        row_selection[u + "_selected_pct"] = pct
        row_selection[u + "_idle_pct"] = 100 - pct
    selection_results.append(row_selection)

    # --- Aggregate queue distributions ---
    for u in env.users:
        total_obs = sum(queue_counts[u])
        probs = queue_counts[u] / total_obs
        row_queue = {"Q_max": Q_val, "SU": u}
        for q_len, p in enumerate(probs):
            row_queue[f"P(Q={q_len})"] = p * 100  # percentage
        queue_distribution_results.append(row_queue)

# Save results
df_selection = pd.DataFrame(selection_results)
df_queue = pd.DataFrame(queue_distribution_results)

df_selection.to_csv("SU_selection_results.csv", index=False)
df_queue.to_csv("Queue_distribution_results.csv", index=False)

print("\n✅ Results saved:")
print(" - SU_selection_results.csv (selection + idle fractions)")
print(" - Queue_distribution_results.csv (queue occupancy distributions)")
