import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from env_test import CRNOMAEnv  # Make sure your CRNOMAEnv class is defined/imported

# Load trained actor models
ddpg_actor = load_model('ddpg_cr_noma_actor.h5')
sac_actor = load_model('sac_cr_noma_actor.h5')
ppo_actor = load_model('ppo_cr_noma_actor.h5')

models = {
    "DDPG": ddpg_actor,
    "SAC": sac_actor,
    "PPO": ppo_actor
}

lambda_values = [0.2, 0.3, 0.5, 0.7, 0.8]

def evaluate_drop_for_lambda(model, lambda_val, time_slots=1000, log=True):
    env = CRNOMAEnv(time_slots=time_slots)
    env.lambda_arrival = lambda_val
    state = env.reset()

    total_arrivals = 0
    total_drops = 0
    done = False
    step_counter = 0

    while not done:
        step_counter += 1

        # Predict action from model
        action_probs = model.predict(state.reshape(1, -1), verbose=0)[0]
        action = np.argmax(action_probs)

        # Step through the environment
        state, reward, done, info = env.step(action)

        # Count dropped packets and arrivals
        step_drops = info.get("dropped_packets", 0)
        total_drops += step_drops
        for u in env.users:
            total_arrivals += np.random.poisson(env.lambda_arrival)

        # 🔍 LOGGING
        if log:
            print(f"Step {step_counter:>4} | Action: {env.users[action]} | "
                  f"Dropped: {step_drops} | Total Drops: {total_drops} | "
                  f"Buffers: {[len(env.Q[u]) for u in env.users]} | "
                  f"AoI: {[env.AoI[u] for u in env.users]}")

    drop_prob = total_drops / total_arrivals if total_arrivals > 0 else 0
    print(f"\n✅ Completed episode for λ = {lambda_val} | Drop Probability = {drop_prob:.4f}\n")
    return drop_prob

# Run evaluation
results = {name: [] for name in models}
for lambda_val in lambda_values:
    for name, model in models.items():
        print(f"\n🔁 Evaluating {name} for λ = {lambda_val}")
        prob = evaluate_drop_for_lambda(model, lambda_val, time_slots=1000, log=True)
        results[name].append(prob)

# ------------------------------
# 📊 Bar Plot: Drop Probability vs λ
# ------------------------------
bar_width = 0.25
x = np.arange(len(lambda_values))
r1 = x
r2 = r1 + bar_width
r3 = r2 + bar_width

plt.figure(figsize=(10, 6))
plt.bar(r1, results['DDPG'], width=bar_width, label='DDPG', color='orange')
plt.bar(r2, results['SAC'], width=bar_width, label='SAC', color='steelblue')
plt.bar(r3, results['PPO'], width=bar_width, label='PPO', color='seagreen')

plt.xlabel('Arrival Rate λ', fontsize=12)
plt.ylabel('Packet Drop Probability', fontsize=12)
plt.title('Packet Drop Probability vs λ (1000 Time Slots)', fontsize=14)
plt.xticks([r + bar_width for r in x], lambda_values)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('packet_drop_vs_lambda.png')