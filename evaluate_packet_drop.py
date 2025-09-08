import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from env_test import CRNOMAEnv  # Replace with your env file or inline class

# ----------------------
# ✅ Load DRL Models
# ----------------------
print("🔄 Loading trained actor models...")

sac_actor = load_model('sac_cr_noma_actor.h5')
print("✅ SAC actor model loaded.")

ppo_actor = load_model('ppo_cr_noma_actor.h5')
print("✅ PPO actor model loaded.")

ddpg_actor = load_model('ddpg_cr_noma_actor.h5')
print("✅ DDPG actor model loaded.")

models = {
    "SAC": sac_actor,
    "PPO": ppo_actor,
    "DDPG": ddpg_actor
}

# ----------------------
# ✅ Evaluation Function
# ----------------------
def evaluate_packet_drop(model, model_name, episodes=10):
    drop_probs = []

    for ep in range(episodes):
        print(f"\n🎬 Running {model_name} Episode {ep + 1}")
        env = CRNOMAEnv(time_slots=250)
        # env.lambda_arrival = 0.8
        state = env.reset()
        total_arrivals = 0
        total_drops = 0
        done = False
        step = 0

        while not done:
            step += 1
            # Inference
            action_probs = model.predict(state.reshape(1, -1), verbose=0)[0]
            action = np.argmax(action_probs)

            next_state, reward, done, info = env.step(action)
            state = next_state

            # Track packet stats
            total_drops += info.get("dropped_packets", 0)
            for u in env.users:
                arrivals = np.random.poisson(env.lambda_arrival)
                total_arrivals += arrivals

            # Optional: Live step log
            print(f"[Step {step}] Action: {env.users[action]} | Drops: {info['dropped_packets']} | Total Arrivals: {total_arrivals}")

        drop_prob = total_drops / total_arrivals if total_arrivals > 0 else 0
        drop_probs.append(drop_prob)

        print(f"✅ Episode {ep + 1} Summary: Total Drops = {total_drops}, Total Arrivals = {total_arrivals}, Drop Probability = {drop_prob:.4f}")

    print(f"\n📊 {model_name} Evaluation Complete. Drop Probabilities: {drop_probs}")
    return drop_probs

# ----------------------
# ✅ Run Evaluation
# ----------------------
episodes = list(range(1, 11))
results = {}

for name, model in models.items():
    results[name] = evaluate_packet_drop(model, model_name=name, episodes=len(episodes))

# ----------------------
# 📈 Plot Results
# ----------------------
bar_width = 0.25
r1 = np.arange(len(episodes))
r2 = r1 + bar_width
r3 = r2 + bar_width

plt.figure(figsize=(10, 6))
plt.bar(r1, results['SAC'], width=bar_width, label='SAC', color='steelblue')
plt.bar(r2, results['PPO'], width=bar_width, label='PPO', color='seagreen')
plt.bar(r3, results['DDPG'], width=bar_width, label='DDPG', color='orange')

plt.xlabel('Episode', fontsize=12)
plt.ylabel('Packet Drop Probability', fontsize=12)
plt.title('Packet Drop Probability Across 10 Episodes (DRL Models)', fontsize=14)
plt.xticks([r + bar_width for r in range(len(episodes))], episodes)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', axis='y', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
# ✅ Save the plot
plt.savefig('packet_drop_probability_comparison.png')