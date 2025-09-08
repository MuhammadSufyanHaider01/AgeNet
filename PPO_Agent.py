import numpy as np
import tensorflow as tf
from tensorflow.keras import layers # type: ignore

class PPOAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-4,
                 clip_ratio=0.2, lambd=0.9, policy_epochs=20, batch_size=512):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.lambd = lambd
        self.clip_ratio = clip_ratio
        self.policy_epochs = policy_epochs
        self.batch_size = batch_size

        # Build models
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Memory for trajectories
        self.reset_buffer()

    def build_actor(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        outputs = layers.Dense(self.action_size, activation='softmax')(x)

        return tf.keras.Model(inputs, outputs)

    def build_critic(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        outputs = layers.Dense(1, activation='linear')(x)

        return tf.keras.Model(inputs, outputs)

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def get_action(self, state, valid_actions):
        state = state.reshape(1, -1)
        probs = self.actor(state).numpy()[0]

        # Mask invalid actions
        mask = np.full_like(probs, 0.0)
        mask[valid_actions] = 1.0
        masked_probs = probs * mask

        if masked_probs.sum() == 0:
            masked_probs = mask / mask.sum()

        masked_probs = masked_probs / masked_probs.sum()

        action = np.random.choice(self.action_size, p=masked_probs)
        log_prob = np.log(masked_probs[action] + 1e-10)

        return action, log_prob

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def compute_gae(self, rewards, dones, values, next_values):
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambd * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def learn(self):
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards, dtype=np.float32)
        next_states = np.vstack(self.next_states)
        dones = np.array(self.dones, dtype=np.float32)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)

        values = self.critic(states).numpy().flatten()
        next_values = self.critic(next_states).numpy().flatten()

        advantages, returns = self.compute_gae(rewards, dones, values, next_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.shape[0]
        for _ in range(self.policy_epochs):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = idx[start:end]

                self.train_step(
                    states[batch_idx],
                    actions[batch_idx],
                    old_log_probs[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx]
                )

        self.reset_buffer()

    @tf.function
    def train_step(self, states, actions, old_log_probs, advantages, returns):
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        with tf.GradientTape(persistent=True) as tape:
            probs = self.actor(states)
            action_probs = tf.gather(probs, actions[:, None], batch_dims=1)
            log_probs = tf.math.log(action_probs + 1e-10)

            ratios = tf.exp(log_probs - old_log_probs[:, None])

            clip_adv = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[:, None]
            actor_loss = -tf.reduce_mean(tf.minimum(ratios * advantages[:, None], clip_adv))

            values = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(returns[:, None] - values))

            # ✅ Add entropy regularization
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
            entropy_bonus = 0.01 * tf.reduce_mean(entropy)

            total_loss = actor_loss + 0.5 * critic_loss - entropy_bonus

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
