import numpy as np
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
import random
from collections import deque


class SACAgent:
    def __init__(self, state_size, action_size, gamma=0.99,
                 actor_lr=0.0003, critic_lr=0.0005, alpha=0.2,
                 tau=0.005, batch_size=256, buffer_size=50000):
        
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Entropy coefficient

        self.batch_size = batch_size

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

        # Actor and Critics
        self.actor = self.build_actor()
        self.critic1 = self.build_critic()
        self.critic2 = self.build_critic()

        # Target critics
        self.target_critic1 = self.build_critic()
        self.target_critic2 = self.build_critic()

        # Initialize target weights
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def build_actor(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(self.action_size, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs)
        return model

    def build_critic(self):
        state_input = layers.Input(shape=(self.state_size,))
        action_input = layers.Input(shape=(self.action_size,))

        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        output = layers.Dense(1, activation='linear')(x)

        model = tf.keras.Model([state_input, action_input], output)
        return model

    def get_action(self, state, valid_actions):
        state = state.reshape(1, -1)
        probs = self.actor(state).numpy()[0]

        mask = np.zeros_like(probs)
        mask[valid_actions] = 1.0
        masked_probs = probs * mask

        if masked_probs.sum() == 0:
            masked_probs = mask / mask.sum()
        else:
            masked_probs = masked_probs / masked_probs.sum()

        action = np.random.choice(self.action_size, p=masked_probs)
        return action, masked_probs

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch], dtype=np.float32)
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch], dtype=np.float32)

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        actions_onehot = tf.one_hot(actions, self.action_size, dtype=tf.float32)

        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # ------- Critic Update -------
        next_action_probs = self.actor(next_states)
        next_actions_logp = tf.math.log(next_action_probs + 1e-10)

        # Expand next_states for each action
        batch_size = tf.shape(next_states)[0]
        next_states_tiled = tf.repeat(next_states, repeats=self.action_size, axis=0)
        next_actions_onehot = tf.tile(tf.eye(self.action_size), [batch_size, 1])

        target_q1 = self.target_critic1([next_states_tiled, next_actions_onehot])
        target_q2 = self.target_critic2([next_states_tiled, next_actions_onehot])
        target_min_q = tf.minimum(target_q1, target_q2)
        target_min_q = tf.reshape(target_min_q, (batch_size, self.action_size))

        next_q = tf.reduce_sum(
            next_action_probs * (target_min_q - self.alpha * next_actions_logp), axis=1
        )

        target_q = rewards + (1 - dones) * self.gamma * next_q

        with tf.GradientTape(persistent=True) as tape:
            current_q1 = self.critic1([states, actions_onehot])
            current_q2 = self.critic2([states, actions_onehot])

            critic1_loss = tf.reduce_mean((current_q1[:, 0] - target_q) ** 2)
            critic2_loss = tf.reduce_mean((current_q2[:, 0] - target_q) ** 2)

        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)

        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))

        # ------- Actor Update -------
        with tf.GradientTape() as tape:
            probs = self.actor(states)
            log_probs = tf.math.log(probs + 1e-10)

            states_tiled = tf.repeat(states, repeats=self.action_size, axis=0)
            actions_onehot = tf.tile(tf.eye(self.action_size), [batch_size, 1])

            q1 = self.critic1([states_tiled, actions_onehot])
            q2 = self.critic2([states_tiled, actions_onehot])
            q1 = tf.reshape(q1, (batch_size, self.action_size))
            q2 = tf.reshape(q2, (batch_size, self.action_size))

            min_q = tf.minimum(q1, q2)

            actor_loss = tf.reduce_mean(
                tf.reduce_sum(probs * (self.alpha * log_probs - min_q), axis=1)
            )

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # ------- Soft Update -------
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

    def soft_update(self, source, target):
        for src_var, tgt_var in zip(source.trainable_variables, target.trainable_variables):
            tgt_var.assign(self.tau * src_var + (1 - self.tau) * tgt_var)
