import tensorflow as tf
import numpy as np
import gym
from environments.custom_env import CustomEnv
import copy
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def create_model(input_shape, action_space):
    # Define the neural network architecture for image processing
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(action_space, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(optimizer=optimizer, loss='mse')

    return model

def train_model(model, env, episode_rewards, episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    # Implement training loop with reinforcement learning using Double DQN
    target_model = tf.keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())

    epsilon = epsilon_start
    replay_buffer = []
    batch_size = 32

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 64, 64, 3])  # Reshape state for model input
        done = False
        total_reward = 0

        while not done:
            # Choose action using epsilon-greedy policy with decaying epsilon
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(state))

            # Take action and observe new state and reward
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 64, 64, 3])
            total_reward += reward

            # Store experience in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > 10000:
                replay_buffer.pop(0)

            # Train on a batch of experiences
            if len(replay_buffer) >= batch_size:
                batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
                for i in batch:
                    s, a, r, ns, d = replay_buffer[i]
                    target = r
                    if not d:
                        # Double DQN update
                        a_max = np.argmax(model.predict(ns)[0])
                        target = r + gamma * target_model.predict(ns)[0][a_max]
                    target_f = model.predict(s)
                    target_f[0][a] = target
                    model.fit(s, target_f, epochs=1, verbose=0)

            state = next_state

        # Update target model
        if episode % 10 == 0:
            target_model.set_weights(model.get_weights())

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    return model, episode_rewards

def meta_learning_update(model, env, num_tasks=10, inner_steps=20, outer_steps=10, alpha=0.01, beta=0.001, gamma=0.99):
    original_weights = model.get_weights()

    for _ in range(outer_steps):
        gradients = []
        for _ in range(num_tasks):
            task_model = tf.keras.models.clone_model(model)
            task_model.set_weights(original_weights)
            task_env = gym.make('CustomEnv-v0')
            task_env.reset()

            # Inner loop: adapt to the task
            for _ in range(inner_steps):
                state = task_env.reset()
                state = np.reshape(state, [1, 64, 64, 3])  # Reshape state for model input
                done = False

                while not done:
                    action = np.argmax(task_model.predict(state))
                    next_state, reward, done, _ = task_env.step(action)
                    next_state = np.reshape(next_state, [1, 64, 64, 3])  # Reshape next_state

                    target = reward + gamma * np.max(task_model.predict(next_state))
                    target_f = task_model.predict(state)
                    target_f[0][action] = target

                    with tf.GradientTape(persistent=True) as tape:
                        predictions = task_model(state)
                        loss = tf.keras.losses.mse(target_f, predictions)

                    grads = tape.gradient(loss, task_model.trainable_variables)
                    for j, g in enumerate(grads):
                        task_model.trainable_variables[j].assign_sub(alpha * g)

                    state = next_state

            # Compute gradient for meta-update
            final_loss = tf.keras.losses.mse(target_f, task_model(state))
            task_gradients = tape.gradient(final_loss, task_model.trainable_variables)
            if any(g is None for g in task_gradients):
                print("Warning: Some gradients are None. This may indicate unused variables.")
            gradients.append(task_gradients)
            del tape  # Delete the tape to free up resources

        # Meta-update
        meta_gradients = [tf.reduce_mean([g[i] for g in gradients if g[i] is not None], axis=0) for i in range(len(gradients[0]))]
        for i, g in enumerate(meta_gradients):
            model.trainable_variables[i].assign_sub(beta * g)

    return model

def self_play(model, env, episodes=100, gamma=0.99):
    replay_buffer = []

    for _ in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 64, 64, 3])  # Reshape for image input
        done = False

        while not done:
            action = np.argmax(model.predict(state))
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 64, 64, 3])  # Reshape for image input

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state

    # Train on collected data
    for state, action, reward, next_state, done in replay_buffer:
        target = reward
        if not done:
            target = reward + gamma * np.max(model.predict(next_state))
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

    return model

def main():
    # Create or load environment
    env = gym.make('CustomEnv-v0')

    # Create model
    state_shape = (64, 64, 3)  # Updated to match the expected input shape
    action_space = env.action_space.n
    model = create_model(state_shape, action_space)

    # Initialize list to store episode rewards
    episode_rewards = []

    # Train model with improved parameters
    episodes = 2000  # Increased number of episodes
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    model, episode_rewards = train_model(model, env, episode_rewards, episodes=episodes,
                                         epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                                         epsilon_decay=epsilon_decay)

    # Implement meta-learning with increased complexity
    model = meta_learning_update(model, env, num_tasks=15, inner_steps=25, outer_steps=15)

    # Implement self-play with more episodes
    model = self_play(model, env, episodes=200)

    # Evaluate results
    total_reward = 0
    eval_episodes = 100
    for _ in range(eval_episodes):
        state = env.reset()
        state = np.reshape(state, [1, 64, 64, 3])  # Updated reshaping
        done = False
        episode_reward = 0

        while not done:
            action = np.argmax(model.predict(state))
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 64, 64, 3])  # Updated reshaping
            episode_reward += reward
            state = next_state

        total_reward += episode_reward

    print(f"Evaluation complete. Average reward over {eval_episodes} episodes: {total_reward / eval_episodes}")

    # Save episode rewards and model
    import json
    with open('episode_rewards.json', 'w') as f:
        json.dump(episode_rewards, f)

    model.save('trained_model.h5')

    # Close the environment
    env.close()

if __name__ == '__main__':
    main()