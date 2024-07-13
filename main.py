import tensorflow as tf
import numpy as np
import gym
from environments.custom_env import CustomEnv
import copy

def create_model(input_shape, action_space):
    # Define the neural network architecture for image processing
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(action_space, activation='linear')  # Output layer for RL
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')  # Mean Squared Error for RL

    return model

def train_model(model, env, episode_rewards, episodes=1000, gamma=0.99, epsilon=0.1):
    # Implement training loop with reinforcement learning
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 64, 64, 3])  # Reshape state for model input
        done = False
        total_reward = 0

        while not done:
            # Choose action using epsilon-greedy policy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(state))

            # Take action and observe new state and reward
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 64, 64, 3])
            total_reward += reward

            # Update Q-value (simplified Q-learning update)
            target = reward + gamma * np.max(model.predict(next_state))
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)

            state = next_state

        episode_rewards.append(total_reward)
        print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}")

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

    # Train model
    model, episode_rewards = train_model(model, env, episode_rewards)

    # Implement meta-learning with increased complexity
    model = meta_learning_update(model, env, num_tasks=10, inner_steps=20, outer_steps=10)

    # Implement self-play
    model = self_play(model, env)

    # Evaluate results
    total_reward = 0
    episodes = 100
    for _ in range(episodes):
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

    print(f"Evaluation complete. Average reward over {episodes} episodes: {total_reward / episodes}")

    # Save episode rewards to a file
    import json
    with open('episode_rewards.json', 'w') as f:
        json.dump(episode_rewards, f)

    # Close the environment
    env.close()

if __name__ == '__main__':
    main()