# Neural Learning Agent Documentation

## 1. Project Overview

This project aims to create a neural network agent incorporating deep learning, reinforcement learning, meta-learning, and self-play, inspired by Ilya Sutskever's ideas. The agent is designed to learn and adapt in a custom environment, showcasing advanced AI techniques in a practical setting.

## 2. Environment Setup

- Framework: TensorFlow
- Custom Gym environment: CustomEnv-v0
- Environment location: /home/ubuntu/nueral-learning-agent/environments/custom_env.py

## 3. Neural Network Architecture

The neural network model is created using the `create_model()` function in main.py. Here's a breakdown of the architecture:

- Input shape: (64, 64, 3) - Representing a 64x64 RGB image
- Convolutional layers:
  - Conv2D layer 1: 32 filters, 3x3 kernel, ReLU activation
  - Conv2D layer 2: 64 filters, 3x3 kernel, ReLU activation
- MaxPooling layers: 2x2 pool size, following each Conv2D layer
- Flatten layer: To convert 2D feature maps to 1D feature vector
- Dense layers:
  - Hidden layer: 64 units, ReLU activation
  - Output layer: Linear activation (number of units depends on action space)

## 4. Training Process

Our training process incorporates multiple advanced techniques:

### Main Training Loop
- Implemented in the `main()` function
- Utilizes a persistent GradientTape: `tf.GradientTape(persistent=True)`
- Runs for a specified number of episodes (1000)
- Incorporates meta-learning with parameters:
  - Number of tasks: 10
  - Inner steps: 20
  - Outer steps: 10

### Reinforcement Learning
- Algorithm: Q-learning
- Policy: Epsilon-greedy (ε = 0.1)
- Discount factor (γ): 0.99

### Gradient Handling
- The `meta_learning_update` function includes logic to handle None gradients, improving stability and error reporting
- Enhanced error handling and logging have been implemented to diagnose gradient computation issues during training

### Meta-learning
- Adaptation to different tasks through rapid learning
- Implemented in the `meta_learning_update()` function

### Self-play
- Generation of training data through agent vs. agent gameplay
- Implemented in the `self_play()` function

## 5. Custom Environment (CustomEnv-v0)

A custom OpenAI Gym environment has been created for training:

- Observation space: Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
- Action space: Discrete(2)
- Step method: Generates random state, reward, and done flag
- Reset method: Initializes state randomly
- Render method: Prints current state

## 6. Key Functions

- `create_model()`: Constructs the neural network model
- `main()`: Implements the main training loop
- `meta_learning_update()`: Handles meta-learning updates
- `self_play()`: Manages self-play for data generation

## 7. Performance Metrics

- Episode rewards are calculated and reported during training
- Metrics are logged to 'episode_rewards.json' for performance analysis and visualization
- Per-episode rewards provide insights into the agent's learning progress over time
- The logged data can be used to generate learning curves and assess model performance

## 8. Training Parameters

- Number of episodes: 1000
- Epsilon (for epsilon-greedy policy): 0.1
- Gamma (discount factor): 0.99
- Meta-learning parameters:
  - Number of tasks: 10
  - Inner steps: 20
  - Outer steps: 10

## 9. Known Issues and Future Improvements

- The script is currently running on CPU due to lack of CUDA drivers, which may impact training speed
- Recent improvements in gradient handling have addressed some stability issues, but further optimization may be needed
- Ongoing monitoring of the training process is required to ensure consistent performance across episodes
- The training process is still ongoing, and further iterations may be needed to achieve optimal performance
- Future work could include:
  - Optimizing the model architecture
  - Fine-tuning hyperparameters
  - Expanding the complexity of the custom environment
  - Implementing more advanced meta-learning techniques
  - Exploring the use of GPU acceleration to improve training speed

This documentation provides an overview of the neural learning agent's architecture, training process, and environment. For more detailed information, please refer to the source code and comments within the implementation files.