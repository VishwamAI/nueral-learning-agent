import unittest
from src.neural_network_agent import NeuralNetworkAgent
from src.environments.custom_env import CustomEnv

class TestNeuralNetworkAgent(unittest.TestCase):

    def setUp(self):
        # Set up for each test
        self.env = CustomEnv()
        input_shape = self.env.observation_space.shape
        action_space = self.env.action_space.n
        self.agent = NeuralNetworkAgent(input_shape, action_space)

    def test_initialization(self):
        # Test initialization of the neural network model
        self.assertIsInstance(self.agent, NeuralNetworkAgent, "Agent is not an instance of NeuralNetworkAgent")
        self.assertIsNotNone(self.agent.model, "Neural network model is not initialized")

    def test_action(self):
        # Test the agent's ability to take actions
        state = self.env.reset()
        action = self.agent.act(state)
        self.assertIn(action, range(self.env.action_space.n), "Action taken is not within the environment's action space")

    def test_environment_interaction(self):
        # Test the agent's interaction with the Gym environment
        initial_state = self.env.reset()
        action = self.agent.act(initial_state)
        state, reward, done, info = self.env.step(action)
        self.assertIsNotNone(state, "Environment did not return a new state after taking an action")
        self.assertIsInstance(reward, (int, float), "Reward is not a numeric value")
        self.assertIsInstance(done, bool, "Done flag is not a boolean")

    def test_learning(self):
        # Test the learning process of the agent
        initial_state = self.env.reset()
        action = self.agent.act(initial_state)
        next_state, reward, done, _ = self.env.step(action)

        # Store the initial model parameters
        initial_params = self.agent.get_model_parameters()

        # Perform learning
        self.agent.learn(initial_state, action, reward, next_state, done)

        # Check if the model parameters have changed after learning
        updated_params = self.agent.get_model_parameters()
        self.assertNotEqual(initial_params, updated_params, "Agent's model parameters did not change after learning")

if __name__ == '__main__':
    unittest.main()