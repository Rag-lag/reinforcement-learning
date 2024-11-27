# Taxi Navigation with Reinforcement Learning

This project implements Q-learning and SARSA (State-Action-Reward-State-Action) algorithms for solving the Taxi-v3 environment from OpenAI Gym. The goal is to train a taxi agent to navigate efficiently while picking up and dropping off passengers at designated locations.

## Project Overview

The taxi operates in a 5x5 grid world where it needs to:
1. Navigate to a passenger's pickup location
2. Pick up the passenger
3. Navigate to the correct drop-off location
4. Drop off the passenger

## Requirements

- Python 3.7+
- OpenAI Gym
- NumPy
- Matplotlib

Install the required packages using:
```bash
pip install gym numpy matplotlib
```

## Implementation Details

### Algorithms

1. **Q-Learning Implementation**
   - Uses epsilon-greedy exploration strategy
   - Implements decaying epsilon for better exploration-exploitation balance
   - Hyperparameters:
     - Learning rate (alpha): 0.7
     - Discount factor (gamma): 0.8
     - Initial epsilon: 1.0
     - Episodes: 4000

2. **SARSA Implementation**
   - On-policy learning algorithm
   - Also uses epsilon-greedy strategy
   - Hyperparameters:
     - Learning rate (alpha): 0.7
     - Discount factor (gamma): 0.7
     - Initial epsilon: 1.0
     - Episodes: 4000

### Key Functions

- `qLearn(env, alpha, gamma, episodes, epsilon, epsilonFlag)`: Implements Q-learning algorithm
- `sarsa(env, alpha, gamma, epsilon, episodes, epsilonFlag)`: Implements SARSA algorithm
- `evalQ(env, table, episodes)`: Evaluates Q-learning agent performance
- `evalSarsa(env, table, episodes)`: Evaluates SARSA agent performance
- `visualAgent(env, table)`: Visualizes agent behavior in the environment

## Usage

1. **Training the Agents**:
```python
# Train Q-learning agent
qTab, qRewards, qSteps = qLearn(env, alpha, gamma, episodes, epsilon, epsilonFlag)

# Train SARSA agent
sarsaTab, sarsaRewards, sarsaSteps = sarsa(env, alpha, gamma, epsilon, episodes, epsilonFlag)
```

2. **Evaluating the Agents**:
```python
# Evaluate Q-learning agent
evalQ(env, qTab, numEpisodes)

# Evaluate SARSA agent
evalSarsa(env, sarsaTab, numEpisodes)
```

3. **Visualizing Agent Behavior**:
```python
# Visualize Q-learning agent
visualAgent(env, qTab)

# Visualize SARSA agent
visualAgent(env, sarsaTab)
```

## Results

The project includes visualization of:
- Accumulated rewards over episodes
- Number of steps taken per episode
- Average performance metrics for both algorithms
- Visual demonstration of learned policies

## Saving and Loading Models

Q-tables for both algorithms are automatically saved after training:
```python
# Save Q-tables
np.save('q_table.npy', qTab)
np.save('sarsa_table.npy', sarsaTab)

# Load Q-tables
qTab = np.load('q_table.npy')
sarsaTab = np.load('sarsa_table.npy')
```

## Performance Metrics

The evaluation functions provide:
- Average steps per episode
- Average accumulated reward per episode
- Success rate in completing episodes

## Customization

You can modify the hyperparameters in the code to experiment with:
- Learning rates
- Discount factors
- Exploration rates (epsilon)
- Number of training episodes
- Epsilon decay rate

## Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## License

This project is open source and available under the MIT License.
