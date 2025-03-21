import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
#TO-DO:\S   \S
# 1. Model a smarter player that starts out by exploring and then switches to exploitation after n trials.
# Track how often behavioral and mocap disagree
# Look into combining the results into a single intelligence
# 2. Generate Q tables for different player strategies / house parameters and compare them. Weight them and average them.
#    How sneaky the player is. Have different phases for when the player "catches on" to the house's strategy.
# 3. Implement a more sophisticated learning algorithm (e.g., SARSA, DQN, etc.). ???
# 4. Implement a more complex reward function.
# 5. Implement a more complex observation space.
#    Augment the state space: 
#    Continue training real-time and wait those trials higher with a higher learning rate
#    We'll have several different models telling us what to do and we'll have to weight them into one output. Assign them a confidence 
# 6. Implement a more complex action space.
# 7. Figure out how to store different Q-tables to compare different learning strategies.
# --- Environment Definition ---
class BanditEnv(gym.Env):
    """
    A bandit environment for a casino with a learning player.
    
    - There are 4 machines.
    - The environment state is a 4-element vector of actual win probabilities (between 0 and 1).
    - The player's perception of each machine's win probability is maintained separately.
    - The action space is discrete (0 to 7):
         For each machine (0,1,2,3):
             - Even-numbered actions increase its actual win probability by 0.1.
             - Odd-numbered actions decrease it by 0.1.
    - After the casino adjusts one machine's probability, the player selects which machine to play.
      The player chooses the machine with the highest perceived win probability.
    - The outcome of the play is simulated using the actual win probability:
         * If the player wins (with probability equal to the actual win chance), the casino loses $1.
         * Otherwise, the casino gains $1.
    - After the play, the player's perception for the played machine is updated:
         new_perception = old_perception + player_lr * (observed_outcome - old_perception),
         where observed_outcome is 1 if win and 0 if loss.
    """
    def __init__(self):
        super(BanditEnv, self).__init__()
        # Action space: 8 discrete actions (4 machines × 2 adjustments)
        self.action_space = spaces.Discrete(8)
        # Observation space: actual win probabilities for 4 machines (continuous between 0 and 1)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Actual win probabilities (the casino's setting)
        self.state = [0.5, 0.5, 0.5, 0.5]
        self.adjustment = 0.1  # adjustment step for the actual win probabilities
        
        # Player's perception of win probabilities (initially set to 0.5 for all machines)
        self.player_perception = [0.5, 0.5, 0.5, 0.5]
        self.player_lr = 0.1    # player's learning rate for updating their perception

    def reset(self, seed=None, options=None):
        # Reset both the actual win probabilities and the player's perception.
        self.state = [0.5, 0.5, 0.5, 0.5]
        self.player_perception = [0.5, 0.5, 0.5, 0.5]
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        # --- Casino adjusts one machine's actual win probability ---
        machine_to_adjust = action // 2
        change = self.adjustment if (action % 2 == 0) else -self.adjustment
        self.state[machine_to_adjust] += change
        # Keep the actual probability within [0, 1]
        self.state[machine_to_adjust] = max(0.0, min(1.0, self.state[machine_to_adjust]))
        
        # --- Player's turn: Choose a machine to play based on perceived win probabilities ---
        # The player selects the machine with the highest perceived win probability.
        played_machine = int(np.argmax(self.player_perception))
        
        # --- Simulate the play using the actual win probability of the chosen machine ---
        p_actual = self.state[played_machine]
        if np.random.rand() < p_actual:
            outcome = 1  # player wins
            reward = -1  # casino loses $1
        else:
            outcome = 0  # player loses
            reward = 1   # casino gains $1
        
        # --- Update player's perception for the played machine ---
        old_perception = self.player_perception[played_machine]
        self.player_perception[played_machine] = old_perception + self.player_lr * (outcome - old_perception)
        
        # In this bandit problem, episodes do not have a terminal state.
        done = False
        truncated = False
        info = {}
        return np.array(self.state, dtype=np.float32), reward, done, truncated, info

# --- Q-Learning Setup ---

num_episodes = 10000
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1.0           # initial exploration rate (epsilon)
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

env = BanditEnv()
num_actions = env.action_space.n

# Use a dictionary-based Q-table; since the state is continuous, round the values to two decimals.
Q = {}

def get_q(state):
    """Return the Q-values for a given state (rounded to 2 decimals), initializing to zeros if unseen."""
    state_key = tuple(np.round(state, 2))
    if state_key not in Q:
        Q[state_key] = np.zeros(num_actions)
    return Q[state_key]

rewards_all_episodes = []

# --- Q-Learning Algorithm ---
for episode in range(num_episodes):
    state, _ = env.reset()
    total_rewards = 0

    for step in range(max_steps_per_episode):
        # Epsilon-greedy action selection:
        if random.uniform(0, 1) > exploration_rate:
            action = int(np.argmax(get_q(state)))
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, truncated, info = env.step(action)
        total_rewards += reward

        # Q-learning update:
        state_key = tuple(np.round(state, 2))
        old_q_value = get_q(state)[action]
        next_max = np.max(get_q(new_state))
        new_q_value = old_q_value + learning_rate * (reward + discount_rate * next_max - old_q_value)
        Q[state_key][action] = new_q_value

        state = new_state
        if done or truncated:
            break

    # Decay exploration rate after each episode.
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * \
                         np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(total_rewards)

print("Training finished.\n")
print("Average reward per episode:", np.mean(rewards_all_episodes))
