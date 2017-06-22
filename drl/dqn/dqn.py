

# Algorithm info
info = {
    'name': 'DQN'
}

# Algorithm options
options = {
    'batch_size': 32,                           # No. of training cases over each SGD update
    'replay_memory_size': 1000000,              # SGD updates sampled from this number of most recent frames
    'agent_history_length': 4,                  # No. of most recent frames given as input to Q network
    'target_network_update_frequency': 10000,   # No. of parameter updates after which target network is updated
    'discount_factor': 0.99,                    # Gamma used in Q-learning update
    'action_repeat': 4,                         # Repeat each action selected by agent this many times. (Using 4 results in agent seeing only every 4th input frame)
    'update_frequency': 4,                      # No. of actions selected by agent between successive SGD updates
    'learning_rate': 0.00025,                   # Learning rate used by RMSProp
    'gradient_momentum': 0.95,                  # Gradient momentum used by RMSProp
    'squared_gradient_momentum': 0.95,          # Squared gradient (denominator) momentum used by RMSProp
    'min_squared_gradient': 0.01,               # Constant added to the squared gradient in denominator of RMSProp update
    'initial_exploration': 1,                   # Initial value of epsilon in epsilon-greedy exploration
    'final_exploration': 0.1,                   # Final value of epsilon in epsilon-greedy exploration
    'final_exploration_frame': 1000000,         # No. of frames over which initial value of epsilon is linearly annealed to it's final value
    'replay_start_size': 50000,                 # A uniform random policy is run for this many frames before learning starts and resulting experience is used to populate the replay memory
    'no-op_max': 30                             # Max no. of do nothing actions to be performed by agent at the start of an episode
}

class DQN(object):

    def __init__(self):
        pass