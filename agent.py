import torch
import random 
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# Constants for memory capacity, batch size, and learning rate
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3) # Neural network: input size 11, hidden 256, output 3 actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # Trainer for the neural network

    def get_state(self, game):
        head = game.snake[0] # Position of the snake's head
        point_l = Point(head.x - 20, head.y) # Position to the left of the head
        point_r = Point(head.x + 20, head.y) # Position to the right of the head
        point_u = Point(head.x, head.y - 20) # Position above the head
        point_d = Point(head.x, head.y + 20) # Position below the head

        # Boolean flags for the current direction of the snake
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Construct the state as a list of booleans:
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)), 

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y, # food down
        ]
    
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # Append experience tuple to memory
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: # Randomly sample a batch if memory has more than BATCH_SIZE experiences
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else: # Use the entire memory if it is smaller than BATCH_SIZE
            mini_sample = self.memory
        
        # Unpack the batch into individual components
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # Train the model on the batch
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done) # Trains the model using a single experience

    def get_action(self, state):
         # Decrease randomness (exploration) as more games are played
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2) # Exploration: Choose a random action
            final_move[move] = 1
        else:
            # Exploitation: Choose action based on model's prediction
            state0 = torch.tensor(state, dtype=torch.float) # Convert state to tensor
            prediction = self.model(state0) # Get Q-values from the model
            move = torch.argmax(prediction).item() # Get the action with the highest Q-value
            final_move[move] = 1
        
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent() # Create a new agent
    game = SnakeGameAI() # Create a new game instance
    while True:
         # Get the current state of the game
        state_old = agent.get_state(game)

        # Determine the next action
        final_move = agent.get_action(state_old)

        # Execute the action and observe the new state and reward
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        ## Train the model with short-term memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Store the experience in long-term memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Game is over; reset and train on long-term memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                # Save the model if a new high score is achieved
                record = score 
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)  # Log game information

            # Update plots
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train() # Start the training process