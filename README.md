# Snake-AI 🐍🌐
This project implements a deep Q-learning AI to play Snake. It uses a neural network to predict optimal moves based on the game state, trains through experience replay, and visualizes performance over time.

## Credits 🤖
[![Python + PyTorch + Pygame Reinforcement Learning – Train an AI to Play Snake](https://www.youtube.com/watch?v=L8ypSXwyBds&list=LL.jpg)](https://www.youtube.com/watch?v=L8ypSXwyBds&list=LL) - 
**freeCodeCamp.org**.
The base code for this project was adapted from freeCodeCamp.org. While the original concept and code were used as a foundation, style modifications were made to suit the features of this Snake AI Simulation.

## Demo 🎬

## The Code files: 📄

**agent.py:** Contains the main AI agent logic, including state extraction, memory management, training steps, and gameplay decisions.

**model.py:** Defines the neural network architecture and the training process, including saving and loading models.

**helper.py:** Handles real-time plotting of scores and performance metrics during training.

**game.py:** Implements the Snake game logic, game environment, and interaction with the AI agent.

## Functionality ⚙️
**Training:**
- Uses Deep Q-Learning with experience replay.
- Optimizes a neural network to predict the best moves based on the current game state.

**Gameplay:**
- The agent interacts with the Snake game environment to learn optimal strategies.
- Tracks performance metrics, such as scores and averages, during training.

## Installation 💻
Ensure you have the following dependencies installed:

*pip install torch==2.5.1+cu118*

*pip install tensorflow==2.18.0*

*pip install matplotlib==3.9.2*

*pip install ipython==8.31.0*

*pip install pygame==2.6.1*

*pip install numpy==1.26.4*

## Usage 📌
- Clone the repository and navigate to the project folder.
- Run the training script:

  *python agent.py*

## Theory 💡
#### States (S):
The state represents the current snapshot of the game environment, which the agent uses to decide its next action. In the Snake-AI project, the state is encoded as an **11-dimensional feature vector:**

**Danger Indicators (3):**
- Danger straight (e.g., moving forward leads to collision).
- Danger left (e.g., turning left leads to collision).
- Danger right (e.g., turning right leads to collision).

**Direction (4):**
- Current movement direction (left, right, up, down).

**Food Location (4):**
- Is food to the left, right, above, or below the snake's head?

**Example:**
- For a snake moving upward with food to the left and danger straight, the state could be:
**[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]**

#### Actions (A):
The action space represents all possible moves the snake can make at any step:

**Turn left:** [0,0,1]
**Continue straight:** [1,0,0]
**Turn right:** [0,1,0]

- The agent selects an action based on the predicted Q-values (**expected rewards**) for each action, either randomly (**exploration**) or through learned predictions (**exploitation**).

#### Rewards (R):
Rewards incentivize the agent to learn desirable behaviors:

**+10:** When the snake eats food.
**-10:** When the snake collides with a wall or itself.
**0:** Otherwise (normal movement).

- The reward system guides the agent to prioritize survival and maximize the score.

#### Q-Learning and Bellman’s Equation:

![image](https://github.com/user-attachments/assets/ff3f79f4-4f7e-4774-b97b-b638af8a7791)


- Q-Learning is a reinforcement learning technique where the agent learns to estimate the Q-value for each state-action pair **(𝑆,𝐴).**

- The Q-value is the expected cumulative reward of taking an action **𝐴** in a state **𝑆**, considering future rewards.

 #### Bellman’s Equation:
 The Q-values are updated iteratively using Bellman’s equation:
 
 ![image](https://github.com/user-attachments/assets/0675e138-2ee4-423b-8a8a-b8d572b0bc89)

 where in terms of words it basically means: 
 
 *New Q value for that state = current Q value + (learning rate * reward for taking that action at that state) + (discount rate * maximum predicted reward given new state and all possible actions at that state)*

 but can be **simplified:**

 ![image](https://github.com/user-attachments/assets/a1b020bc-ec13-4540-a89e-d1fe411426f0)

 where old Q value is first line

 and second line is new Q value in words would be:

 *New Q value = reward + gamma value (**discount rate**) * maximum value of Q state with model predict of state 1*

 hence making **loss function:**

 equivalent to Mean Squared Error:

 ![image](https://github.com/user-attachments/assets/b391d9c1-7c9d-425a-b6f3-653841a42da6)


 
 
