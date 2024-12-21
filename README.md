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
