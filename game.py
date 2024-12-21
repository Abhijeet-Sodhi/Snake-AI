import pygame
import random
from enum import Enum # For defining directions as enumerations.
from collections import namedtuple # For creating structured data objects (e.g., Point).
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

# reset 
# reward
# play(action) -> direction
# game_iteration
# is_collision

class Direction(Enum): # Enum for the direction the snake can move.
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y') 

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN1 = (0, 200, 0)
GREEN2 = (0, 150, 0)
BLACK = (13,17,19)

BLOCK_SIZE = 20
SPEED = 20

class SnakeGameAI: # Game environment for AI-based snake.
    
    def __init__(self, w=640, h=480): # Initialize game dimensions.
        self.w = w # Game width.
        self.h = h # Game height.
        # Initialize display window.
        self.display = pygame.display.set_mode((self.w, self.h)) 
        pygame.display.set_caption('Snake') # Set window title.
        self.clock = pygame.time.Clock() # Clock to control game speed.
        self.reset() # Initialize game state.

    def reset(self): # Reset the game to start state.
        
        self.direction = Direction.RIGHT # Initial direction of snake movement.
        
        self.head = Point(self.w/2, self.h/2) # Snake starts in the center.
        self.snake = [self.head, # Initial snake body (head + two segments).
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0 # Initial score.
        self.food = None
        self._place_food()
        self.frame_iteration = 0 # Track the number of frames in the current game.
        
    def _place_food(self): # Randomly place food on the board.
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y) # Assign the food's position.
        if self.food in self.snake: # Ensure food does not overlap with the snake.
            self._place_food()
        
    def play_step(self, action): # Main game loop for each step.
        self.frame_iteration += 1 # Increment frame counter.
        # 1. collect user input 
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: # If the user closes the game window.
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False

        # End game if collision or max iterations exceeded.
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10 # Penalize collision or timeout.
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food: # Check if the snake ate food.
            self.score += 1
            reward = 10
            self._place_food() # Place new food.
        else:
            self.snake.pop() # Remove tail if food not eaten.
        
        # 5. update ui and clock 
        self._update_ui() # Render game visuals.
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None): # Check for collisions.
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK) # Clear the screen.
        
        for pt in self.snake: # Draw the snake.
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))  # Draw food.
        
        text = font.render("Score: " + str(self.score), True, WHITE) # Render score text.
        self.display.blit(text, [0, 0]) # Display score on screen.
        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction) # Current direction index.

        if np.array_equal(action, [1, 0, 0]): # Move straight.
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]): # Turn right.
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # Turn left [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir # Update direction.

        x = self.head.x # Update position based on new direction.
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y) # Set new head position.