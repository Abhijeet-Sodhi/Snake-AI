import matplotlib.pyplot as plt
from IPython import display

plt.ion() # Enable interactive mode for live updates

def plot(scores, mean_scores):
    display.clear_output(wait=True) # Clear previous output in the notebook
    display.display(plt.gcf()) # Display the current figure
    plt.clf() # Clear the current figure
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores) 
    plt.plot(mean_scores) 
    plt.ylim(ymin=0) 
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))  # Display the latest score
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1])) # Display the latest mean score
    plt.show(block=False) # Display the updated plot without blocking execution
    plt.pause(.1) # Pause briefly to allow the plot to update