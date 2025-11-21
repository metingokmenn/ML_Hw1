"""
Visualization Module
Contains functions for plotting and visualization.
"""

import matplotlib.pyplot as plt
import numpy as np

# This module will contain:
# - plot_data_distribution(X, y) -> None
#   Plots scatter plot with exam1 vs exam2 scores, colored by class
# - plot_loss_history(train_losses, val_losses=None, save_path=None) -> None
#   Plots cross-entropy loss over epochs for training and validation
# - Other visualization helper functions


def plot_data_distribution(X,y):
    
    plt.plot(X, marker='o', color='red', linestyle='None', label='X')
    plt.plot(y, marker='^', color='blue', linestyle='None', label='Y')
    plt.legend()
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.savefig('results/data_distribution.png')
    plt.title('Data Distribution')
    
    plt.show()
    
    
X = np.array([1,2,3,4,5])
y = np.array([3,4,2,1,6])
plot_data_distribution(X,y)