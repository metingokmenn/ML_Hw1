"""
Main Entry Point
Orchestrates the entire logistic regression workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.model import LogisticRegression
from src.train import train
from src.eval import evaluate
from src.visualization import plot_data_distribution, plot_loss_history
from src.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, LEARNING_RATE, NUM_EPOCHS, RANDOM_SEED
from src.dataset import load_data, split_data

# This script will:
# 1. Import necessary modules
# 2. Load and split data using dataset module
# 3. Visualize data distribution using visualization module
# 4. Initialize model using model module
# 5. Train model using train module
# 6. Evaluate on all sets using eval module
# 7. Generate and save all plots and results


def main():
    # 1. Load and split data using dataset module
    data = load_data()
    train_data, val_data, test_data = split_data(data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    # 2. Visualize data distribution using visualization module
    plot_data_distribution(train_data[:, 0], train_data[:, 1])
    
    # 3. Initialize model using model module



main()