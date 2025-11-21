"""
Main Entry Point
Orchestrates the entire logistic regression workflow.
"""

""" import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.model import LogisticRegression
from src.train import train
from src.eval import evaluate
from src.visualization import plot_data_distribution, plot_loss_history
from src.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, LEARNING_RATE, NUM_EPOCHS, RANDOM_SEED """
from src.dataset import load_and_process_data
from src.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, LEARNING_RATE, NUM_EPOCHS, RANDOM_SEED
from src.model import LogisticRegression
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



# Verileri yükle
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_process_data('data/hw1_data.csv', TRAIN_RATIO, RANDOM_SEED)

print(f"Eğitim Seti: {X_train.shape}")      # Örn: (60, 2)
print(f"Doğrulama Seti: {X_val.shape}")     # Örn: (20, 2)
print(f"Test Seti: {X_test.shape}")         # Örn: (20, 2)

""" for i in range(len(X_train)):
    print(X_train[i], y_train[i]) """
    
sigmoid_val = LogisticRegression.sigmoid(y_train[50])

print(sigmoid_val)