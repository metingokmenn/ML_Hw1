# Source Code Structure

This directory contains the modular implementation of the logistic regression project.

## Module Descriptions

### `dataset.py`
Handles all data-related operations:
- Loading data from CSV files
- Data preprocessing
- Splitting data into train/validation/test sets (60/20/20)

### `model.py`
Contains the core LogisticRegression model class:
- Sigmoid activation function
- Forward pass
- Prediction method
- Cross-entropy loss calculation
- Weight update logic (SGD)

### `train.py`
Implements the training loop:
- Training function with SGD
- Loss tracking
- Training history management

### `eval.py`
Handles model evaluation:
- Evaluation on datasets
- Metric calculations
- Results aggregation

### `metrics.py`
Contains metric calculation functions:
- Accuracy
- Precision
- Recall
- F-Score

### `visualization.py`
Plotting and visualization functions:
- Data distribution plots
- Loss curve plots
- Result visualizations

### `config.py`
Project configuration and hyperparameters:
- Data split ratios
- Learning rate
- Number of epochs
- Random seed

### `utils.py`
General utility functions used across modules.

