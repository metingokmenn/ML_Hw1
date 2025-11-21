"""
Logistic Regression Model Module
Contains the LogisticRegression class with core model functionality.
"""

# This module will contain:
# - LogisticRegression class with:
#   - __init__() method for initialization
#   - sigmoid() method (activation function) +
#   - forward() method (forward pass)
#   - predict() method (binary predictions)
#   - cross_entropy_loss() method (loss calculation)  +
#   - update_weights() method (SGD weight updates)

import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
        
    
    def sigmoid(z):
        """
        
        Sigmoid fonksiyonu:
        -------------------
        Sigmoid fonksiyonu, "S" şeklinde bir eğri üreten matematiksel bir fonksiyondur.
        Şu şekilde tanımlanır:
        f(z) = 1 / (1 + exp(-z))
        Argümanlar:
        z: float -> Sigmoid fonksiyonunun girdisi.
        Returns:
        float -> Sigmoid fonksiyonunun çıktısı. (0-1 arasında bir değer)

        """
    
        return 1 / (1 + np.exp(-z))

    def cross_entropy_loss(y_target, y_predicted):
        
        """
        Cross-entropy loss fonksiyonu:
        ------------------------------
        Şu şekilde tanımlanır:
        L(y_target, y_predicted) = - (y_target * log(y_predicted) + (1-y_target) * log(1-y_predicted))
        Argümanlar:
        y_target: float -> Gerçek değer.
        y_predicted: float -> Tahmin edilen değer.
        Returns:
        float -> Cross-entropy loss fonksiyonunun çıktısı.
        """
        return  -1 * (y_target * np.log(y_predicted) + (1-y_target) * np.log(1-y_predicted))

