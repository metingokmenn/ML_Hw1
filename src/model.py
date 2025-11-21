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
#   - cross_entropy_loss() method (loss calculation)
#   - update_weights() method (SGD weight updates)


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
        
    
    def sigmoid(z):
    '''
    
    Sigmoid fonksiyonu:
    -------------------
    Sigmoid fonksiyonu, "S" şeklinde bir eğri üreten matematiksel bir fonksiyondur.
    Şu şekilde tanımlanır:
    f(z) = 1 / (1 + exp(-z))
    burada z, fonksiyona gelen girdidir.
    Sigmoid fonksiyonu lojistik regresyonda, özelliklerin doğrusal kombinasyonunu olasılığa dönüştürmek için kullanılır.

    '''
    
    return 1 / (1 + np.exp(-z))

