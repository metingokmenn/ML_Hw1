
import numpy as np


class LogisticRegression:
    """
    
    Bu sınıf, sigmoid aktivasyon fonksiyonu ve cross-entropy loss kullanarak
    binary classification yapar. Stochastic Gradient Descent (SGD) ile hata düşürme yapılır.
    """
    
    def __init__(self, learning_rate=0.01, num_epochs=1000):
        """
        Model parametrelerini başlatır.
        
        Argümanlar:
        -----------
        learning_rate : float, default=0.01
            Öğrenme oranı (gradient descent için)
        num_epochs : int, default=1000
            Maksimum epoch sayısı
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
        
    def _initialize_weights(self, n_features):
        """
        Ağırlıkları ve bias'ı rastgele küçük değerlerle başlatır.
        
        Argümanlar:
        -----------
        n_features : int
            Özellik sayısı (bu projede 2: Exam1 ve Exam2)
        """
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    
    @staticmethod
    def sigmoid(z):
        """
        Sigmoid aktivasyon fonksiyonu.
        
        Sigmoid fonksiyonu, "S" şeklinde bir eğri üreten matematiksel bir fonksiyondur.
        Formül: f(z) = 1 / (1 + exp(-z))
        
        Argümanlar:
        -----------
        z : float veya numpy array
            Sigmoid fonksiyonunun girdisi (linear combination)
        
        Returns:
        --------
        float veya numpy array
            Sigmoid fonksiyonunun çıktısı (0-1 arasında bir değer)
        """
        # Numerik kararlılık için: z çok büyükse exp(-z) 0'a yakın olur
        z = np.clip(z, -500, 500)  # Overflow önleme
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        """
        Forward pass: Linear kombinasyon ve sigmoid aktivasyon.
        
        Argümanlar:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Girdi özellikleri
        
        Returns:
        --------
        numpy array, shape (n_samples,)
            Tahmin edilen olasılıklar (0-1 arası)
        """
        if self.weights is None:
            self._initialize_weights(X.shape[1])
        
        
        z = np.dot(X, self.weights) + self.bias
        
        
        y_pred = self.sigmoid(z)
        
        return y_pred
    
    def predict(self, X, threshold=0.5):
        """
        Binary tahminler yapar (0 veya 1).
        
        Argümanlar:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Girdi özellikleri
        threshold : float, default=0.5
            Sınıflandırma eşiği (0.5'ten büyükse 1, değilse 0)
        
        Returns:
        --------
        numpy array, shape (n_samples,)
            Binary tahminler (0 veya 1)
        """
        y_pred_proba = self.forward(X)
        return (y_pred_proba >= threshold).astype(int)
    
    @staticmethod
    def compute_loss(y_true, y_pred):
        """
        Cross-entropy loss fonksiyonu.
        
        Formül: L(y_true, y_pred) = -[y_true * log(y_pred) + (1-y_true) * log(1-y_pred)]
        
        Argümanlar:
        -----------
        y_true : numpy array, shape (n_samples,)
            Gerçek etiketler (0 veya 1)
        y_pred : numpy array, shape (n_samples,)
            Tahmin edilen olasılıklar (0-1 arası)
        
        Returns:
        --------
        float
            Ortalama cross-entropy loss değeri
        """
        
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        
        return np.mean(loss)
    
    def update_weights(self, x, y_true, y_pred):
        """
        Stochastic Gradient Descent (SGD) kullanarak ağırlıkları günceller.
        
        Her bir örnek için ayrı ayrı gradient hesaplanır ve ağırlıklar güncellenir.
        
        Gradient formülleri:
        - dL/dw = (y_pred - y_true) * x
        - dL/db = (y_pred - y_true)
        
        Argümanlar:
        -----------
        x : numpy array, shape (n_features,)
            Tek bir örneğin özellikleri
        y_true : float
            Gerçek etiket (0 veya 1)
        y_pred : float
            Tahmin edilen olasılık (0-1 arası)
        """
        
        error = y_pred - y_true
        
        
        self.weights -= self.learning_rate * error * x
        
        
        self.bias -= self.learning_rate * error
