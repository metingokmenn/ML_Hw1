
import numpy as np


def load_and_process_data(filepath, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """
    Veriyi CSV dosyasından yükler, karıştırır ve train/validation/test setlerine ayırır.
    
    Argümanlar:
    -----------
    filepath : str
        CSV dosyasının yolu. Dosya formatı: Exam1, Exam2, Label
    train_ratio : float, default=0.6
        Eğitim seti için kullanılacak veri oranı (0.6 = %60)
    val_ratio : float, default=0.2
        Doğrulama seti için kullanılacak veri oranı (0.2 = %20)
    test_ratio : float, default=0.2
        Test seti için kullanılacak veri oranı (0.2 = %20)
    random_state : int, default=42
        Rastgele sayı üreteci için seed değeri (tekrarlanabilirlik için)
    
    Returns:
    --------
    tuple : ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        Normalize edilmiş özellikler ve etiketler içeren tuple'lar
        X_train, X_val, X_test: numpy array, shape (n_samples, 2)
        y_train, y_val, y_test: numpy array, shape (n_samples,)
    """
    
    raw_data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    
    X = raw_data[:, 0:2]  
    y = raw_data[:, 2]    
    
    
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    
    
    train_std = train_std + 1e-8
    
    
    X_train_scaled = (X_train - train_mean) / train_std
    X_val_scaled = (X_val - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std
    
    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)
