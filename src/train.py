

import numpy as np
from .model import LogisticRegression


def train(model, X_train, y_train, X_val, y_val, patience=10, verbose=True, early_stopping=True):
    """
    Modeli Stochastic Gradient Descent (SGD) kullanarak eğitir ve eğitim geçmişini döndürür.
    
    Bu fonksiyon, gerçek Stochastic Gradient Descent algoritmasını implemente eder.
    Her epoch'ta veri karıştırılır ve her bir örnek için ayrı ayrı ağırlık güncellemesi
    yapılır. Her epoch sonunda hem training hem de validation loss hesaplanır ve
    kaydedilir. Overfitting kontrolü için validation loss takibi yapılır ve eğer
    validation loss artarken training loss azalıyorsa overfitting tespit edilir.
    
    Eğitim Süreci:
    --------------
    1. Model ağırlıkları rastgele küçük değerlerle başlatılır
    2. Her epoch için:
       a. Eğitim verisi karıştırılır (shuffle)
       b. Her örnek için sırayla:
          - Forward pass yapılır (tahmin hesaplanır)
          - Loss hesaplanır
          - Gradient hesaplanır
          - Ağırlıklar güncellenir (SGD)
       c. Epoch sonunda tüm eğitim seti için loss hesaplanır
       d. Validation seti için loss hesaplanır
       e. En iyi model takip edilir
       f. Overfitting kontrolü yapılır
    
    Argümanlar:
    -----------
    model : LogisticRegression
        Eğitilecek LogisticRegression model instance'ı. Modelin learning_rate ve
        num_epochs parametreleri önceden ayarlanmış olmalıdır. Eğitim sırasında
        modelin weights ve bias değerleri güncellenir.
    X_train : numpy array, shape (n_samples, n_features)
        Eğitim seti özellik matrisi. Her satır bir örneği, her sütun bir özelliği
        temsil eder. Normalize edilmiş olmalıdır.
    y_train : numpy array, shape (n_samples,)
        Eğitim seti gerçek sınıf etiketleri. Binary classification için 0 veya 1
        değerlerini içermelidir.
    X_val : numpy array, shape (n_val_samples, n_features)
        Doğrulama seti özellik matrisi. Overfitting kontrolü ve model seçimi için
        kullanılır. X_train ile aynı normalizasyon parametreleriyle normalize
        edilmiş olmalıdır.
    y_val : numpy array, shape (n_val_samples,)
        Doğrulama seti gerçek sınıf etiketleri. Binary classification için 0 veya 1
        değerlerini içermelidir.
    patience : int, default=10
        Early stopping için patience değeri. Validation loss iyileşmediğinde (arttığında
        veya aynı kaldığında), bu durumun kaç epoch boyunca devam etmesine izin verileceğini
        belirler. Patience sayısı aşılırsa eğitim durdurulur (early_stopping=True ise).
    verbose : bool, default=True
        Eğitim sırasında bilgi mesajlarının konsola yazdırılıp yazdırılmayacağını
        kontrol eder. True ise her 100 epoch'ta bir loss değerleri yazdırılır ve
        eğitim sonunda özet bilgiler gösterilir.
    early_stopping : bool, default=True
        Early stopping aktif edilsin mi? True ise, validation loss patience epoch
        boyunca iyileşmezse eğitim durdurulur ve en iyi model kullanılır.
    
    Returns:
    --------
    dict
        Eğitim geçmişi içeren dictionary. Anahtarlar:
        - 'train_losses': list of float, her epoch için training loss değerleri
        - 'val_losses': list of float, her epoch için validation loss değerleri
        - 'best_val_loss': float, eğitim sırasında elde edilen en düşük validation loss değeri
        - 'best_epoch': int, en iyi validation loss'un elde edildiği epoch numarası (0-indexed)
        - 'overfitting_detected': bool, eğitim sırasında overfitting tespit edilip edilmediği
        - 'early_stopped': bool, early stopping ile eğitimin durdurulup durdurulmadığı
    """
    
    model._initialize_weights(X_train.shape[1])
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    overfitting_detected = False
    early_stopped = False
    
    
    best_weights = None
    best_bias = None
    
    n_samples = len(X_train)
    
    for epoch in range(model.num_epochs):
        
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        
        for i in range(n_samples):
            x_sample = X_train_shuffled[i]
            y_sample = y_train_shuffled[i]
            
            y_pred_sample = model.forward(x_sample.reshape(1, -1))[0]
            
            model.update_weights(x_sample, y_sample, y_pred_sample)
        
        
        y_train_pred = model.forward(X_train)
        train_loss = model.compute_loss(y_train, y_train_pred)
        train_losses.append(train_loss)
        
        y_val_pred = model.forward(X_val)
        val_loss = model.compute_loss(y_val, y_val_pred)
        val_losses.append(val_loss)
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            best_weights = model.weights.copy()
            best_bias = model.bias
        else:
            patience_counter += 1
        
        
        if epoch > 0:
            if val_losses[-1] > val_losses[-2] and train_losses[-1] < train_losses[-2]:
                if patience_counter >= patience:
                    overfitting_detected = True
                    if verbose:
                        print(f"\nOverfitting tespit edildi! Epoch {epoch + 1}")
                        print(f"   Validation loss artıyor, training loss azalıyor.")
                        print(f"   Patience: {patience_counter}/{patience}")
        
        
        if early_stopping and patience_counter >= patience:
            early_stopped = True
            if verbose:
                print(f"\nEarly stopping: Epoch {epoch + 1}")
                print(f"   Validation loss {patience} epoch boyunca iyileşmedi.")
                print(f"   En iyi model: Epoch {best_epoch + 1} (Val Loss: {best_val_loss:.4f})")
                print(f"   En iyi model ağırlıklarına geri dönülüyor...")
            
            model.weights = best_weights.copy()
            model.bias = best_bias
            break
        
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{model.num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if verbose:
        print(f"\nEğitim tamamlandı!")
        if early_stopped:
            print(f"   Early stopping ile durduruldu.")
        print(f"   En iyi validation loss: {best_val_loss:.4f} (Epoch {best_epoch + 1})")
        if overfitting_detected:
            print(f"   Overfitting tespit edildi: Validation loss artışı gözlemlendi.")
        if not early_stopped and overfitting_detected:
            print(f"   Öneri: Early stopping aktif edilirse overfitting önlenebilir.")
    
    
    if not early_stopped and best_weights is not None:
        model.weights = best_weights.copy()
        model.bias = best_bias
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'overfitting_detected': overfitting_detected,
        'early_stopped': early_stopped
    }
    
    return history
