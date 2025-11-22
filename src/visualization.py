import matplotlib.pyplot as plt
import numpy as np
import os


def plot_data_distribution(X, y, save_path='results/initial_data_dist.png'):
    """
    Veri dağılımını scatter plot olarak görselleştirir ve dosyaya kaydeder.
    
    Bu fonksiyon, iki boyutlu özellik uzayında veri noktalarını görselleştirir.
    Her sınıf farklı renk ve marker ile gösterilir. Grafik, veri setinin
    genel yapısını ve sınıfların ayrılabilirliğini anlamak için kullanılır.
    
    Argümanlar:
    -----------
    X : numpy array, shape (n_samples, 2)
        Özellik matrisi. İlk sütun Exam1 skorlarını, ikinci sütun Exam2 skorlarını
        içermelidir. Her satır bir örneği temsil eder.
    y : numpy array, shape (n_samples,)
        Sınıf etiketleri. Binary classification için 0 (Red/Reject) veya 1
        (Accept) değerlerini içermelidir. Her değer X'in ilgili satırına karşılık gelir.
    save_path : str, default='results/initial_data_dist.png'
        Grafiğin kaydedileceği dosya yolu. Eğer klasör yoksa otomatik olarak
        oluşturulur. Dosya formatı PNG olarak kaydedilir (300 DPI çözünürlükte).
    
    Returns:
    --------
    Yok (fonksiyon grafiği dosyaya kaydeder ve konsola bilgi mesajı yazdırır)
    
    Görselleştirme Detayları:
    -------------------------
    - X ekseni: Exam1 skoru
    - Y ekseni: Exam2 skoru
    - Sınıf 0 (Reject): Kırmızı renk, yuvarlak marker (o)
    - Sınıf 1 (Accept): Mavi renk, üçgen marker (^)
    - Grid çizgileri ve legend eklidir
    """
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    
    class_0_indices = y == 0
    class_1_indices = y == 1
    
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[class_0_indices, 0], X[class_0_indices, 1], 
                c='red', marker='o', label='Reject (0)', alpha=0.6, s=50)
    plt.scatter(X[class_1_indices, 0], X[class_1_indices, 1], 
                c='blue', marker='^', label='Accept (1)', alpha=0.6, s=50)
    
    plt.xlabel('Exam 1 Score', fontsize=12)
    plt.ylabel('Exam 2 Score', fontsize=12)
    plt.title('Data Distribution: Exam Scores by Class', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Veri dağılım grafiği kaydedildi: {save_path}")
    plt.close()


def plot_loss_history(train_losses, val_losses=None, save_path='results/loss_curve.png'):
    """
    Epoch'lara göre cross-entropy loss değerlerini görselleştirir ve dosyaya kaydeder.
    
    Bu fonksiyon, model eğitimi sırasında hesaplanan loss değerlerini zaman serisi
    grafiği olarak gösterir. Training ve validation loss'ların birlikte görselleştirilmesi,
    overfitting tespiti ve model performansı değerlendirmesi için önemlidir.
    
    Argümanlar:
    -----------
    train_losses : list veya numpy array, shape (n_epochs,)
        Her epoch için hesaplanan training loss değerleri. Liste veya numpy array
        formatında olabilir. Her eleman bir epoch'taki ortalama cross-entropy loss
        değerini temsil eder.
    val_losses : list veya numpy array, shape (n_epochs,), optional
        Her epoch için hesaplanan validation loss değerleri. Eğer None ise sadece
        training loss gösterilir. Varsa, training loss ile birlikte aynı grafikte
        gösterilir. train_losses ile aynı uzunlukta olmalıdır.
    save_path : str, default='results/loss_curve.png'
        Grafiğin kaydedileceği dosya yolu. Eğer klasör yoksa otomatik olarak
        oluşturulur. Dosya formatı PNG olarak kaydedilir (300 DPI çözünürlükte).
    
    Returns:
    --------
    Yok (fonksiyon grafiği dosyaya kaydeder ve konsola bilgi mesajı yazdırır)
    
    Görselleştirme Detayları:
    -------------------------
    - X ekseni: Epoch numarası (1'den başlar)
    - Y ekseni: Cross-entropy loss değeri
    - Training loss: Mavi çizgi
    - Validation loss: Kırmızı çizgi (varsa)
    - Grid çizgileri ve legend eklidir
    - Grafik başlığı: "Training and Validation Loss Over Epochs"
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss eğrisi grafiği kaydedildi: {save_path}")
    plt.close()


def plot_test_predictions(X_test, y_test, y_pred, save_path='results/test_predictions.png'):
    """
    Test setindeki tahminlerin doğruluğunu görselleştirir ve dosyaya kaydeder.
    
    Bu fonksiyon, test setindeki her örneğin gerçek sınıfını ve tahmin edilen sınıfını
    karşılaştırarak doğru ve yanlış tahminleri görselleştirir. Doğru tahminler yeşil,
    yanlış tahminler kırmızı renkle gösterilir.
    
    Argümanlar:
    -----------
    X_test : numpy array, shape (n_samples, 2)
        Test seti özellik matrisi. İlk sütun Exam1 skorlarını, ikinci sütun Exam2 skorlarını
        içermelidir. Her satır bir örneği temsil eder.
    y_test : numpy array, shape (n_samples,)
        Test seti gerçek sınıf etiketleri. Binary classification için 0 (Reject) veya 1
        (Accept) değerlerini içermelidir.
    y_pred : numpy array, shape (n_samples,)
        Model tarafından tahmin edilen sınıf etiketleri. Binary classification için 0 veya 1
        değerlerini içermelidir. y_test ile aynı uzunlukta olmalıdır.
    save_path : str, default='results/test_predictions.png'
        Grafiğin kaydedileceği dosya yolu. Eğer klasör yoksa otomatik olarak
        oluşturulur. Dosya formatı PNG olarak kaydedilir (300 DPI çözünürlükte).
    
    Returns:
    --------
    Yok (fonksiyon grafiği dosyaya kaydeder ve konsola bilgi mesajı yazdırır)
    
    Görselleştirme Detayları:
    -------------------------
    - X ekseni: Exam1 skoru
    - Y ekseni: Exam2 skoru
    - Doğru tahminler: Yeşil renk (✓)
    - Yanlış tahminler: Kırmızı renk (✗)
    - Marker şekli: Gerçek sınıfı gösterir (o: Reject, ^: Accept)
    - Border: Tahmin edilen sınıfı gösterir (siyah border: Accept, beyaz border: Reject)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    correct_predictions = y_test == y_pred
    incorrect_predictions = ~correct_predictions
    
    correct_class_0 = correct_predictions & (y_test == 0)
    correct_class_1 = correct_predictions & (y_test == 1)
    
    incorrect_class_0 = incorrect_predictions & (y_test == 0)
    incorrect_class_1 = incorrect_predictions & (y_test == 1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if np.any(correct_class_0):
        ax.scatter(X_test[correct_class_0, 0], X_test[correct_class_0, 1],
                  c='green', marker='o', s=100, alpha=0.7, 
                  edgecolors='darkgreen', linewidths=2,
                  label=f'Doğru - Reject (0) [{np.sum(correct_class_0)}]')
    
    if np.any(correct_class_1):
        ax.scatter(X_test[correct_class_1, 0], X_test[correct_class_1, 1],
                  c='green', marker='^', s=100, alpha=0.7,
                  edgecolors='darkgreen', linewidths=2,
                  label=f'Doğru - Accept (1) [{np.sum(correct_class_1)}]')
    
    if np.any(incorrect_class_0):
        ax.scatter(X_test[incorrect_class_0, 0], X_test[incorrect_class_0, 1],
                  c='red', marker='o', s=150, alpha=0.8,
                  edgecolors='darkred', linewidths=3,
                  label=f'Yanlış - Reject (0) [{np.sum(incorrect_class_0)}]')
    
    if np.any(incorrect_class_1):
        ax.scatter(X_test[incorrect_class_1, 0], X_test[incorrect_class_1, 1],
                  c='red', marker='^', s=150, alpha=0.8,
                  edgecolors='darkred', linewidths=3,
                  label=f'Yanlış - Accept (1) [{np.sum(incorrect_class_1)}]')
    
    accuracy = np.mean(correct_predictions) * 100
    n_correct = np.sum(correct_predictions)
    n_total = len(y_test)
    
    ax.set_xlabel('Exam 1 Score', fontsize=12)
    ax.set_ylabel('Exam 2 Score', fontsize=12)
    ax.set_title(f'Test Seti Tahmin Sonuçları\n'
                f'Doğruluk: {accuracy:.2f}% ({n_correct}/{n_total})', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Test tahmin grafiği kaydedildi: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """
    Confusion matrix'i görselleştirir ve dosyaya kaydeder.
    
    Bu fonksiyon, binary classification için confusion matrix'i heatmap olarak gösterir.
    TP, TN, FP, FN değerlerini renkli bir matris olarak görselleştirir.
    
    Argümanlar:
    -----------
    y_true : numpy array, shape (n_samples,)
        Gerçek sınıf etiketleri. Binary classification için 0 veya 1 değerlerini içermelidir.
    y_pred : numpy array, shape (n_samples,)
        Tahmin edilen sınıf etiketleri. Binary classification için 0 veya 1 değerlerini içermelidir.
    save_path : str, default='results/confusion_matrix.png'
        Grafiğin kaydedileceği dosya yolu. Eğer klasör yoksa otomatik olarak oluşturulur.
    
    Returns:
    --------
    Yok (fonksiyon grafiği dosyaya kaydeder ve konsola bilgi mesajı yazdırır)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    cm = np.array([[tn, fp],
                   [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    
    plt.colorbar(im, ax=ax)
    
    classes = ['Reject (0)', 'Accept (1)']
    tick_marks = np.arange(len(classes))
    
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Tahmin: ' + c for c in classes])
    ax.set_yticklabels(['Gerçek: ' + c for c in classes])
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i - 0.15, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')
    
    ax.text(0, 0 + 0.15, 'TN', ha="center", va="center", 
           color="white" if cm[0, 0] > thresh else "black",
           fontsize=11, fontweight='bold')
    ax.text(1, 0 + 0.15, 'FP', ha="center", va="center", 
           color="white" if cm[0, 1] > thresh else "black",
           fontsize=11, fontweight='bold')
    ax.text(0, 1 + 0.15, 'FN', ha="center", va="center", 
           color="white" if cm[1, 0] > thresh else "black",
           fontsize=11, fontweight='bold')
    ax.text(1, 1 + 0.15, 'TP', ha="center", va="center", 
           color="white" if cm[1, 1] > thresh else "black",
           fontsize=11, fontweight='bold')
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    ax.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}%', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix grafiği kaydedildi: {save_path}")
    plt.close()
