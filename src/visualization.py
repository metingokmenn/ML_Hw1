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
                c='red', marker='o', label='Reddet (0)', alpha=0.6, s=50)
    plt.scatter(X[class_1_indices, 0], X[class_1_indices, 1], 
                c='blue', marker='^', label='Kabul (1)', alpha=0.6, s=50)
    
    plt.xlabel('Sınav 1 Skoru', fontsize=12)
    plt.ylabel('Sınav 2 Skoru', fontsize=12)
    plt.title('Veri Dağılımı: Sınıflara Göre Sınav Skorları', fontsize=14, fontweight='bold')
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
    plt.plot(epochs, train_losses, 'b-', label='Eğitim Kaybı', linewidth=2, alpha=0.8)
    
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Doğrulama Kaybı', linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Çapraz Entropi Kaybı', fontsize=12)
    plt.title('Epochlar Boyunca Eğitim ve Doğrulama Kaybı', fontsize=14, fontweight='bold')
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
                  label=f'Doğru - Reddet (0) [{np.sum(correct_class_0)}]')
    
    if np.any(correct_class_1):
        ax.scatter(X_test[correct_class_1, 0], X_test[correct_class_1, 1],
                  c='green', marker='^', s=100, alpha=0.7,
                  edgecolors='darkgreen', linewidths=2,
                  label=f'Doğru - Kabul (1) [{np.sum(correct_class_1)}]')
    
    if np.any(incorrect_class_0):
        ax.scatter(X_test[incorrect_class_0, 0], X_test[incorrect_class_0, 1],
                  c='red', marker='o', s=150, alpha=0.8,
                  edgecolors='darkred', linewidths=3,
                  label=f'Yanlış - Reddet (0) [{np.sum(incorrect_class_0)}]')
    
    if np.any(incorrect_class_1):
        ax.scatter(X_test[incorrect_class_1, 0], X_test[incorrect_class_1, 1],
                  c='red', marker='^', s=150, alpha=0.8,
                  edgecolors='darkred', linewidths=3,
                  label=f'Yanlış - Kabul (1) [{np.sum(incorrect_class_1)}]')
    
    accuracy = np.mean(correct_predictions) * 100
    n_correct = np.sum(correct_predictions)
    n_total = len(y_test)
    
    ax.set_xlabel('Sınav 1 Skoru', fontsize=12)
    ax.set_ylabel('Sınav 2 Skoru', fontsize=12)
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
    
    classes = ['Reddet (0)', 'Kabul (1)']
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
    ax.set_title(f'Karışıklık Matrisi\nDoğruluk: {accuracy:.2f}%', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix grafiği kaydedildi: {save_path}")
    plt.close()


def plot_decision_boundary(model, X, y, save_path='results/decision_boundary.png'):
    """
    Modelin karar sınırını (decision boundary) görselleştirir ve dosyaya kaydeder.
    
    Bu fonksiyon, eğitilmiş logistic regression modelinin karar sınırını görselleştirir.
    Karar sınırı, modelin sınıfları ayırdığı çizgidir (probability = 0.5). Ayrıca
    olasılık konturları (probability contours) gösterilerek modelin tahmin güvenini
    görselleştirir. Modelin ağırlıkları ve bias değeri kullanılarak karar sınırının
    denklemi hesaplanır ve grafik üzerinde gösterilir.
    
    Karar Sınırı Denklemi:
    ----------------------
    Logistic regression için karar sınırı, z = w1*x1 + w2*x2 + b = 0 denkleminden
    elde edilir. Bu, sigmoid(z) = 0.5 olduğu noktadır.
    
    Denklem: w1*x1 + w2*x2 + b = 0
    Çözüm: x2 = -(w1*x1 + b) / w2
    
    Argümanlar:
    -----------
    model : LogisticRegression
        Eğitilmiş LogisticRegression model instance'ı. Modelin weights ve bias
        değerleri set edilmiş olmalıdır.
    X : numpy array, shape (n_samples, 2)
        Özellik matrisi. İlk sütun Exam1 skorlarını, ikinci sütun Exam2 skorlarını
        içermelidir. Her satır bir örneği temsil eder. Normalize edilmiş olabilir.
    y : numpy array, shape (n_samples,)
        Sınıf etiketleri. Binary classification için 0 (Reject) veya 1 (Accept)
        değerlerini içermelidir.
    save_path : str, default='results/decision_boundary.png'
        Grafiğin kaydedileceği dosya yolu. Eğer klasör yoksa otomatik olarak
        oluşturulur. Dosya formatı PNG olarak kaydedilir (300 DPI çözünürlükte).
    
    Returns:
    --------
    str
        Karar sınırının denklemini string olarak döndürür.
    
    Görselleştirme Detayları:
    -------------------------
    - X ekseni: Exam1 skoru (normalize edilmiş)
    - Y ekseni: Exam2 skoru (normalize edilmiş)
    - Sınıf 0 (Reject): Kırmızı renk, yuvarlak marker (o)
    - Sınıf 1 (Accept): Mavi renk, üçgen marker (^)
    - Karar sınırı: Siyah kalın çizgi
    - Olasılık konturları: Renkli konturlar (kırmızı: Reject bölgesi, mavi: Accept bölgesi)
    - Denklem: Grafik üzerinde gösterilir
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Model ağırlıklarını ve bias'ı al
    weights = model.weights
    bias = model.bias
    
    # Veri aralığını belirle (biraz padding ile)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Meshgrid oluştur (olasılık konturları için)
    h = 0.02  # Grid çözünürlüğü
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Meshgrid üzerinde tahmin yap
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Sınıf tahminleri (0.5 threshold ile)
    Z_pred = (Z >= 0.5).astype(int)
    
    # Grafik oluştur
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Olasılık konturlarını çiz
    contour = ax.contourf(xx, yy, Z, levels=20, alpha=0.4, cmap='RdYlBu_r')
    plt.colorbar(contour, ax=ax, label='Tahmin Edilen Olasılık (Kabul)')
    
    # Karar sınırını çiz (probability = 0.5)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3, linestyles='solid')
    
    # Veri noktalarını çiz
    class_0_indices = y == 0
    class_1_indices = y == 1
    
    ax.scatter(X[class_0_indices, 0], X[class_0_indices, 1],
              c='red', marker='o', label='Reddet (0)', s=80, alpha=0.8,
              edgecolors='darkred', linewidths=1.5, zorder=3)
    ax.scatter(X[class_1_indices, 0], X[class_1_indices, 1],
              c='blue', marker='^', label='Kabul (1)', s=80, alpha=0.8,
              edgecolors='darkblue', linewidths=1.5, zorder=3)
    
    # Karar sınırı çizgisini çiz (daha belirgin)
    # z = w1*x1 + w2*x2 + b = 0
    # x2 = -(w1*x1 + b) / w2
    if abs(weights[1]) > 1e-10:  # w2 sıfıra yakın değilse
        x_boundary = np.linspace(x_min, x_max, 100)
        y_boundary = -(weights[0] * x_boundary + bias) / weights[1]
        ax.plot(x_boundary, y_boundary, 'k-', linewidth=3, 
               label='Karar Sınırı', zorder=2)
    
    # Denklemi oluştur ve göster
    w1, w2 = weights[0], weights[1]
    
    # Denklemi formatla
    if abs(w1) < 1e-6:
        w1_str = "0"
    elif abs(w1 - 1) < 1e-6:
        w1_str = "x₁"
    elif abs(w1 + 1) < 1e-6:
        w1_str = "-x₁"
    else:
        w1_str = f"{w1:.4f}·x₁"
    
    if abs(w2) < 1e-6:
        w2_str = "0"
    elif abs(w2 - 1) < 1e-6:
        w2_str = "x₂"
    elif abs(w2 + 1) < 1e-6:
        w2_str = "-x₂"
    else:
        w2_str = f"{w2:.4f}·x₂"
    
    if abs(bias) < 1e-6:
        bias_str = ""
    elif bias > 0:
        bias_str = f" + {bias:.4f}"
    else:
        bias_str = f" - {abs(bias):.4f}"
    
    equation = f"{w1_str} + {w2_str}{bias_str} = 0"
    
    # Daha okunabilir formatta göster
    equation_text = f"Karar Sınırı:\n{w1:.4f}·x₁ + {w2:.4f}·x₂ + {bias:.4f} = 0"
    
    # Alternatif form: x2 = ...
    if abs(weights[1]) > 1e-10:
        slope = -weights[0] / weights[1]
        intercept = -bias / weights[1]
        equation_text += f"\n\nAlternatif form:\nx₂ = {slope:.4f}·x₁ + {intercept:.4f}"
    
    # Denklemi grafik üzerine ekle
    ax.text(0.02, 0.98, equation_text,
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           family='monospace')
    
    # Model parametrelerini göster
    params_text = f"Model Parametreleri:\nw₁ = {w1:.6f}\nw₂ = {w2:.6f}\nb = {bias:.6f}"
    ax.text(0.98, 0.02, params_text,
           transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
           family='monospace')
    
    ax.set_xlabel('Sınav 1 Skoru (Normalize)', fontsize=12)
    ax.set_ylabel('Sınav 2 Skoru (Normalize)', fontsize=12)
    ax.set_title('Karar Sınırı Görselleştirmesi\nLojistik Regresyon Modeli', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Karar sınırı grafiği kaydedildi: {save_path}")
    print(f"Karar sınırı denklemi:\n{equation_text}")
    plt.close()
    
    return equation_text


def plot_metrics_comparison(results, save_path='results/metrics_comparison.png'):
    """
    Train, Validation ve Test setleri için Accuracy, Precision, Recall ve F1-Score
    değerlerini karşılaştırmalı bar chart olarak görselleştirir ve dosyaya kaydeder.
    
    Bu fonksiyon, modelin farklı veri setleri üzerindeki performansını görsel olarak
    karşılaştırmak için kullanılır. Her metrik için üç setin değerleri yan yana
    gösterilir.
    
    Argümanlar:
    -----------
    results : dict
        evaluate_all() fonksiyonunun döndürdüğü dictionary. Yapı:
        {
            'train': dict,      # accuracy, precision, recall, f1_score içerir
            'validation': dict, # accuracy, precision, recall, f1_score içerir
            'test': dict        # accuracy, precision, recall, f1_score içerir
        }
    save_path : str, default='results/metrics_comparison.png'
        Grafiğin kaydedileceği dosya yolu. Eğer klasör yoksa otomatik olarak
        oluşturulur. Dosya formatı PNG olarak kaydedilir (300 DPI çözünürlükte).
    
    Returns:
    --------
    Yok (fonksiyon grafiği dosyaya kaydeder ve konsola bilgi mesajı yazdırır)
    
    Görselleştirme Detayları:
    -------------------------
    - X ekseni: Metrikler (Accuracy, Precision, Recall, F1-Score)
    - Y ekseni: Metrik değerleri (0-1 arası)
    - Her metrik için üç bar: Eğitim (mavi), Doğrulama (turuncu), Test (yeşil)
    - Legend ile setler gösterilir
    - Değerler bar'ların üzerinde gösterilir
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Metrikleri çıkar
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Doğruluk\n(Accuracy)', 'Kesinlik\n(Precision)', 
                     'Duyarlılık\n(Recall)', 'F1-Skoru\n(F1-Score)']
    
    train_values = [results['train'][m] for m in metrics]
    val_values = [results['validation'][m] for m in metrics]
    test_values = [results['test'][m] for m in metrics]
    
    # Grafik oluştur
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(metric_labels))
    width = 0.25  # Bar genişliği
    
    # Bar'ları çiz
    bars1 = ax.bar(x - width, train_values, width, label='Eğitim Seti', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, val_values, width, label='Doğrulama Seti', 
                   color='#e67e22', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + width, test_values, width, label='Test Seti', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Değerleri bar'ların üzerine yaz
    def autolabel(bars):
        """Bar'ların üzerine değerleri yazdırır."""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    # Eksenleri ayarla
    ax.set_xlabel('Metrikler', fontsize=12, fontweight='bold')
    ax.set_ylabel('Değer', fontsize=12, fontweight='bold')
    ax.set_title('Model Performans Metrikleri Karşılaştırması\n'
                'Eğitim, Doğrulama ve Test Setleri', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim([0, 1.15])  # Üstte biraz boşluk bırak
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Mükemmel Skor')
    
    # Y eksenini 0-1 arası göster
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrik karşılaştırma grafiği kaydedildi: {save_path}")
    plt.close()
