
import numpy as np


def calculate_accuracy(y_true, y_pred):
    """
    Doğruluk (Accuracy) metriğini hesaplar.
    
    Accuracy, tüm tahminler içinde doğru tahminlerin oranını gösterir. Binary
    classification için en temel performans metriğidir. Ancak dengesiz veri
    setlerinde yanıltıcı olabilir.
    
    Formül: Accuracy = (True Positive + True Negative) / Total Samples
    
    Argümanlar:
    -----------
    y_true : numpy array, shape (n_samples,)
        Gerçek sınıf etiketleri. Binary classification için 0 veya 1 değerlerini
        içermelidir. Her eleman bir örneğin gerçek sınıfını temsil eder.
    y_pred : numpy array, shape (n_samples,)
        Model tarafından tahmin edilen sınıf etiketleri. Binary classification için
        0 veya 1 değerlerini içermelidir. y_true ile aynı uzunlukta olmalıdır.
    
    Returns:
    --------
    float
        Doğruluk değeri. 0 ile 1 arasında bir değer döner:
        - 1.0: Tüm tahminler doğru
        - 0.0: Hiçbir tahmin doğru değil
        - 0.5: Rastgele tahmin seviyesi
    """
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    return correct_predictions / total_samples


def calculate_precision(y_true, y_pred):
    """
    Kesinlik (Precision) metriğini hesaplar.
    
    Precision, pozitif olarak tahmin edilen örnekler içinde gerçekten pozitif olanların
    oranını gösterir. Modelin pozitif tahminlerinin ne kadar güvenilir olduğunu ölçer.
    Yüksek precision, modelin pozitif sınıfı tahmin ederken daha az yanlış pozitif
    ürettiği anlamına gelir.
    
    Formül: Precision = True Positive / (True Positive + False Positive)
    
    Argümanlar:
    -----------
    y_true : numpy array, shape (n_samples,)
        Gerçek sınıf etiketleri. Binary classification için 0 veya 1 değerlerini
        içermelidir. Her eleman bir örneğin gerçek sınıfını temsil eder.
    y_pred : numpy array, shape (n_samples,)
        Model tarafından tahmin edilen sınıf etiketleri. Binary classification için
        0 veya 1 değerlerini içermelidir. y_true ile aynı uzunlukta olmalıdır.
    
    Returns:
    --------
    float
        Kesinlik değeri. 0 ile 1 arasında bir değer döner:
        - 1.0: Tüm pozitif tahminler doğru (hiç false positive yok)
        - 0.0: Hiçbir pozitif tahmin doğru değil veya hiç pozitif tahmin yok
        Eğer hiç pozitif tahmin yoksa (TP+FP=0), 0.0 döner.
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    
    denominator = true_positive + false_positive
    if denominator == 0:
        return 0.0
    
    return true_positive / denominator


def calculate_recall(y_true, y_pred):
    """
    Duyarlılık (Recall) veya Sensitivity metriğini hesaplar.
    
    Recall, gerçek pozitif örnekler içinde doğru şekilde pozitif olarak tahmin edilenlerin
    oranını gösterir. Modelin pozitif sınıfı ne kadar iyi yakaladığını ölçer. Yüksek recall,
    modelin pozitif örnekleri kaçırmadığı (az false negative) anlamına gelir.
    
    Formül: Recall = True Positive / (True Positive + False Negative)
    
    Argümanlar:
    -----------
    y_true : numpy array, shape (n_samples,)
        Gerçek sınıf etiketleri. Binary classification için 0 veya 1 değerlerini
        içermelidir. Her eleman bir örneğin gerçek sınıfını temsil eder.
    y_pred : numpy array, shape (n_samples,)
        Model tarafından tahmin edilen sınıf etiketleri. Binary classification için
        0 veya 1 değerlerini içermelidir. y_true ile aynı uzunlukta olmalıdır.
    
    Returns:
    --------
    float
        Duyarlılık değeri. 0 ile 1 arasında bir değer döner:
        - 1.0: Tüm gerçek pozitifler doğru tahmin edildi (hiç false negative yok)
        - 0.0: Hiçbir gerçek pozitif doğru tahmin edilmedi veya hiç gerçek pozitif yok
        Eğer hiç gerçek pozitif yoksa (TP+FN=0), 0.0 döner.
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    
    denominator = true_positive + false_negative
    if denominator == 0:
        return 0.0
    
    return true_positive / denominator


def calculate_f_score(y_true, y_pred, beta=1):
    """
    F-Score metriğini hesaplar (beta parametresi ile ayarlanabilir).
    
    F-Score, Precision ve Recall metriklerinin harmonik ortalamasıdır. Bu metrik,
    precision ve recall arasındaki dengeyi ölçer. beta parametresi, recall'un precision'a
    göre ne kadar önemli olduğunu belirler. beta=1 için F1-Score hesaplanır ve bu en yaygın
    kullanılan versiyondur.
    
    Formül: F-Score = (1 + beta^2) * (Precision * Recall) / (beta^2 * Precision + Recall)
    
    - beta < 1: Precision'a daha fazla ağırlık verir
    - beta = 1: Precision ve Recall'a eşit ağırlık verir (F1-Score)
    - beta > 1: Recall'a daha fazla ağırlık verir
    
    Argümanlar:
    -----------
    y_true : numpy array, shape (n_samples,)
        Gerçek sınıf etiketleri. Binary classification için 0 veya 1 değerlerini
        içermelidir. Her eleman bir örneğin gerçek sınıfını temsil eder.
    y_pred : numpy array, shape (n_samples,)
        Model tarafından tahmin edilen sınıf etiketleri. Binary classification için
        0 veya 1 değerlerini içermelidir. y_true ile aynı uzunlukta olmalıdır.
    beta : float, default=1
        F-Score için beta parametresi. Bu parametre, recall'un precision'a göre
        ne kadar önemli olduğunu belirler:
        - beta=1: F1-Score (en yaygın kullanılan)
        - beta=2: F2-Score (recall'a daha fazla ağırlık)
        - beta=0.5: F0.5-Score (precision'a daha fazla ağırlık)
    
    Returns:
    --------
    float
        F-Score değeri. 0 ile 1 arasında bir değer döner:
        - 1.0: Mükemmel precision ve recall
        - 0.0: Precision veya Recall'dan biri 0 ise
        Eğer payda 0 ise (precision ve recall'dan biri 0), 0.0 döner.
    """
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    
    denominator = (beta ** 2 * precision + recall)
    if denominator == 0:
        return 0.0
    
    f_score = (1 + beta ** 2) * (precision * recall) / denominator
    return f_score


def calculate_all_metrics(y_true, y_pred):
    """
    Tüm classification metriklerini hesaplar ve dictionary olarak döndürür.
    
    Bu fonksiyon, binary classification için gerekli tüm metrikleri tek seferde
    hesaplar. Hem performans metriklerini (accuracy, precision, recall, F1-score)
    hem de confusion matrix bileşenlerini (TP, TN, FP, FN) içerir.
    
    Argümanlar:
    -----------
    y_true : numpy array, shape (n_samples,)
        Gerçek sınıf etiketleri. Binary classification için 0 veya 1 değerlerini
        içermelidir. Her eleman bir örneğin gerçek sınıfını temsil eder.
    y_pred : numpy array, shape (n_samples,)
        Model tarafından tahmin edilen sınıf etiketleri. Binary classification için
        0 veya 1 değerlerini içermelidir. y_true ile aynı uzunlukta olmalıdır.
    
    Returns:
    --------
    dict
        Tüm metrikleri içeren dictionary. Anahtarlar ve değerleri:
        - 'accuracy': float, doğruluk değeri (0-1 arası)
        - 'precision': float, kesinlik değeri (0-1 arası)
        - 'recall': float, duyarlılık değeri (0-1 arası)
        - 'f1_score': float, F1 skoru (0-1 arası, beta=1 için)
        - 'true_positive': int, gerçek pozitif sayısı (TP)
        - 'true_negative': int, gerçek negatif sayısı (TN)
        - 'false_positive': int, yanlış pozitif sayısı (FP)
        - 'false_negative': int, yanlış negatif sayısı (FN)
        
    Notlar:
    -------
    Confusion Matrix bileşenleri:
    - TP: Gerçek pozitif, pozitif olarak tahmin edildi
    - TN: Gerçek negatif, negatif olarak tahmin edildi
    - FP: Gerçek negatif, pozitif olarak tahmin edildi (Tip I hatası)
    - FN: Gerçek pozitif, negatif olarak tahmin edildi (Tip II hatası)
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    
    metrics = {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred),
        'recall': calculate_recall(y_true, y_pred),
        'f1_score': calculate_f_score(y_true, y_pred, beta=1),
        'true_positive': int(true_positive),
        'true_negative': int(true_negative),
        'false_positive': int(false_positive),
        'false_negative': int(false_negative)
    }
    
    return metrics
