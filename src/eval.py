
from .model import LogisticRegression
from .metrics import calculate_all_metrics


def evaluate(model, X, y, set_name="Dataset"):
    """
    Modeli bir veri seti üzerinde değerlendirir ve tüm metrikleri hesaplar.
    
    Bu fonksiyon, verilen model ve veri seti için binary classification metriklerini
    hesaplar ve sonuçları hem konsola yazdırır hem de dictionary olarak döndürür.
    Hesaplanan metrikler: Accuracy, Precision, Recall, F1-Score ve Confusion Matrix
    bileşenleri (TP, TN, FP, FN).
    
    Argümanlar:
    -----------
    model : LogisticRegression
        Değerlendirilecek LogisticRegression model instance'ı. Modelin predict()
        metodunu kullanarak tahminler yapar.
    X : numpy array, shape (n_samples, n_features)
        Değerlendirme için kullanılacak özellik matrisi. Her satır bir örneği,
        her sütun bir özelliği temsil eder.
    y : numpy array, shape (n_samples,)
        Gerçek sınıf etiketleri. Binary classification için 0 veya 1 değerlerini
        içermelidir.
    set_name : str, default="Dataset"
        Veri setinin adı. Konsol çıktısında hangi set için metriklerin
        gösterildiğini belirtmek için kullanılır (örn: "Training", "Validation", "Test").
    
    Returns:
    --------
    dict
        Tüm metrikleri içeren dictionary. Anahtarlar:
        - 'accuracy': float, doğruluk değeri (0-1 arası)
        - 'precision': float, kesinlik değeri (0-1 arası)
        - 'recall': float, duyarlılık değeri (0-1 arası)
        - 'f1_score': float, F1 skoru (0-1 arası)
        - 'true_positive': int, gerçek pozitif sayısı
        - 'true_negative': int, gerçek negatif sayısı
        - 'false_positive': int, yanlış pozitif sayısı
        - 'false_negative': int, yanlış negatif sayısı
    """
    
    y_pred = model.predict(X)
    
    
    metrics = calculate_all_metrics(y, y_pred)
    
    
    print(f"\n{'='*60}")
    print(f"{set_name} Seti Metrikleri:")
    print(f"{'='*60}")
    print(f"Doğruluk (Accuracy)  : {metrics['accuracy']:.4f}")
    print(f"Kesinlik (Precision) : {metrics['precision']:.4f}")
    print(f"Duyarlılık (Recall)  : {metrics['recall']:.4f}")
    print(f"F1-Skoru (F1-Score) : {metrics['f1_score']:.4f}")
    print(f"\nKarışıklık Matrisi (Confusion Matrix):")
    print(f"  Gerçek Pozitif  (TP): {metrics['true_positive']}")
    print(f"  Gerçek Negatif  (TN): {metrics['true_negative']}")
    print(f"  Yanlış Pozitif  (FP): {metrics['false_positive']}")
    print(f"  Yanlış Negatif  (FN): {metrics['false_negative']}")
    print(f"{'='*60}\n")
    
    return metrics


def evaluate_all(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Modeli tüm veri setleri (train, validation, test) üzerinde değerlendirir.
    
    Bu fonksiyon, modelin performansını eğitim, doğrulama ve test setleri üzerinde
    ayrı ayrı değerlendirir. Her set için tüm metrikleri hesaplar ve konsola
    yazdırır. Sonuçları organize bir şekilde dictionary olarak döndürür.
    
    Argümanlar:
    -----------
    model : LogisticRegression
        Değerlendirilecek LogisticRegression model instance'ı. Modelin eğitilmiş
        olması ve predict() metodunun çalışır durumda olması gerekir.
    X_train : numpy array, shape (n_train_samples, n_features)
        Eğitim seti özellik matrisi. Modelin üzerinde eğitildiği veri seti.
    y_train : numpy array, shape (n_train_samples,)
        Eğitim seti gerçek sınıf etiketleri (0 veya 1).
    X_val : numpy array, shape (n_val_samples, n_features)
        Doğrulama seti özellik matrisi. Hyperparameter ayarları için kullanılan
        veri seti.
    y_val : numpy array, shape (n_val_samples,)
        Doğrulama seti gerçek sınıf etiketleri (0 veya 1).
    X_test : numpy array, shape (n_test_samples, n_features)
        Test seti özellik matrisi. Modelin final performansını değerlendirmek
        için kullanılan veri seti.
    y_test : numpy array, shape (n_test_samples,)
        Test seti gerçek sınıf etiketleri (0 veya 1).
    
    Returns:
    --------
    dict
        Tüm veri setleri için metrikleri içeren nested dictionary. Yapı:
        {
            'train': dict,      # Eğitim seti metrikleri (accuracy, precision, recall, f1_score, vb.)
            'validation': dict, # Doğrulama seti metrikleri
            'test': dict        # Test seti metrikleri
        }
        Her iç dictionary, evaluate() fonksiyonunun döndürdüğü formatı içerir.
    """
    train_metrics = evaluate(model, X_train, y_train, "Eğitim")
    val_metrics = evaluate(model, X_val, y_val, "Doğrulama")
    test_metrics = evaluate(model, X_test, y_test, "Test")
    
    results = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    return results
