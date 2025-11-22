"""
Main Evaluation Script
Evaluates the trained logistic regression model on test set.
"""

from src.dataset import load_and_process_data
from src.model import LogisticRegression
from src.eval import evaluate_all
from src.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
from src.utils import setup_logging


def main():
    """
    Ana değerlendirme fonksiyonu: Eğitilmiş modeli tüm veri setleri üzerinde değerlendirir.
    
    Bu fonksiyon, eğitilmiş bir Logistic Regression modelini train, validation ve test
    setleri üzerinde değerlendirir. Her set için ayrı ayrı metrikleri hesaplar ve konsola
    yazdırır.
    
    Önemli Not: Bu script, örnek amaçlı olarak yeni bir model instance'ı oluşturur ancak
    eğitmez. Gerçek kullanımda, eğitilmiş modelin ağırlıkları kaydedilmeli ve burada
    yüklenmelidir.
    
    Argümanlar:
    -----------
    Yok (tüm parametreler src/config.py dosyasından alınır)
    
    Returns:
    --------
    Yok (fonksiyon tüm metrikleri konsola yazdırır)
    
    Hesaplanan Metrikler:
    ---------------------
    Her veri seti için:
    - Accuracy (Doğruluk)
    - Precision (Kesinlik)
    - Recall (Duyarlılık)
    - F1-Score
    - Confusion Matrix bileşenleri (TP, TN, FP, FN)
    """
    print("="*60)
    print("Model Değerlendirmesi Başlatılıyor...")
    print("="*60)
    
    # 1. Veriyi yükle (eğitim sırasındakiyle aynı split)
    print("\n1. Veri yükleniyor...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_process_data(
        'data/hw1_data.csv',
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_state=RANDOM_SEED
    )
    
    # 2. Modeli yükle (bu örnekte yeni bir model oluşturuyoruz, 
    # gerçek kullanımda kaydedilmiş model yüklenmeli)
    print("\n2. Model yükleniyor...")
    print("   Not: Bu örnekte model yeniden eğitilmedi.")
    print("   Gerçek kullanımda kaydedilmiş model ağırlıkları yüklenmelidir.")
    
    # Model instance'ı oluştur (gerçek kullanımda kaydedilmiş ağırlıklar yüklenmeli)
    model = LogisticRegression()
    
    # 3. Modeli değerlendir
    print("\n3. Model değerlendiriliyor...")
    results = evaluate_all(model, X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("\nDeğerlendirme tamamlandı!")


if __name__ == "__main__":
    # Loglama sistemini başlat
    tee = setup_logging('results/evaluation_log.txt')
    
    try:
        main()
        print("\nTüm loglar 'results/evaluation_log.txt' dosyasına kaydedildi.")
    finally:
        # Loglama sistemini kapat
        tee.close()

