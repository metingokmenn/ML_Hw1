"""
Main Entry Point
Orchestrates the entire logistic regression workflow.
"""

import numpy as np
from src.dataset import load_and_process_data
from src.model import LogisticRegression
from src.train import train as train_model
from src.eval import evaluate_all
from src.visualization import plot_data_distribution, plot_loss_history, plot_test_predictions, plot_confusion_matrix
from src.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, LEARNING_RATE, NUM_EPOCHS, RANDOM_SEED
from src.utils import setup_logging


def main():
    """
    Ana fonksiyon: Tüm logistic regression workflow'unu yönetir ve çalıştırır.
    
    Bu fonksiyon, projenin tüm adımlarını sırasıyla gerçekleştirir:
    1. CSV dosyasından veriyi yükler ve train/validation/test setlerine ayırır (60/20/20)
    2. Eğitim setinin veri dağılımını scatter plot olarak görselleştirir ve kaydeder
    3. LogisticRegression modelini belirtilen hyperparameter'larla başlatır
    4. Modeli Stochastic Gradient Descent (SGD) kullanarak eğitir
    5. Eğitim ve doğrulama loss değerlerini epoch'lara göre görselleştirir ve kaydeder
    6. Modeli tüm veri setleri (train, validation, test) üzerinde değerlendirir
    7. Test seti tahminlerini görselleştirir (doğru/yanlış tahminler ve confusion matrix)
    8. Final özet bilgileri konsola yazdırır
    
    Tüm görselleştirmeler ve loglar 'results/' klasörüne kaydedilir:
    - results/initial_data_dist.png: Veri dağılımı scatter plot
    - results/loss_curve.png: Loss eğrisi grafiği
    - results/test_predictions.png: Test seti tahmin sonuçları (doğru/yanlış)
    - results/confusion_matrix.png: Confusion matrix görselleştirmesi
    - results/training_log.txt: Tüm konsol çıktıları
    
    Argümanlar:
    -----------
    Yok (tüm parametreler src/config.py dosyasından alınır)
    
    Returns:
    --------
    Yok (fonksiyon tüm sonuçları konsola yazdırır ve dosyalara kaydeder)
    
    Notlar:
    -------
    - Eğitim işlemi uzun sürebilir (NUM_EPOCHS kadar epoch çalışır)
    - Overfitting tespit edilirse konsola uyarı yazdırılır
    - Tüm metrikler (accuracy, precision, recall, F1-score) konsola yazdırılır
    """
    print("="*70)
    print(" " * 15 + "LOGISTIC REGRESSION FROM SCRATCH")
    print("="*70)
    
    # 1. Veriyi yükle ve ayır
    print("\n[1/6] Veri yükleniyor ve bölünüyor...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_process_data(
        'data/hw1_data.csv',
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_state=RANDOM_SEED
    )
    
    print(f"   Eğitim seti: {X_train.shape[0]} örnek")
    print(f"   Doğrulama seti: {X_val.shape[0]} örnek")
    print(f"   Test seti: {X_test.shape[0]} örnek")
    
    # 2. İlk veri dağılımını görselleştir
    print("\n[2/6] Veri dağılımı görselleştiriliyor...")
    plot_data_distribution(X_train, y_train, save_path='results/initial_data_dist.png')
    print("   Grafik kaydedildi: results/initial_data_dist.png")
    
    # 3. Modeli başlat
    print("\n[3/6] Model başlatılıyor...")
    model = LogisticRegression(learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS)
    print(f"   Öğrenme oranı: {LEARNING_RATE}")
    print(f"   Maksimum epoch: {NUM_EPOCHS}")
    
    # 4. Modeli eğit
    print("\n[4/6] Model eğitiliyor (Stochastic Gradient Descent)...")
    print("   Bu işlem biraz zaman alabilir...\n")
    history = train_model(model, X_train, y_train, X_val, y_val, patience=10, verbose=True)
    
    # 5. Loss eğrisini görselleştir
    print("\n[5/6] Loss eğrisi görselleştiriliyor...")
    plot_loss_history(
        history['train_losses'],
        history['val_losses'],
        save_path='results/loss_curve.png'
    )
    print("   Grafik kaydedildi: results/loss_curve.png")
    
    # 6. Modeli değerlendir
    print("\n[6/6] Model tüm setler üzerinde değerlendiriliyor...")
    results = evaluate_all(model, X_train, y_train, X_val, y_val, X_test, y_test)
    
    # 7. Test seti tahminlerini görselleştir
    print("\n[7/7] Test seti tahminleri görselleştiriliyor...")
    y_test_pred = model.predict(X_test)
    plot_test_predictions(X_test, y_test, y_test_pred, save_path='results/test_predictions.png')
    plot_confusion_matrix(y_test, y_test_pred, save_path='results/confusion_matrix.png')
    print("   Grafikler kaydedildi: results/test_predictions.png, results/confusion_matrix.png")
    
    # Final özet
    print("\n" + "="*70)
    print(" " * 20 + "FİNAL ÖZET")
    print("="*70)
    print(f"\nEğitim Özeti:")
    print(f"  • En iyi validation loss: {history['best_val_loss']:.4f} (Epoch {history['best_epoch'] + 1})")
    print(f"  • Final training loss: {history['train_losses'][-1]:.4f}")
    print(f"  • Final validation loss: {history['val_losses'][-1]:.4f}")
    if history['overfitting_detected']:
        print(f"  • Overfitting tespit edildi!")
    
    print(f"\nTest Seti Performansı:")
    print(f"  • Accuracy: {results['test']['accuracy']:.4f}")
    print(f"  • Precision: {results['test']['precision']:.4f}")
    print(f"  • Recall: {results['test']['recall']:.4f}")
    print(f"  • F1-Score: {results['test']['f1_score']:.4f}")
    
    print("\n" + "="*70)
    print("Tüm işlemler tamamlandı!")
    print("   Grafikler 'results/' klasörüne kaydedildi:")
    print("   - initial_data_dist.png: Veri dağılımı")
    print("   - loss_curve.png: Loss eğrisi")
    print("   - test_predictions.png: Test tahmin sonuçları")
    print("   - confusion_matrix.png: Confusion matrix")
    print("="*70)


if __name__ == "__main__":
    # Loglama sistemini başlat
    tee = setup_logging('results/training_log.txt')
    
    try:
        main()
        print("\nTüm loglar 'results/training_log.txt' dosyasına kaydedildi.")
    finally:
        # Loglama sistemini kapat
        tee.close()
