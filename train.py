"""
Main Training Script
Trains the logistic regression model and saves results.
"""

import numpy as np
from src.dataset import load_and_process_data
from src.model import LogisticRegression
from src.train import train as train_model
from src.visualization import plot_data_distribution, plot_loss_history
from src.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, LEARNING_RATE, NUM_EPOCHS, RANDOM_SEED
from src.utils import setup_logging


def main():
    """
    Ana eğitim fonksiyonu: Logistic Regression modelini eğitir ve sonuçları kaydeder.
    
    Bu fonksiyon, sadece model eğitimi için gerekli adımları gerçekleştirir:
    1. CSV dosyasından veriyi yükler ve train/validation/test setlerine ayırır
    2. Eğitim setinin veri dağılımını görselleştirir ve kaydeder
    3. LogisticRegression modelini başlatır
    4. Modeli Stochastic Gradient Descent (SGD) ile eğitir
    5. Eğitim ve doğrulama loss değerlerini görselleştirir ve kaydeder
    6. Eğitim özet bilgilerini konsola yazdırır
    
    Eğitim sırasında her 100 epoch'ta bir loss değerleri konsola yazdırılır.
    Overfitting tespit edilirse uyarı mesajı gösterilir.
    
    Argümanlar:
    -----------
    Yok (tüm parametreler src/config.py dosyasından alınır)
    
    Returns:
    --------
    Yok (fonksiyon tüm sonuçları konsola yazdırır ve grafikleri 'results/' klasörüne kaydeder)
    
    Kaydedilen Dosyalar:
    --------------------
    - results/initial_data_dist.png: Veri dağılımı scatter plot
    - results/loss_curve.png: Training ve validation loss eğrisi
    - results/training_log.txt: Tüm konsol çıktıları
    """
    print("="*60)
    print("Logistic Regression Eğitimi Başlatılıyor...")
    print("="*60)
    
    # 1. Veriyi yükle ve ayır
    print("\n1. Veri yükleniyor ve bölünüyor...")
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
    print("\n2. Veri dağılımı görselleştiriliyor...")
    plot_data_distribution(X_train, y_train, save_path='results/initial_data_dist.png')
    
    # 3. Modeli başlat
    print("\n3. Model başlatılıyor...")
    model = LogisticRegression(learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS)
    print(f"   Öğrenme oranı: {LEARNING_RATE}")
    print(f"   Maksimum epoch: {NUM_EPOCHS}")
    
    # 4. Modeli eğit
    print("\n4. Model eğitiliyor...")
    history = train_model(model, X_train, y_train, X_val, y_val, patience=10, verbose=True)
    
    # 5. Loss eğrisini görselleştir
    print("\n5. Loss eğrisi görselleştiriliyor...")
    plot_loss_history(
        history['train_losses'],
        history['val_losses'],
        save_path='results/loss_curve.png'
    )
    
    # 6. Sonuçları yazdır
    print("\n" + "="*60)
    print("Eğitim Özeti:")
    print("="*60)
    print(f"En iyi validation loss: {history['best_val_loss']:.4f} (Epoch {history['best_epoch'] + 1})")
    print(f"Final training loss: {history['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {history['val_losses'][-1]:.4f}")
    if history['overfitting_detected']:
        print("Overfitting tespit edildi!")
    print("="*60)
    
    print("\nEğitim tamamlandı! Sonuçlar 'results/' klasörüne kaydedildi.")


if __name__ == "__main__":
    # Loglama sistemini başlat
    tee = setup_logging('results/training_log.txt')
    
    try:
        main()
        print("\nTüm loglar 'results/training_log.txt' dosyasına kaydedildi.")
    finally:
        # Loglama sistemini kapat
        tee.close()

