"""
Manuel Tahmin Scripti
Kullanıcının girdiği Exam1 ve Exam2 değerleriyle model tahmini yapar.
"""

import numpy as np
from src.dataset import load_and_process_data
from src.model import LogisticRegression
from src.train import train as train_model
from src.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, LEARNING_RATE, NUM_EPOCHS, RANDOM_SEED


def load_trained_model():
    """
    Eğitilmiş modeli yükler veya yeni eğitir.
    
    Returns:
    --------
    tuple : (model, train_mean, train_std)
        Eğitilmiş model ve normalizasyon parametreleri (orijinal veriden hesaplanmış)
    """
    print("Model yükleniyor ve eğitiliyor...")
    print("Bu işlem biraz zaman alabilir...\n")
    
    # Orijinal veriyi yükle (normalize edilmemiş)
    raw_data = np.loadtxt('data/hw1_data.csv', delimiter=',', dtype=np.float32)
    X_orig = raw_data[:, 0:2]
    y_orig = raw_data[:, 2]
    
    # Veriyi karıştır ve böl (eğitim sırasındakiyle aynı)
    np.random.seed(RANDOM_SEED)
    indices = np.arange(len(X_orig))
    np.random.shuffle(indices)
    X_orig = X_orig[indices]
    y_orig = y_orig[indices]
    
    # Veriyi train/val/test olarak ayır (eğitim sırasındakiyle aynı)
    n_samples = len(X_orig)
    n_train = int(n_samples * TRAIN_RATIO)
    n_val = int(n_samples * VAL_RATIO)
    
    X_train_orig = X_orig[:n_train]
    y_train = y_orig[:n_train]
    
    X_val_orig = X_orig[n_train:n_train + n_val]
    y_val = y_orig[n_train:n_train + n_val]
    
    # Normalizasyon parametrelerini ORİJİNAL train setinden hesapla
    train_mean = np.mean(X_train_orig, axis=0)
    train_std = np.std(X_train_orig, axis=0)
    train_std = train_std + 1e-8  # Epsilon ekle
    
    # Normalize et
    X_train = (X_train_orig - train_mean) / train_std
    X_val = (X_val_orig - train_mean) / train_std
    
    # Modeli başlat ve eğit
    model = LogisticRegression(learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS)
    history = train_model(model, X_train, y_train, X_val, y_val, 
                         patience=10, verbose=False, early_stopping=True)
    
    print(f"Model eğitildi! En iyi validation loss: {history['best_val_loss']:.4f}")
    print(f"Eğitim {history['best_epoch'] + 1} epoch'ta durduruldu.\n")
    
    return model, train_mean, train_std


def predict_single(model, train_mean, train_std, exam1, exam2):
    """
    Tek bir örnek için tahmin yapar.
    
    Argümanlar:
    -----------
    model : LogisticRegression
        Eğitilmiş model
    train_mean : numpy array
        Eğitim setinden hesaplanan ortalama değerler
    train_std : numpy array
        Eğitim setinden hesaplanan standart sapma değerleri
    exam1 : float
        İlk sınav skoru
    exam2 : float
        İkinci sınav skoru
    
    Returns:
    --------
    dict
        Tahmin sonuçları (olasılık, tahmin, açıklama)
    """
    # Girdiyi numpy array'e çevir
    X_input = np.array([[exam1, exam2]], dtype=np.float32)
    
    # Normalizasyon uygula (train istatistikleri kullanarak)
    X_normalized = (X_input - train_mean) / train_std
    
    # Tahmin yap
    y_pred_proba = model.forward(X_normalized)[0]  # Olasılık
    y_pred = model.predict(X_normalized)[0]  # Binary tahmin (0 veya 1)
    
    # Sonuçları hazırla
    result = {
        'exam1': exam1,
        'exam2': exam2,
        'probability': float(y_pred_proba),
        'prediction': int(y_pred),
        'prediction_text': 'Kabul Edildi (Accept)' if y_pred == 1 else 'Reddedildi (Reject)',
        'confidence': 'Yüksek' if abs(y_pred_proba - 0.5) > 0.3 else 'Orta' if abs(y_pred_proba - 0.5) > 0.15 else 'Düşük'
    }
    
    return result


def print_prediction(result):
    """
    Tahmin sonuçlarını güzel bir formatta yazdırır.
    
    Argümanlar:
    -----------
    result : dict
        predict_single fonksiyonundan dönen sonuç dictionary'si
    """
    print("\n" + "="*60)
    print(" " * 15 + "TAHMIN SONUÇLARI")
    print("="*60)
    print(f"\nGirdi Değerleri:")
    print(f"  Exam 1 Skoru: {result['exam1']:.2f}")
    print(f"  Exam 2 Skoru: {result['exam2']:.2f}")
    print(f"\nTahmin:")
    print(f"  Sonuç: {result['prediction_text']}")
    print(f"  Olasılık: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
    print(f"  Güven Seviyesi: {result['confidence']}")
    
    # Yorum ekle
    if result['prediction'] == 1:
        if result['probability'] > 0.8:
            print(f"\nYorum: Yüksek olasılıkla kabul edilecek.")
        elif result['probability'] > 0.6:
            print(f"\nYorum: Muhtemelen kabul edilecek.")
        else:
            print(f"\nYorum: Kabul edilme olasılığı var ama kesin değil.")
    else:
        if result['probability'] < 0.2:
            print(f"\nYorum: Yüksek olasılıkla reddedilecek.")
        elif result['probability'] < 0.4:
            print(f"\nYorum: Muhtemelen reddedilecek.")
        else:
            print(f"\nYorum: Reddedilme olasılığı var ama kesin değil.")
    
    print("="*60 + "\n")


def interactive_mode():
    """
    İnteraktif mod: Kullanıcıdan sürekli girdi alır.
    """
    print("="*70)
    print(" " * 15 + "MANUEL TAHMIN MODU")
    print("="*70)
    print("\nBu mod, girdiğiniz Exam1 ve Exam2 değerleriyle model tahmini yapar.")
    print("Çıkmak için 'q' veya 'quit' yazın.\n")
    
    # Modeli yükle
    model, train_mean, train_std = load_trained_model()
    
    print("Model hazır! Tahmin yapmaya başlayabilirsiniz.\n")
    
    while True:
        try:
            # Kullanıcıdan girdi al
            user_input = input("Exam1 ve Exam2 değerlerini girin (örn: 75.5 82.3) veya 'q' ile çıkın: ").strip()
            
            # Çıkış kontrolü
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\nÇıkılıyor...")
                break
            
            # Girdiyi parse et
            values = user_input.split()
            if len(values) != 2:
                print("Hata: Lütfen iki değer girin (Exam1 Exam2)\n")
                continue
            
            exam1 = float(values[0])
            exam2 = float(values[1])
            
            # Tahmin yap
            result = predict_single(model, train_mean, train_std, exam1, exam2)
            print_prediction(result)
            
        except ValueError:
            print("Hata: Lütfen geçerli sayısal değerler girin.\n")
        except KeyboardInterrupt:
            print("\n\nÇıkılıyor...")
            break
        except Exception as e:
            print(f"Hata: {e}\n")


def single_prediction_mode(exam1, exam2):
    """
    Tek tahmin modu: Komut satırından değer alır.
    
    Argümanlar:
    -----------
    exam1 : float
        İlk sınav skoru
    exam2 : float
        İkinci sınav skoru
    """
    print("="*70)
    print(" " * 15 + "TEK TAHMIN MODU")
    print("="*70)
    
    # Modeli yükle
    model, train_mean, train_std = load_trained_model()
    
    # Tahmin yap
    result = predict_single(model, train_mean, train_std, exam1, exam2)
    print_prediction(result)


def main():
    """
    Ana fonksiyon: Komut satırı argümanlarını kontrol eder.
    """
    import sys
    
    if len(sys.argv) == 1:
        # Argüman yoksa interaktif mod
        interactive_mode()
    elif len(sys.argv) == 3:
        # İki argüman varsa tek tahmin modu
        try:
            exam1 = float(sys.argv[1])
            exam2 = float(sys.argv[2])
            single_prediction_mode(exam1, exam2)
        except ValueError:
            print("Hata: Lütfen geçerli sayısal değerler girin.")
            print("Kullanım: python predict.py <exam1> <exam2>")
            print("Örnek: python predict.py 75.5 82.3")
    else:
        print("Kullanım:")
        print("  İnteraktif mod: python predict.py")
        print("  Tek tahmin: python predict.py <exam1> <exam2>")
        print("\nÖrnekler:")
        print("  python predict.py")
        print("  python predict.py 75.5 82.3")


if __name__ == "__main__":
    main()

