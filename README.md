# Logistic Regression from Scratch

## Açıklama

Makine Öğrenmesi (BLM5110) dersi kapsamında sıfırdan (scratch) geliştirilen Logistic Regression sınıflandırma modeli. Yüksek seviye makine öğrenmesi kütüphaneleri (sklearn, keras, tensorflow) kullanılmadan, sadece numpy, pandas ve matplotlib kullanılarak implemente edilmiştir.

**Amaç:** İki sınav skoruna (Exam1, Exam2) dayanarak öğrencilerin iş başvurusu kabul durumunu (1: Kabul, 0: Red) tahmin etmek.

## Gereksinimler

```bash
pip install -r requirements.txt
```

## Çalıştırma

### Tüm Workflow'u Çalıştırma

```bash
python main.py
```

### Eğitim

```bash
python train.py
```

### Değerlendirme

```bash
python eval.py
```

### Manuel Tahmin

Kendi Exam1 ve Exam2 değerlerinizle tahmin yapmak için:

**İnteraktif Mod:**

```bash
python predict.py
```

**Tek Tahmin Modu:**

```bash
python predict.py 75.5 82.3
```

## Dosya Düzeni

```
proje_klasoru/
├── data/
│   └── hw1_data.csv          # Veri seti (Exam1, Exam2, Label)
│
├── src/
│   ├── __init__.py
│   ├── config.py             # Hyperparameter ayarları
│   ├── dataset.py             # Veri yükleme ve preprocessing
│   ├── data_loader.py         # Alternatif veri yükleme modülü
│   ├── model.py              # LogisticRegression sınıfı
│   ├── train.py              # Eğitim döngüsü ve SGD
│   ├── eval.py               # Model değerlendirme
│   ├── metrics.py            # Metrik hesaplamaları (manuel)
│   ├── utils.py              # Yardımcı fonksiyonlar ve loglama
│   ├── visualization.py      # Görselleştirme fonksiyonları
│   └── README.md             # Kaynak kod dokümantasyonu
│
├── results/
│   ├── initial_data_dist.png  # İlk veri dağılımı grafiği
│   ├── loss_curve.png         # Loss eğrisi grafiği
│   ├── test_predictions.png   # Test seti tahmin sonuçları
│   ├── confusion_matrix.png   # Confusion matrix görselleştirmesi
│   ├── training_log.txt      # Tüm konsol çıktıları
│   └── evaluation_log.txt    # Değerlendirme logları
│
├── main.py                    # Tüm workflow'u çalıştıran ana script
├── train.py                   # Ana eğitim scripti
├── eval.py                    # Ana değerlendirme scripti
├── predict.py                 # Manuel tahmin scripti
├── requirements.txt           # Gerekli Python paketleri
└── README.md                  # Bu dosya
```

## Özellikler

- ✅ Sıfırdan Logistic Regression implementasyonu
- ✅ Stochastic Gradient Descent (SGD) optimizasyonu
- ✅ Early stopping ve overfitting kontrolü
- ✅ Manuel metrik hesaplamaları (Accuracy, Precision, Recall, F1-Score)
- ✅ Veri görselleştirmeleri ve sonuç analizi
- ✅ Detaylı loglama sistemi

## Sonuçlar

Eğitim tamamlandıktan sonra `results/` klasöründe:

- Veri dağılımı grafikleri
- Loss eğrisi grafikleri
- Test tahmin görselleştirmeleri
- Confusion matrix
- Tüm konsol çıktıları (log dosyaları)
