# Kaynak Kod Yapısı

Bu dizin, logistic regression projesinin modüler implementasyonunu içerir.

## Modül Açıklamaları

### `dataset.py`

Tüm veri ile ilgili işlemleri yönetir:

- CSV dosyalarından veri yükleme
- Veri ön işleme
- Veriyi train/validation/test setlerine ayırma (60/20/20)

### `model.py`

Ana LogisticRegression model sınıfını içerir:

- Sigmoid aktivasyon fonksiyonu
- Forward pass (ileri geçiş)
- Tahmin metodu
- Cross-entropy loss hesaplama
- Ağırlık güncelleme mantığı (SGD)

### `train.py`

Eğitim döngüsünü implemente eder:

- SGD ile eğitim fonksiyonu
- Loss takibi
- Eğitim geçmişi yönetimi
- Early stopping ve overfitting kontrolü

### `eval.py`

Model değerlendirmesini yönetir:

- Veri setleri üzerinde değerlendirme
- Metrik hesaplamaları
- Sonuç toplama ve raporlama

### `metrics.py`

Metrik hesaplama fonksiyonlarını içerir:

- Accuracy (Doğruluk)
- Precision (Kesinlik)
- Recall (Duyarlılık)
- F-Score

### `visualization.py`

Grafik ve görselleştirme fonksiyonları:

- Veri dağılımı grafikleri
- Loss eğrisi grafikleri
- Test tahmin görselleştirmeleri
- Confusion matrix görselleştirmesi

### `config.py`

Proje konfigürasyonu ve hyperparameter'lar:

- Veri bölme oranları
- Öğrenme oranı (learning rate)
- Epoch sayısı
- Random seed

### `utils.py`

Modüller arasında kullanılan genel yardımcı fonksiyonlar:

- Loglama sistemi (Tee sınıfı)
- Konsol çıktısını dosyaya yazdırma

### `data_loader.py`

Alternatif veri yükleme modülü (dataset.py'den fonksiyonları yeniden export eder).
