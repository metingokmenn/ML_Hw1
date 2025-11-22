
import sys
import os
from datetime import datetime


class Tee:
    """
    Konsol çıktısını hem ekrana hem de dosyaya yazdıran sınıf.
    
    Bu sınıf, Python'ın print() fonksiyonlarının çıktısını hem konsola
    hem de belirtilen bir dosyaya yazdırmak için kullanılır.
    """
    
    def __init__(self, file_path):
        """
        Tee sınıfını başlatır.
        
        Argümanlar:
        -----------
        file_path : str
            Çıktının kaydedileceği dosya yolu
        """
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
    
    def write(self, data):
        """
        Veriyi hem konsola hem de dosyaya yazar.
        
        Argümanlar:
        -----------
        data : str
            Yazdırılacak veri
        """
        self.file.write(data)
        self.file.flush()  
        self.stdout.write(data)
        self.stdout.flush()
    
    def flush(self):
        """
        Buffer'ı temizler.
        """
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        """
        Dosyayı kapatır ve stdout'u eski haline döndürür.
        """
        if self.file:
            self.file.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr


def setup_logging(log_file_path='results/training_log.txt'):
    """
    Loglama sistemini başlatır ve Tee instance'ı döndürür.
    
    Bu fonksiyon, tüm konsol çıktısını hem ekrana hem de belirtilen dosyaya
    yazdırmak için gerekli ayarlamaları yapar. Kullanım sonrası close()
    metodunu çağırmayı unutmayın.
    
    Argümanlar:
    -----------
    log_file_path : str, default='results/training_log.txt'
        Log dosyasının kaydedileceği yol. Klasör yoksa otomatik oluşturulur.
    
    Returns:
    --------
    Tee
        Konsol çıktısını yönlendiren Tee instance'ı. Kullanım sonrası
        tee.close() ile kapatılmalıdır.
    
    Örnek Kullanım:
    ---------------
    >>> tee = setup_logging('results/my_log.txt')
    >>> print("Bu hem ekrana hem de dosyaya yazılacak")
    >>> tee.close()
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    tee = Tee(log_file_path)
    
    sys.stdout = tee
    sys.stderr = tee
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("="*70)
    print(f"Log Dosyası Başlatıldı: {timestamp}")
    print(f"Log Yolu: {log_file_path}")
    print("="*70)
    print()
    
    return tee
