# 🚁 Pixhawk Flight Analyzer - Hızlı Kullanım Kılavuzu

## 📋 Gereksinimler

Paket otomatik olarak kuruldu. Gerekli kütüphaneler:
- pymavlink
- numpy
- pandas
- matplotlib
- plotly
- scipy
- click

## 🎯 Kullanım Yöntemleri

### **Yöntem 1: Python Script ile (EN KOLAY)**

1. `.tlog` veya `.bin` dosyanızı hazırlayın

2. `test_analyzer.py` dosyasını çalıştırın:

```bash
# Dosyayı düzenleyin ve kendi .tlog dosyanızın yolunu yazın
nano test_analyzer.py  # veya herhangi bir editör

# Çalıştırın
python3 test_analyzer.py
```

**Örnek Çıktı:**
```
🚁 Pixhawk Flight Analyzer
============================================================
📥 Flight log dosyası yükleniyor...
✅ 15 farklı mesaj tipi yüklendi

📊 İstatistikler hesaplanıyor...

FLIGHT STATISTICS SUMMARY
============================================================
📅 Duration: 12.50 minutes (750.0 seconds)
📏 Distance: 2.45 km (2450 m)
🔝 Altitude:
   Max: 120.5 m
   Min: 5.2 m
   Avg: 45.3 m
⚡ Speed (Ground):
   Max: 15.2 km/h (4.2 m/s)
   Avg: 8.5 km/h (2.4 m/s)
```

### **Yöntem 2: CLI Komutları ile**

```bash
# Analiz yap
python3 cli_analyzer.py analyze your_flight.tlog

# Sadece görselleştirme
python3 cli_analyzer.py visualize your_flight.tlog

# Tam analiz (her şey)
python3 cli_analyzer.py process your_flight.tlog --output-dir sonuclar

# Dosya bilgisi
python3 cli_analyzer.py info your_flight.tlog
```

**CLI Komut Seçenekleri:**

```bash
# Tüm grafikleri PNG ve HTML olarak oluştur
python3 cli_analyzer.py visualize ucus.tlog --format both --output-dir grafikler

# Sadece 3D görselleştirme
python3 cli_analyzer.py visualize ucus.tlog --plot-type 3d --format html

# Dashboard oluştur ve göster
python3 cli_analyzer.py visualize ucus.tlog --plot-type dashboard --show
```

### **Yöntem 3: Kendi Python Kodunuzda**

```python
from pixhawk_flight_analyzer import (
    FlightDataLoader,
    FlightDataProcessor,
    FlightAnalyzer,
    FlightVisualizer
)

# 1. Veriyi yükle
loader = FlightDataLoader('ucus.tlog')
data = loader.load()

# 2. İşle
processor = FlightDataProcessor(data)
cleaned_data = processor.clean_data()
flight_path = processor.extract_flight_path()

# 3. Analiz et
analyzer = FlightAnalyzer(data)
stats = analyzer.get_statistics()
analyzer.print_summary()

# 4. Görselleştir
visualizer = FlightVisualizer(data)
visualizer.plot_flight_path_3d_interactive(save_path='ucus.html')
visualizer.plot_dashboard(save_path='dashboard.png')
```

### **Yöntem 4: Jupyter Notebook'ta**

```python
import sys
sys.path.insert(0, '/home/user/project-')

from pixhawk_flight_analyzer import *

# Inline grafik gösterimi için
%matplotlib inline

loader = FlightDataLoader('ucus.tlog')
data = loader.load()

visualizer = FlightVisualizer(data)
visualizer.plot_altitude_profile(show=True)
```

## 📊 Oluşturulan Dosyalar

Analiz sonucunda şu dosyalar oluşur:

### **İstatistikler:**
- `ucus_istatistikleri.csv` - Tüm istatistikler CSV formatında

### **Görselleştirmeler (PNG):**
- `ucus_yolu_2d.png` - 2D uçuş yolu (kuşbakışı)
- `ucus_yolu_3d.png` - 3D uçuş yolu (statik)
- `yukseklik_profili.png` - Zaman-yükseklik grafiği
- `hiz_profili.png` - Zaman-hız grafiği
- `attitude.png` - Roll, pitch, yaw grafiği
- `dashboard.png` - Tüm grafikler tek ekranda

### **İnteraktif (HTML):**
- `ucus_yolu_3d.html` - 3D interaktif uçuş yolu (fareyle döndürülebilir)
- `dashboard.html` - İnteraktif dashboard

## 🎨 Ne Tür Analizler Yapar?

### **Zaman İstatistikleri:**
- Toplam uçuş süresi
- Başlangıç/bitiş zamanları

### **Yükseklik İstatistikleri:**
- Minimum, maksimum, ortalama yükseklik
- Yükseklik değişim aralığı

### **Hız İstatistikleri:**
- Maksimum ve ortalama yer hızı
- Maksimum dikey hız
- Maksimum tırmanma/iniş hızı

### **Mesafe:**
- Toplam kat edilen mesafe (GPS bazlı)

### **Attitude (Duruş):**
- Maksimum roll, pitch, yaw açıları
- Ortalama ve standart sapma

### **GPS Kalitesi:**
- Görünen uydu sayısı
- GPS doğruluk değerleri (HDOP)

### **Batarya:**
- Voltaj değişimi
- Akım tüketimi
- Tüketilen batarya yüzdesi

## 🔧 Gelişmiş Kullanım

### **Belirli Mesaj Tiplerini Yükle:**
```python
data = loader.load(message_types=['GPS', 'ATTITUDE', 'BATTERY_STATUS'])
```

### **Veri Temizleme:**
```python
processor = FlightDataProcessor(data)
cleaned = processor.clean_data(
    remove_outliers=True,
    interpolate_gaps=True,
    max_gap_size=10
)
```

### **Low-pass Filtre Uygula:**
```python
filtered = processor.apply_lowpass_filter(
    msg_type='GPS',
    column='Alt',
    cutoff_freq=2.0,
    fs=10.0
)
```

### **Mesafe Hesapla:**
```python
distance = processor.calculate_distance_traveled()
print(f"Toplam mesafe: {distance:.2f} metre")
```

## 📁 Örnek Kullanım Senaryoları

### **Senaryo 1: Hızlı Analiz**
```bash
python3 cli_analyzer.py process ucus.tlog --output-dir sonuclar
```
→ Tüm analizler `sonuclar/` klasöründe

### **Senaryo 2: Sadece İstatistikler**
```python
loader = FlightDataLoader('ucus.tlog')
data = loader.load()
analyzer = FlightAnalyzer(data)
stats = analyzer.get_statistics()
print(stats['altitude_max_m'])
print(stats['speed_ground_max_kmh'])
```

### **Senaryo 3: Sadece 3D Görselleştirme**
```python
loader = FlightDataLoader('ucus.tlog')
data = loader.load()
visualizer = FlightVisualizer(data)
visualizer.plot_flight_path_3d_interactive(save_path='ucus3d.html', show=True)
```

### **Senaryo 4: Karşılaştırmalı Analiz**
```python
# İki farklı uçuş karşılaştırması
data1 = FlightDataLoader('ucus1.tlog').load()
data2 = FlightDataLoader('ucus2.tlog').load()

stats1 = FlightAnalyzer(data1).get_statistics()
stats2 = FlightAnalyzer(data2).get_statistics()

print(f"Uçuş 1 max hız: {stats1['speed_ground_max_kmh']} km/h")
print(f"Uçuş 2 max hız: {stats2['speed_ground_max_kmh']} km/h")
```

## ❓ Sorun Giderme

### **"FileNotFoundError" hatası:**
```python
# Dosya yolunu kontrol edin
import os
print(os.path.exists('ucus.tlog'))  # True dönmeli
```

### **"No data loaded" hatası:**
```python
# Dosyanın geçerli bir .tlog veya .bin dosyası olduğundan emin olun
is_valid = FlightDataLoader.is_valid_file('ucus.tlog')
print(f"Geçerli dosya mı? {is_valid}")
```

### **Grafik gösterilmiyor:**
```python
# show=True parametresini kullanın
visualizer.plot_altitude_profile(show=True)
```

## 📚 Daha Fazla Bilgi

- Ana README: `README.md`
- Örnek kod: `pixhawk_flight_analyzer/examples/example_usage.py`
- Test dosyaları: `pixhawk_flight_analyzer/tests/`

## 🚀 Hızlı Başlangıç (5 Dakikada)

```bash
# 1. Test scriptini düzenle
nano test_analyzer.py
# → flight_file = 'KENDI_DOSYANIZ.tlog' satırını düzenleyin

# 2. Çalıştır
python3 test_analyzer.py

# 3. Sonuçları görüntüle
ls -lh *.png *.html *.csv
```

---

**Happy Flying!** 🚁✨
