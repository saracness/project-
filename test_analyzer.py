#!/usr/bin/env python3
"""
Pixhawk Flight Analyzer - Basit Kullanım Örneği
"""

import sys
sys.path.insert(0, '/home/user/project-')

from pixhawk_flight_analyzer import (
    FlightDataLoader,
    FlightDataProcessor,
    FlightAnalyzer,
    FlightVisualizer
)

def main():
    # Buraya kendi .tlog veya .bin dosyanızın yolunu yazın
    flight_file = 'ornekucus.tlog'  # BURAYA DOSYA YOLUNUZU YAZIN

    print("\n" + "="*60)
    print("🚁 Pixhawk Flight Analyzer")
    print("="*60)

    # 1. Dosyayı Yükle
    print("\n📥 Flight log dosyası yükleniyor...")
    loader = FlightDataLoader(flight_file)
    data = loader.load()

    print(f"✅ {len(data)} farklı mesaj tipi yüklendi")

    # 2. İstatistikleri Hesapla
    print("\n📊 İstatistikler hesaplanıyor...")
    analyzer = FlightAnalyzer(data)
    stats = analyzer.get_statistics()

    # İstatistikleri göster
    analyzer.print_summary(stats)

    # CSV olarak kaydet
    analyzer.export_statistics_to_csv('ucus_istatistikleri.csv', stats)
    print("\n💾 İstatistikler 'ucus_istatistikleri.csv' dosyasına kaydedildi")

    # 3. Görselleştirmeler Oluştur
    print("\n🎨 Görselleştirmeler oluşturuluyor...")
    visualizer = FlightVisualizer(data)

    # 2D uçuş yolu
    visualizer.plot_flight_path_2d(
        save_path='ucus_yolu_2d.png',
        show=False
    )
    print("  ✅ 2D uçuş yolu: ucus_yolu_2d.png")

    # 3D interaktif uçuş yolu
    visualizer.plot_flight_path_3d_interactive(
        save_path='ucus_yolu_3d.html',
        show=False
    )
    print("  ✅ 3D interaktif: ucus_yolu_3d.html")

    # Yükseklik profili
    visualizer.plot_altitude_profile(
        save_path='yukseklik_profili.png',
        show=False
    )
    print("  ✅ Yükseklik profili: yukseklik_profili.png")

    # Hız profili
    visualizer.plot_speed_profile(
        save_path='hiz_profili.png',
        show=False
    )
    print("  ✅ Hız profili: hiz_profili.png")

    # Dashboard
    visualizer.plot_dashboard(
        save_path='dashboard.png',
        show=False
    )
    print("  ✅ Dashboard: dashboard.png")

    print("\n" + "="*60)
    print("✨ Analiz tamamlandı!")
    print("="*60)
    print("\n📁 Oluşturulan dosyalar:")
    print("  - ucus_istatistikleri.csv")
    print("  - ucus_yolu_2d.png")
    print("  - ucus_yolu_3d.html (tarayıcıda açın)")
    print("  - yukseklik_profili.png")
    print("  - hiz_profili.png")
    print("  - dashboard.png")
    print()

if __name__ == '__main__':
    main()
