```text

N-QUEENS YAPISAL DÜZENLERİ PROJESİ
(Discovering Structural Regularities in the N-Queens Solution Space Using Machine Learning)

PROJE AMACI

Bu projenin amacı, N-Queens probleminin çözüm uzayında insan tarafından doğrudan fark edilmesi zor olan yapısal düzenleri (pattern / regularities) keşfetmektir.

Odak noktası klasik algoritmalarla çözüm üretmek değil, çözüm uzayının geometrik ve yapısal özelliklerini istatistiksel yöntemler ve makine öğrenmesi yardımıyla anlamaktır.

Makine öğrenmesi bu projede kesin çözüm üretmek için değil, çözüm uzayındaki gizli yapıları ve olası matematiksel sezgileri keşfetmek için bir araç olarak kullanılmaktadır.

⸻

PROJE YAPISI

analysis_outputs
├── global_structure_summary
│   ├── pca_variance_vs_N.png
│   ├── silhouette_vs_N.png
│   └── summary_metrics.csv
│
└── per_N_solution_space_analysis
├── N4
├── N5
├── N6
├── N7
├── N8
├── N9
├── N10
├── N11
├── N12
├── N13
├── N14
└── N15

scripts
├── generate_solution_space.py
├── extract_handcrafted_features.py
├── autoencoder_analysis.py
└── pca_clustering_analysis.py

solution_space
├── raw_solutions
│   ├── nqueen_N4.csv
│   ├── nqueen_N5.csv
│   ├── …
│   └── nqueen_N15.csv
│
├── handcrafted_features
│   ├── features_n4.csv
│   ├── features_n5.csv
│   ├── …
│   ├── features_n15.csv
│   └── summary_features.csv
│
└── learned_latent_space
├── latent_N4.csv
├── latent_N5.csv
├── …
└── latent_N15.csv

project_log.md

Büyük veri dosyaları GitHub reposuna yüklenmemiştir.
Tüm veriler bu repository’de bulunan script’ler aracılığıyla yeniden üretilebilir yapıdadır.
Ham ve işlenmiş veri setleri ayrıca Kaggle üzerinden paylaşılmaktadır.

⸻

	1.	ADIM – N-QUEENS ÇÖZÜM UZAYININ ÜRETİLMESİ

N = 4 ile N = 15 arasındaki tüm N-Queens çözümleri klasik backtracking algoritması ile üretilmiştir.

Her çözüm uzunluğu N olan bir vektörle temsil edilmiştir.
İndeks sütunu, değer vezirin bulunduğu satırı temsil eder.

Simetri eliminasyonu uygulanmıştır.

90°, 180°, 270° döndürmeler
Ayna yansıması

Her çözüm için 8 simetrik varyasyon üretilmiş, leksikografik olarak en küçük olan çözüm kanonik temsil olarak seçilmiştir.
Aynı kanonik temsile sahip çözümler tek çözüm olarak kabul edilmiştir.

Bu adımın sonunda simetrilerden arındırılmış, temiz ve ham çözüm uzayı elde edilmiştir.

Çıktı:
solution_space/raw_solutions/nqueen_N*.csv

⸻

	2.	ADIM – İNSAN TANIMLI FEATURE ÇIKARIMI

Ham çözümler üzerinden insan sezgisine dayalı, tamamen açıklanabilir özellikler çıkarılmıştır.

Amaç:
Çözüm uzayını istatistiksel ve yapısal olarak incelemek
Otomatik feature öğrenimiyle karşılaştırılabilecek bir referans tabanı oluşturmak

Çıkarılan feature grupları:

Temel istatistikler
row_mean, row_std, row_min, row_max, row_range

Merkez dağılımı
center_dist_mean, center_dist_std

Komşu sütun ilişkileri
adj_diff_mean, adj_diff_std, adj_diff_max, adj_diff_min

Global yapı / monotonluk
increasing_pairs_ratio, decreasing_pairs_ratio, flat_pairs_ratio

Dağılım şekli
row_skewness, row_kurtosis

Konumsal dağılım
unique_rows_ratio, even_row_ratio, odd_row_ratio

Konumsal enerji ölçümleri
positional_energy, diagonal_energy

Bu adımda makine öğrenmesi kullanılmamıştır.
Tüm feature’lar bilinçli olarak insan tarafından tanımlanmıştır.

Çıktı:
solution_space/handcrafted_features/features_n*.csv

⸻

	3.	ADIM – AUTOENCODER İLE OTOMATİK FEATURE ÖĞRENİMİ

Bu adımda insan tanımlı feature’lar bilinçli olarak kullanılmamış, modele yalnızca ham çözümler verilmiştir.

Her N değeri bağımsız ele alınmıştır.
Her N için ayrı bir autoencoder eğitilmiştir.

Autoencoder yapısı:

Girdi: [c0, c1, …, c(N-1)]
Çıkış: aynı vektör
Orta katman: model tarafından öğrenilen latent feature uzayı

solution_id modele dahil edilmemiştir.
N değeri dosya isminden alınmıştır.
Eğitim süreci sıralı ve progress bar ile izlenmiştir.

Autoencoder bu projede yalnızca feature extractor olarak kullanılmıştır.

Çıktı:
solution_space/learned_latent_space/latent_N*.csv

⸻

	4.	ADIM – PCA VE CLUSTERING ANALİZİ

Her N için öğrenilen latent uzay PCA ile 2 boyuta indirgenmiştir.
Açıklanan varyans oranları hesaplanmıştır.

KMeans clustering uygulanmıştır.
Silhouette score hesaplanmıştır.

Büyük N değerleri için silhouette hesaplaması hibrit yapılmıştır.

Küçük N → tüm veri
Büyük N → büyük ama sınırlı örnekleme

Her N için PCA scatter plot oluşturulmuştur.
Global olarak N’ye karşı PCA variance ve silhouette grafikleri üretilmiştir.

Çıktı:
analysis_outputs/per_N_solution_space_analysis/
analysis_outputs/global_structure_summary/

⸻

DENEY AYARLARI

N aralığı: 4 – 15
Simetri eliminasyonu: Var (8 dönüşüm)

Autoencoder:
Epoch: 300
Batch size: 64
Optimizer: Adam
Loss: MSE
Latent boyut: N * 0.5

Modeller saklanmamıştır.
Yalnızca latent feature çıktıları kaydedilmiştir.

⸻

MEVCUT DURUM

Ham çözüm uzayları üretilmiştir.
İnsan tanımlı feature’lar çıkarılmıştır.
Autoencoder eğitimleri tamamlanmıştır.
Her N için latent feature uzayları elde edilmiştir.
PCA ve clustering analizleri gerçekleştirilmiştir.

⸻

DATA AVAILABILITY

The full dataset used in this project is publicly available on Kaggle.

Kaggle Dataset:
https://www.kaggle.com/datasets/metineren/n-queens-solution-space-structural-features

The dataset includes:
- Symmetry-free N-Queens raw solution spaces (N = 4–15)
- Handcrafted statistical and structural features
- Autoencoder-learned latent representations

All data can be fully regenerated using the scripts provided in this repository.
```
