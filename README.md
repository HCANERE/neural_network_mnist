# NumPy-Only Neural Network from Scratch (MNIST)

Bu proje, herhangi bir derin öğrenme kütüphanesi (TensorFlow, PyTorch vb.) kullanmadan, sadece **NumPy** kütüphanesi kullanılarak sıfırdan 
inşa edilmiş bir Yapay Sinir Ağı (ANN) uygulamasıdır. Proje, popüler MNIST el yazısı rakamlar veri seti
üzerinde eğitim yaparak rakam tanıma işlemini gerçekleştirir.

## 🚀 Proje Özellikleri

*   **Sıfırdan Matematiksel Model:** İleri besleme (Forward Pass) ve Geri yayılım (Backpropagation) algoritmaları tamamen lineer cebir temelleriyle kodlanmıştır.
*   **Aktivasyon Fonksiyonları:** Gizli katmanda **ReLU**, çıkış katmanında **Sigmoid** kullanılarak non-lineerite sağlanmıştır.
*   **Kayıp Fonksiyonu:** Model başarısını ölçmek ve gradyanları hesaplamak için **Binary Cross-Entropy** kayıp fonksiyonu entegre edilmiştir.
*   **Modüler Mimari:** Sinir ağı mantığı bir Python sınıfı (`class`) altında toplanarak okunabilirlik ve yeniden kullanılabilirlik artırılmıştır.

## 🧠 Ağ Mimarisi

Ağ tasarımı şu şekilde kurgulanmıştır:
*   **Giriş Katmanı:** 784 Nöron (28x28 piksellik MNIST görüntüleri).
*   **Gizli Katman:** 128 Nöron (Özellik çıkarımı ve karmaşık örüntülerin tespiti için).
*   **Çıkış Katmanı:** 10 Nöron (0-9 arası her bir rakam için olasılık değerleri).

Eğitim Performansı
Model, 50 epoch sonunda Full-Batch Gradient Descent yaklaşımıyla şu sonuçları vermektedir:

Başlangıç Kaybı (Loss): ~6.93

Final Kaybı (Loss): ~2.94

Eğitim Doğruluğu (Accuracy): %65+

Geliştirme Önerisi: Eğitim sürecini Mini-Batch Gradient Descent ve Softmax aktivasyonu ile güncelleyerek doğruluk oranını %98+ seviyelerine çıkarabilirsiniz.

📂 Dosya Yapısı
neural_network.py: Ağırlıkların başlatılması, aktivasyon fonksiyonları, ileri ve geri yayılım metodlarını içeren sınıf dosyası.

train.py: Veri ön işleme, normalizasyon, One-Hot Encoding ve ana eğitim döngüsünü içeren dosya.

👨‍💻 Geliştirici
Hüsnü Caner Eyüpoğlu

Selçuk Üniversitesi - Bilgisayar Mühendisliği Bölümü
