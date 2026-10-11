# 🧠 Beyin Tümörü Tespiti ve Sınıflandırması (Deep Learning & XAI)

Bu proje, Manyetik Rezonans (MR) görüntülerinden beyin tümörlerinin tespit edilmesi, sınıflandırılması ve **Açıklanabilir Yapay Zeka (XAI - Explainable AI)** teknikleriyle teşhisin görsel olarak açıklanmasını sağlayan uçtan uca bir derin öğrenme ve masaüstü arayüz sistemidir.

Projede **ResNet18 + CBAM (Convolutional Block Attention Module)** tabanlı derin öğrenme mimarisi, **Hibrit CNN-SVM** karar mekanizması, **Grad-CAM** görsel ısı haritası üretimi ve **C# .NET Windows Forms** tabanlı modern bir masaüstü kullanıcı arayüzü entegre edilmiştir.

---

## 📌 Temel Özellikler

1. **İki Aşamalı Hiyerarşik Teşhis:**
   - **Aşama 1 (Binary Sınıflandırma):** Görüntüde tümör olup olmadığının tespiti (*Sağlıklı / Tümörlü*).
   - **Aşama 2 (Çok Sınıflı Sınıflandırma):** Tümör tespit edildiğinde türünün belirlenmesi (*Glioma, Meningioma, Pituitary*).
2. **CBAM Dikkat Mekanizması:**
   - Standart ResNet18 katmanlarına Kanal (Channel) ve Mekansal (Spatial) Dikkat modülleri eklenerek modelin patolojik dokulara odaklanması sağlanmıştır.
3. **Açıklanabilir Yapay Zeka (Grad-CAM):**
   - Modelin hangi piksel ve doku bölgelerine bakarak karar verdiğini gösteren renkli ısı haritaları (Heatmap).
4. **Hibrit Karar Mekanizması (CNN + SVM):**
   - Derin öğrenme modelinden çıkarılan öznitelikler (features), Destek Vektör Makineleri (SVM) ile sınıflandırılarak doğruluk oranları karşılaştırılmıştır.
5. **C# .NET Masaüstü Arayüzü (TumorUI):**
   - Kullanıcı dostu masaüstü uygulaması ile MR görseli yükleme, model sorgulama ve Grad-CAM ısı haritasını anlık olarak inceleme imkanı.
6. **ONNX Entegrasyon Desteği:**
   - Eğitilen PyTorch modelleri ONNX formatına dönüştürülerek platformlar arası çalışmaya hazır hale getirilmiştir.

---

## 🏗️ Sistem Mimarisi ve İş Akışı

```text
       [ Hasta MR Görüntüsü ]
                 │
                 ▼
   ┌─────────────────────────────┐
   │ Aşama 1: İkili Sınıflandırma│
   │      (ResNet18 + CBAM)      │
   └─────────────┬───────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   [ SAĞLIKLI ]      [ TÜMÖR TESPİT EDİLDİ ]
                          │
                          ▼
            ┌───────────────────────────┐
            │ Aşama 2: Çoklu Sınıflama  │
            │(Glioma/Meningioma/Pituitary)
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │ Grad-CAM Açıklanabilirlik │
            │      (Isı Haritası)       │
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │   C# Masaüstü Arayüzü     │
            │   (Sonuç + Görsel Rapor)  │
            └───────────────────────────┘
```

---

## 📁 Proje Dizin Yapısı

```text
örüntüTanımaProje/
├── .gitignore               # Git tarafından takip edilmeyecek dosyalar
├── requirements.txt         # Gerekli Python kütüphaneleri
├── README.md                # Proje tanıtım ve kurulum dokümantasyonu
├── Rapor/                   # Proje raporu ve akademik dokümanlar
│   └── ProjeRaporu.docx
└── kod/
    ├── config.py            # Hiperparametreler, dizin ve sınıf tanımları
    ├── dataset.py           # Veri seti yükleyicileri (Kaggle & MATLAB)
    ├── model.py             # ResNet18 + CBAM dikkat mimarisi
    ├── train.py             # Model eğitim döngüleri
    ├── evaluate.py          # Metrik hesaplama, Karmaşıklık matrisi çizimi
    ├── gradcam.py           # Grad-CAM ısı haritası algoritması
    ├── hybrid.py            # Hibrit CNN + SVM öznitelik çıkarımı ve eğitimi
    ├── inference.py         # C# arayüzü ve CLI için test scripti (JSON çıktı)
    ├── main.py              # Tüm eğitim ve değerlendirme sürecini başlatan ana dosya
    ├── results/             # Eğitim ve test grafik çıktıları
    │   ├── confusion_matrix_binary.png
    │   ├── confusion_matrix_multiclass.png
    │   ├── training_history_binary.png
    │   ├── training_history_multiclass.png
    │   └── sample_gradcam.png
    ├── saved_models/        # Eğitilmiş PyTorch (.pth) ve ONNX modelleri
    └── TumorUI/             # C# Windows Forms Masaüstü Uygulaması
        ├── Form1.cs
        ├── Program.cs
        └── TumorUI.csproj
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/receptayyipyanik/brain-tumor-detection-xai.git
cd brain-tumor-detection-xai
```

### 2. Python Ortamını Hazırlayın
Önerilen Python sürümü: `3.10` - `3.12`

```bash
# Sanal ortam oluşturma (isteğe bağlı)
python -m venv venv
venv\Scripts\activate      # Windows için

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 3. Veri Setinin Temini
Model eğitiminde popüler **Brain Tumor MRI Dataset** kullanılmıştır:
- Kaggle üzerinden indirdiğiniz veri setini `kod/archive (1)` klasörüne çıkarınız. Klasör yapısı aşağıdaki gibi olmalıdır:
  ```text
  kod/archive (1)/
  ├── Training/
  │   ├── glioma/
  │   ├── meningioma/
  │   ├── notumor/
  │   └── pituitary/
  └── Testing/
      ├── glioma/
      ├── meningioma/
      ├── notumor/
      └── pituitary/
  ```

### 4. Modelleri Eğitme
Tüm ikili, çoklu ve hibrit model eğitimlerini başlatmak için:
```bash
python kod/main.py
```
Eğitim tamamlandığında ağırlıklar `kod/saved_models/` klasörüne, performans grafikleri ise `kod/results/` klasörüne kaydedilir.

### 5. Komut Satırından Test (Inference)
Tek bir MR görüntüsü üzerinde tahmin yapmak ve Grad-CAM ısı haritası üretmek için:
```bash
python kod/inference.py --image "test_mr.jpg"
```
Çıktı formatı:
```json
{
  "has_tumor": true,
  "tumor_type": "glioma",
  "heatmap_path": "c:\\...\\kod\\results\\current_heatmap.png"
}
```

### 6. C# Masaüstü Arayüzünü Çalıştırma
1. Visual Studio veya `dotnet run` ile `kod/TumorUI` projesini açın.
2. `F5` ile uygulamayı başlatın.
3. Uygulama açıldığında **"Görsel Yükle"** butonuna basarak bir MR görseli seçin ve **"Sorgula"** butonuna tıklayın.
4. İlk çalıştırmada gerekirse sağ üstteki **"⚙️ Python Seç"** butonundan sisteminizdeki `python.exe` dosyasını seçebilirsiniz.

---

## 📊 Deneysel Sonuçlar

| Model Türü | Sınıflandırma | Doğruluk (Accuracy) | Kullanılan Teknik |
|:---|:---:|:---:|:---|
| **ResNet18 + CBAM** | Binary (Tümör Var/Yok) | %98+ | Transfer Learning + Attention |
| **ResNet18 + CBAM** | Multiclass (3 Sınıf) | %95+ | Transfer Learning + Attention |
| **Hibrit CNN + SVM** | Binary | %97+ | Feature Extraction + RBF SVM |
| **Grad-CAM** | Açıklanabilirlik | - | Görsel Isı Haritası |

---

## 🛠️ Projeyi Geliştirme Önerileri (Roadmap)

- [ ] **ONNX Runtime C# Doğrudan Entegrasyonu:** C# arayüzünde Python subprocess çağrısı yerine `Microsoft.ML.OnnxRuntime` NuGet paketiyle sıfır Python bağımlılığı ve milisaniyelik yanıt süresi.
- [ ] **Web Arayüzü & Canlı Demo:** Streamlit veya Gradio ile tarayıcı tabanlı demo hazırlanıp Hugging Face Spaces üzerinde ücretsiz yayınlanması.
- [ ] **Semantik Segmentasyon:** U-Net / nnU-Net eklenerek tümör bölgesinin piksel piksel maskelenmesi ve tümör hacim tahmini.
- [ ] **Klinik PDF Rapor Çıktısı:** Teşhis sonucu, güven skoru ve ısı haritasını içeren otomatik PDF doktor raporu oluşturma.
- [ ] **Modern Mimariler:** Vision Transformer (ViT) ve EfficientNet-V2 modelleri ile kıyaslama analizleri.

---

## 👥 Proje Ekibi
- Recep Tayyip Yanık
- Samet Pulat
- Furkan Kubat

## 📄 Lisans
Bu proje [MIT Lisansı](LICENSE) kapsamında açık kaynak olarak paylaşılmıştır.
