<div align="center">

<img src="docs/images/image1.png" alt="Proje Logo" width="160"/>

# 🧠 MRI Görüntülerinden Beyin Tümörü Tespiti ve Sınıflandırılması
### Klinik Karar Destek Sistemi & Açıklanabilir Yapay Zeka (XAI)

**Ders:** Örüntü Tanıma  
**Öğretim Üyesi:** Doç. Dr. Yasemin Gültepe  

**Hazırlayanlar:**  
**Recep Tayyip Yanık** (230710017) &nbsp;|&nbsp; **Samet Pulat** (230710055) &nbsp;|&nbsp; **Furkan Kubat** (220710093)

---

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![C# .NET](https://img.shields.io/badge/.NET-9.0-512BD4.svg?logo=dotnet&logoColor=white)](https://dotnet.microsoft.com/)
[![ONNX](https://img.shields.io/badge/ONNX-Supported-005CED.svg?logo=onnx&logoColor=white)](https://onnx.ai/)
[![License: Educational Use Only](https://img.shields.io/badge/License-Educational%20Use%20Only-yellow.svg)](LICENSE)

</div>

---

## 📋 ÖZET

Bu proje, Manyetik Rezonans Görüntüleme (MRI) verileri üzerinden beyin tümörlerinin otonom tespiti ve sınıflandırılması amacıyla derin öğrenme tabanlı, uçtan uca bir **Klinik Karar Destek Sistemi** geliştirmeyi hedeflemektedir. Modelin tek bir hastane veya cihaz formatını ezberlemesini önlemek amacıyla projede **iki farklı kaynaktan alınan veri setleri (Kaggle resim veri seti ve Figshare `.mat` formatlı medikal veri seti)** birleştirilerek kullanılmıştır. Sistem, tıbbi hata payını en aza indirmek için **2 Aşamalı bir süzgeç mekanizması** (Önce Tümör Var/Yok tespiti, ardından Glioma, Meningioma, Pituitary sınıflandırması) olarak tasarlanmıştır.

Mimari olarak, Transfer Learning destekli **ResNet18** ağı tercih edilmiş ve bu ağın içerisine insan beyninin odaklanma mantığını taklit eden **CBAM (Kanal ve Uzamsal Dikkat Mekanizması)** entegre edilmiştir. Modeller Google Colab üzerindeki bulut GPU'lar (NVIDIA T4) yardımıyla eğitilmiş ve İkili (Binary) sınıflandırmada **%98**, Çok Sınıflı (Multiclass) sınıflandırmada **%96** gibi yüksek doğruluk oranlarına ulaşılmıştır. Son olarak, doktorların yapay zekaya güvenini artırmak amacıyla **Grad-CAM (Açıklanabilir Yapay Zeka - Isı Haritası)** tekniği projeye dahil edilmiş ve tüm arka plan sistemi **C# Windows Forms** ile tasarlanan kullanıcı dostu bir masaüstü arayüzüne entegre edilerek eksiksiz bir yazılım ürünü haline getirilmiştir.

---

## 🎯 1. Projenin Amacı ve Tasarım Mantığı (Neden 2 Aşamalı Yaptık?)

Bu projede amacımız sadece bir resme bakıp sınıflandırma yapan sıradan bir model yapmak değildi. Sistemi gerçek bir doktorun düşünme yapısına benzetmek istedik. Bu yüzden projeyi **2 Aşamalı (Süzgeçli) bir pipeline** olarak tasarladık:

1. **1. Aşama (İkili / Binary):** Hastanın MR'ına bak ve sadece *"Tümör var mı, yok mu?"* sorusuna cevap ver.
2. **2. Aşama (Çok Sınıflı / Multiclass):** Eğer 1. aşama *"Evet tümör var"* dediyse, o zaman *"Peki bu hangi tür? (Glioma, Meningioma, Pituitary)"* diye sor.

> **Neden böyle yaptık?**  
> Çünkü sağlıklı bir beyni doğrudan 4 sınıfın içine sokup tahmin yapmaya çalışmak modelin kafasını karıştırıyor ve doğruluğu düşürüyordu. 2 aşamalı sistemle başarı oranımızı %98 - %99'lara kadar çıkardık.

```text
       [ Hasta MR Görüntüsü ]
                 │
                 ▼
   ┌─────────────────────────────┐
   │ Aşama 1: İkili Sınıflandırma│  ──►  [ SAĞLIKLI ] (Tümör Yok)
   │      (ResNet18 + CBAM)      │
   └─────────────┬───────────────┘
                 │ (Tümör Varsa)
                 ▼
   ┌─────────────────────────────┐
   │ Aşama 2: Çoklu Sınıflama    │  ──►  [ Glioma / Meningioma / Pituitary ]
   │ (Tür Tespiti: ResNet+CBAM)  │
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │ Grad-CAM Açıklanabilir AI   │  ──►  Renkli Isı Haritası (Heatmap)
   └─────────────┬───────────────┘
                 │
                 ▼
   ┌─────────────────────────────┐
   │     C# Masaüstü Arayüzü     │  ──►  Eşzamanlı Teşhis & Isı Haritası Görseli
   └─────────────────────────────┘
```

---

## 🛠️ 2. Kullanılan Kütüphaneler ve Çerçeveler (Frameworks)

Projede en güncel ve endüstri standardı olan kütüphaneler kullanılmıştır:
- **PyTorch & Torchvision:** Derin öğrenme mimarisini (ResNet) kurmak, modelleri eğitmek ve tensör işlemleri yapmak için (TensorFlow yerine daha dinamik ve akademik araştırmalarda daha sık kullanıldığı için PyTorch seçildi).
- **Scikit-learn:** Hibrit model (CNN + SVM) denemesindeki Destek Vektör Makineleri (SVM) ve başarım metriklerinin hesaplanması için.
- **ONNX:** Eğittiğimiz PyTorch modelini, Python'dan bağımsız olarak C# arayüzünde hızlı ve hatasız çalıştırabilmek adına evrensel formata dönüştürmek için.
- **PIL & OpenCV & Matplotlib & Seaborn:** Medikal görüntüleri okumak, veri dönüşümleri yapmak, ısı haritalarını (Grad-CAM) süperempoze etmek ve sonuç grafiklerini çizdirmek için.

---

## 🔬 3. Mühendislik Çözümleri: MATLAB (.mat) Entegrasyonu ve Özel Veri Okuyucu

*(Projede 2 Farklı Veri Seti Birleştirildi)*

Projedeki en büyük teknik zorluklardan biri, Kaggle'daki normal `.jpg` resimleri ile Figshare'deki tıbbi `.mat` (MATLAB) formatlı verileri aynı anda eğitime sokabilmekti. `.mat` dosyaları normal bir resim gibi okunamaz; içlerinde `cjdata.image` (piksel matrisi) ve `cjdata.label` (etiket) gibi matematiksel diziler barındırır. Bu sorunu çözmek için projemize **Özel Bir PyTorch Veri Sınıfı (`MatlabTumorDataset`)** kodladık:

1. Yazdığımız özel algoritma, her bir `.mat` dosyasını `h5py` ile açıp içindeki ondalıklı sayı (float) matrislerini okudu.
2. Bu dizileri **Min-Max Normalizasyonu** ile 0-255 piksel aralığına çevirip Derin Öğrenme ağının anlayabileceği Tensörlere dönüştürdü.
3. Bu mühendislik dokunuşu sayesinde modelimizi çok daha profesyonel (ham) tıbbi verilerle besleyebildik.

---

## ⚡ 4. Donanım ve Eğitim Ortamı: Neden Google Colab Kullanıldı?

Derin Öğrenme modelleri devasa matris çarpımları gerektirdiği için standart bilgisayar işlemcilerinde (CPU) eğitilmeleri saatler sürebilir. Bu darboğazı aşmak için eğitim sürecimizi **Google Colab** bulut ortamına taşıdık:

- **GPU Hızlandırması (NVIDIA T4):** Colab'ın ücretsiz Tensor Core GPU'ları sayesinde, 10 binden fazla medikal görüntü üzerindeki eğitim sürecini dakikalar içinde (Epoch başına ~2 dakika) tamamlayabildik.
- **Donanım Bağımsızlığı:** Ağırlıklı matematiksel hesaplama yükü gerektiren "Eğitim" (Training) aşamasını bulutta hallettik. Öğrenme işlemi bittikten sonra "akıllanmış" modellerimizi (`.pth` ve `.onnx`) masaüstümüze indirdik ve sadece "Tahmin" (Inference) işlemini yapacak olan C# arayüzümüze entegre ettik. Böylece uygulamayı kullanacak olan son kullanıcının güçlü bir bilgisayara sahip olma zorunluluğunu ortadan kaldırdık.

---

## 🧹 5. Veri Ön İşleme ve Zenginleştirme (Data Preprocessing & Augmentation)

### Neleri, Neden Yaptık?
- **Yeniden Boyutlandırma (Resize 224x224):** ResNet18 mimarisinin giriş katmanı tam olarak bu boyutları beklediği için tüm görselleri bu standarta çektik.
- **Normalizasyon (Min-Max ve Z-Score):** Görüntülerdeki parlaklık farklarını ortadan kaldırmak için `mean=[0.485, 0.456, 0.406]` ve `std=[0.229, 0.224, 0.225]` değerleriyle normalizasyon yaptık. Karanlık çekilmiş bir MR ile aydınlık MR, modelin gözünde aynı matematiksel aralığa oturdu.
- **Veri Artırma (Data Augmentation):** Elimizdeki veriyi rastgele döndürdük, yatay çevirdik ve renkleriyle oynadık. Model hep düz duran bir beyni ezberlemesin, beyin ters veya yan durduğunda da tümörü bulabilsin (Overfitting'i engellemek) için.

### Neleri YAPMADIK ve Neden?
- **Kafatası Soyma (Skull Stripping) YAPMADIK:** Gerçek hayatta hastanede doktor bu programı kullanırken önce kafatasını silmekle uğraşmamalıdır. Modelimizi, ham MR görüntüsünden kemiği ve dokuyu ayırt edebilecek kadar "zor şartlarda" eğittik.
- **Gürültü Azaltma (Median Filter vb.) YAPMADIK:** Aşırı bulanıklaştırma yapan filtreler, tümörün ince kenar hatlarını silebilir. CNN katmanları zaten gürültüleri filtreleyecek matematiksel derinliğe sahiptir.

---

## 📊 6. Makine Öğrenmesi Disiplinleri: Train/Test, Epoch ve Shuffle Mantığı

Modelin gerçekten öğrenip öğrenmediğini (ezber yapıp yapmadığını) kanıtlamak için bilimsel makine öğrenmesi standartlarını harfiyen uyguladık:

- **Neden Veriyi Eğitim (Train) ve Test (Validation) Olarak Ayırdık?**  
  Veri setimizin **%80**'ini eğitime ayırırken, geriye kalan **%20**'lik (Test/Doğrulama) kısmını modelden **tamamen sakladık**. Modelin başarısını (%98 ve %96), daha önce hayatında hiç görmediği bu Test verilerindeki performansına bakarak ölçtük. Böylece modelin piksel ezberlemediğini, tümörün "özelliklerini (feature)" öğrendiğini kanıtladık.
- **Neden Veriyi "Epoch"lara Böldük?**  
  Bir **Epoch**, modelin tüm veri setini baştan sona tam 1 kez görmesi demektir. Eğitimi 20 ve 30 Epoch'lara yaydık. Her bir Epoch'ta model, yaptığı tahminlerdeki hatalarını (Loss) hesaplayıp, ağırlıklarını (Weights) milimetrik olarak düzelterek bir sonraki tura daha akıllı girdi.
- **Neden "Shuffle" (Karıştırma) Yaptık?**  
  Veri yükleyicimizde `shuffle=True` parametresini kullandık. Eğer modele önce art arda 1000 tane sağlıklı MR, sonra 1000 tane tümörlü MR verseydik, modelin karar mekanizması tekdüzeleşirdi. Verileri her Epoch'ta tıpkı bir iskambil destesi gibi karıştırarak, modelin aynı anda hem sağlıklı hem tümörlü MR'ları kıyaslamasını sağladık.

---

## 🧠 7. Derin Öğrenme Mimarisi: Neden ResNet18, CBAM ve Dropout(0.5)?

- **Neden ResNet18?** Çok derin modeller (ResNet50 vb.) medikal verilerde çok çabuk ezber (overfitting) yapar. ResNet18'in Residual (Atlama) bağlantıları sayesinde özellikler kaybolmadan aktarılır. Ayrıca, `IMAGENET1K_V1` ağırlıklarını (Transfer Learning) alıp medikal verilere ince ayar (fine-tuning) yaptık.
- **CBAM (Dikkat Mekanizması - Attention) Eklemesi:** Hazır modeli alıp geçmedik. Modelin içine **CBAM (Kanal ve Uzamsal Dikkat)** modülü kodladık. Tıpkı bir insanın MR'a bakarken simsiyah boşluklara değil de gri lekeli yerlere "odaklanması" gibi, modelin sadece tümör olabilecek noktalara odaklanmasını sağladık.
- **Neden Dropout(0.5) ile Nöronları Kestik?** Modelin çıkış katmanına `nn.Dropout(p=0.5)` ekledik. Bu, eğitim sırasında **nöronların (sinirlerin) %50'sinin rastgele uyutulması** demektir. Eğer tüm nöronlar aktif olursa model kolaya kaçıp ezber yapar. Nöronların yarısını sürekli kapatarak, ağı "zor yoldan her durumu öğrenmeye" mecbur bıraktık.

---

## ⚙️ 8. Optimizasyon ve C# / ONNX Kod Mimari Çözümü

Eğitim sürecinde ağı optimize etmek için şu ayarları kullandık:
- **Loss Function:** Yanlış sınıfa sapan modeli en sert şekilde cezalandıran `CrossEntropyLoss` kullanıldı.
- **Optimizer:** Hız ve ivme momentini çok iyi ayarlayan `Adam` optimizasyonu seçildi (`lr=1e-4`, `weight_decay=1e-4`).
- **Teknik Bir Problem ve Çözümü (ONNX Export Uyumluluğu):**  
  PyTorch'ta eğittiğimiz modeli C#'ta çalışabilen ONNX formatına çevirirken, CBAM dikkat mekanizmasının içindeki `AdaptiveMaxPool2d` katmanının ONNX kütüphanesi tarafından desteklenmediği hatasıyla karşılaştık (Sistem Çökmesi). Mimariyi değiştirmek yerine, ilgili katmanı kod seviyesinde ONNX'in anlayabileceği ham `torch.max` matematiksel işlemlerine dönüştürerek (refactoring) çok ciddi bir platformlar arası uyumluluk (Cross-Platform Compatibility) problemini çözdük.

---

## 🧪 9. Hibrit Model (CNN + SVM) Denemesi ve Sonuçları

Özellik çıkarıcı olarak CNN'i (ResNet) alıp, sınıflandırma kararı için geleneksel **SVM (Destek Vektör Makinesi)** eğittik:
- **Sonuç:** ResNet kendi sınıflandırıcısıyla **%98 - %99** başarıya ulaşırken, SVM'li modelin başarısı **%55**'e düştü.
- **Bilimsel Çıkarım:** Tıbbi görüntüler gibi kompleks verilerde klasik makine öğrenmesinin (SVM) yetersiz kaldığını; uçtan uca (End-to-End) Derin Öğrenme mimarilerinin çok daha üstün olduğunu deneysel olarak kanıtladık.

---

## 🔍 10. Açıklanabilir Yapay Zeka (XAI) ve C# UI Entegrasyonu

Tıp dünyasında doktorlar kapalı kutu yapay zekalara güvenmez. Bu yüzden projeye **Grad-CAM** entegre ettik. Sistem karar verdiği zaman, ResNet'in son evrişim (Convolution) katmanından gradyanları çekerek bir **Isı Haritası** üretiyor. Yani *"Tümör burada, çünkü bu bölgeye odaklandım"* diyebiliyor. Tüm bu yapıyı son kullanıcının (doktorların) bilgisayarında kullanabilmesi için C# Windows Forms arayüzüne (UI) bağlayarak projeyi "gerçek bir ürüne" dönüştürdük.

### Uygulamanın Testi ve XAI (İnternet Üzerinden Alınan, Veri Setine Dahil Olmayan Bağımsız Testler)

Aşağıda internetten bağımsız olarak temin edilen ve modelin eğitiminde hiç yer almayan test MR görüntüleri üzerinde sistemin canlı çalışma ekranları görülmektedir:

<div align="center">

#### Test 1: Glioma Tümörü Tespiti ve Isı Haritası
<img src="docs/images/image2.png" alt="Test 1 Ekran Görüntüsü" width="90%"/>

<br/><br/>

#### Test 2: Meningioma Tümörü Tespiti ve Isı Haritası
<img src="docs/images/image3.png" alt="Test 2 Ekran Görüntüsü" width="90%"/>

<br/><br/>

#### Test 3: Pituitary Tümörü Tespiti ve Isı Haritası
<img src="docs/images/image4.png" alt="Test 3 Ekran Görüntüsü" width="90%"/>

<br/><br/>

#### Test 4: Sağlıklı Beyin Testi (Tümör Bulunamadı)
<img src="docs/images/image5.png" alt="Test 4 Ekran Görüntüsü" width="90%"/>

</div>

---

## 📈 11. Model Eğitimi Çıktıları ve Karmaşıklık Matrisleri (Confusion Matrix)

### Aşama 1: İkili (Binary) Sınıflandırma Sonuçları
- **Toplam Eğitim Seti:** 8051 MR Görüntüsü
- **Toplam Test Seti:** 2213 MR Görüntüsü
- **Test Doğruluğu (Accuracy):** **%98**

```text
               precision    recall  f1-score   support
    No Tumor        0.92      1.00      0.96       400
       Tumor        1.00      0.98      0.99      1813

    accuracy                            0.98      2213
   macro avg        0.96      0.99      0.97      2213
weighted avg        0.98      0.98      0.98      2213
```

<div align="center">
<table>
  <tr>
    <td align="center"><b>Binary Karmaşıklık Matrisi (Confusion Matrix)</b></td>
    <td align="center"><b>Binary Eğitim ve Doğrulama Kayıp/Başarım Grafiği</b></td>
  </tr>
  <tr>
    <td><img src="kod/results/confusion_matrix_binary.png" width="400"/></td>
    <td><img src="kod/results/training_history_binary.png" width="450"/></td>
  </tr>
</table>
</div>

---

### Aşama 2: Çok Sınıflı (Multiclass) Sınıflandırma Sonuçları
- **Toplam Eğitim Seti:** 6651 MR Görüntüsü
- **Toplam Test Seti:** 1813 MR Görüntüsü
- **Test Doğruluğu (Accuracy):** **%96**

```text
              precision    recall  f1-score   support
      glioma       0.99      0.90      0.94       677
  meningioma       0.91      0.97      0.94       549
   pituitary       0.96      1.00      0.98       587

    accuracy                            0.96      1813
   macro avg       0.96      0.96      0.96      1813
weighted avg       0.96      0.96      0.96      1813
```

<div align="center">
<table>
  <tr>
    <td align="center"><b>Çok Sınıflı Karmaşıklık Matrisi</b></td>
    <td align="center"><b>Çok Sınıflı Eğitim ve Doğrulama Kayıp/Başarım Grafiği</b></td>
  </tr>
  <tr>
    <td><img src="kod/results/confusion_matrix_multiclass.png" width="400"/></td>
    <td><img src="kod/results/training_history_multiclass.png" width="450"/></td>
  </tr>
</table>
</div>

---

## 💻 12. Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/receptayyipyanik/brain-tumor-detection-xai.git
cd brain-tumor-detection-xai
```

### 2. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatın
- **Masaüstü Arayüzü (C# Windows Forms):**
  ```powershell
  dotnet run --project kod\TumorUI
  ```
  *(veya `kod/TumorUI/bin/Debug/net9.0-windows/TumorUI.exe` dosyasına çift tıklayın)*
- **Terminalden Test:**
  ```bash
  python kod/inference.py --image "test_ornek.jpg"
  ```
- **Web Tarayıcısı Üzerinden (Gradio):**
  ```bash
  python kod/app_web.py
  ```

---

## 📜 13. Lisans ve Kullanım Koşulları

Bu yazılım ve ilgili tüm dosyalar **[EĞİTİM VE AKADEMİK ARAŞTIRMA LİSANSI (Educational & Non-Commercial Use Only)](LICENSE)** kapsamında sunulmaktadır.

> [!WARNING]
> - **Yalnızca Eğitim Amaçlıdır:** Bu proje; üniversite dersleri, akademik araştırmalar ve bireysel öğrenim amacıyla hazırlanmıştır.
> - **Ticari Kullanım Yasaktır:** Projenin veya modellerin doğrudan ya da dolaylı olarak herhangi bir ticari amaçla satılması, kiralanması veya dağıtılması kesinlikle yasaktır.
> - **Tıbbi Cihaz Değildir:** Bu yazılım klinik tanı veya tedavi amacıyla kullanılamaz; gerçek hastalar üzerinde teşhis kararı vermek için uygun değildir.
