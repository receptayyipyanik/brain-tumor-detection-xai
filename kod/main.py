#23071001-RecepTayyipYANIK/230710055-SametPulat/220710093-FurkanKubat
import os
# Kutuphane cakisip cokmesini engelliyoruz
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import config
from dataset import get_dataloaders
from model import get_model
from train import train_model
from evaluate import evaluate_model, plot_history
from gradcam import GradCAM, save_gradcam_image
from torchvision import transforms
from PIL import Image
import os
from hybrid import train_and_evaluate_hybrid

def main():
    # Ekran karti varsa seciyoruz yoksa islemcide kaliyoruz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # ==========================================
    # AŞAMA 1: BİNARY SINIFLANDIRMA
    # ==========================================
    print("\n--- AŞAMA 1: İkili Sınıflandırma Eğitimi Başlıyor ---")
    
    # Veri yukleyicileri ve modeli cagiriyoruz
    binary_train_loader, binary_test_loader = get_dataloaders(mode='binary')
    binary_model = get_model(num_classes=2).to(device)
    
    # Ikili siniflandirma modelimizi egitiyoruz
    binary_model, binary_history = train_model(
        binary_model, binary_train_loader, binary_test_loader, 
        epochs=config.EPOCHS_BINARY, device=device
    )
    
    # Sonuclari degerlendirip grafiklerini cizdiriyoruz
    evaluate_model(binary_model, binary_test_loader, device, mode='binary')
    plot_history(binary_history, mode='binary')
    
    # Egitilen agirliklari kaydediyoruz
    binary_model_path = os.path.join(config.MODEL_DIR, 'binary_model.pth')
    binary_onnx_path = os.path.join(config.MODEL_DIR, 'binary_model.onnx')
    torch.save(binary_model.state_dict(), binary_model_path)
    
    # C sharp ile haberlesmek icin modeli onnx formatinda disari aktariyoruz
    dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)
    torch.onnx.export(binary_model, dummy_input, binary_onnx_path, 
                      input_names=['input'], output_names=['output'])
    print(f"Binary model {binary_onnx_path} olarak kaydedildi.")


    print("\n--- AŞAMA 2: Çok Sınıflı Sınıflandırma Eğitimi Başlıyor ---")
    
    # Coklu siniflandirma icin veriyi ve modeli hazirliyoruz
    multi_train_loader, multi_test_loader = get_dataloaders(mode='multiclass')
    multi_model = get_model(num_classes=3).to(device)
    
    # Tumor tiplerini ayirt etmesi icin modeli egitiyoruz
    multi_model, multi_history = train_model(
        multi_model, multi_train_loader, multi_test_loader, 
        epochs=config.EPOCHS_MULTICLASS, device=device
    )
   
    # Degerlendirme ve cizim islerini yapiyoruz
    evaluate_model(multi_model, multi_test_loader, device, mode='multiclass')
    plot_history(multi_history, mode='multiclass')
    
    # Coklu modelin agirliklarini kaydediyoruz
    multi_model_path = os.path.join(config.MODEL_DIR, 'multiclass_model.pth')
    multi_onnx_path = os.path.join(config.MODEL_DIR, 'multiclass_model.onnx')
    torch.save(multi_model.state_dict(), multi_model_path)
    
    # Bu modeli de onnx olarak disari aliyoruz
    torch.onnx.export(multi_model, dummy_input, multi_onnx_path, 
                      input_names=['input'], output_names=['output'])
    print(f"Multiclass model {multi_onnx_path} olarak kaydedildi.")
    
 
    print("\n--- Grad-CAM Testi Yapılıyor ---")
  
    # Klasorden ornek bir resim seciyoruz
    sample_img_path = os.path.join(config.TEST_DIR, 'glioma', os.listdir(os.path.join(config.TEST_DIR, 'glioma'))[0])
    
    # Resmi modelin anlayacagi formata ceviriyoruz
    test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(sample_img_path).convert('RGB')
    input_tensor = test_transform(img).unsqueeze(0).to(device)
    
  
    # Haritayi uretmek icin son katmani seciyoruz
    target_layer = binary_model.layer4[-1]
    grad_cam = GradCAM(binary_model, target_layer)
    
    # Haritayi olusturuyoruz
    cam, pred_class = grad_cam.generate_heatmap(input_tensor)
    
    # Cikan isil harita resmini kaydediyoruz
    output_cam_path = os.path.join(config.RESULTS_DIR, 'sample_gradcam.png')
    save_gradcam_image(sample_img_path, cam, output_cam_path)
    print(f"Örnek Grad-CAM ısı haritası {output_cam_path} yoluna kaydedildi.")
    
    
    print("\n--- AŞAMA 3: Hibrit Karar Mekanizması Başlıyor ---")
    
    # Ikili siniflandirma icin hibrit modeli egitiyoruz
    print("İkili Sınıflandırma İçin Hibrit Model Eğitimi:")
    hybrid_binary_clf = train_and_evaluate_hybrid(mode='binary')
    
    # Coklu siniflandirma icin hibrit modeli egitiyoruz
    print("\nÇok Sınıflı Sınıflandırma İçin Hibrit Model Eğitimi:")
    hybrid_multi_clf = train_and_evaluate_hybrid(mode='multiclass')
    
    print("\nTüm Eğitim, Hibrit Model Entegrasyonu ve Test İşlemleri Tamamlandı!")

if __name__ == "__main__":
    # Kodu baslatiyoruz
    main()