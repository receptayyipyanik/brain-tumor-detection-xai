#23071001-RecepTayyipYANIK/230710055-SametPulat/220710093-FurkanKubat
import os
# Kutuphane cakisip cokmesini engelliyoruz
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import os
import torch
from torchvision import transforms
from PIL import Image
import config
from model import get_model
from gradcam import GradCAM, save_gradcam_image
import warnings
# Uyari mesajlarini gizliyoruz
warnings.filterwarnings('ignore')

def predict_and_explain(image_path):
    # Ekran karti varsa seciyoruz yoksa islemcide kaliyoruz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Egitilmis modellerin dosya yollarini aliyoruz
    binary_model_path = os.path.join(config.MODEL_DIR, 'binary_model.pth')
    multi_model_path = os.path.join(config.MODEL_DIR, 'multiclass_model.pth')
    
    # Modeller yoksa dogrudan hata mesaji donduruyoruz
    if not os.path.exists(binary_model_path) or not os.path.exists(multi_model_path):
        return {"error": "Modeller bulunamadi. Lutfen once main.py'yi calistirarak egitimi tamamlayin."}

    # Tumor var mi yok mu diyen ilk modeli yukluyoruz
    binary_model = get_model(num_classes=2).to(device)
    binary_model.load_state_dict(torch.load(binary_model_path, map_location=device))
    binary_model.eval()
    
    # Tumorun turunu tahmin eden ikinci modeli yukluyoruz
    multi_model = get_model(num_classes=3).to(device)
    multi_model.load_state_dict(torch.load(multi_model_path, map_location=device))
    multi_model.eval()

    # Resmi modelin anlayacagi standart boyutlara ve tensore ceviriyoruz
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Resmi acmayi deniyoruz sorun olursa hata donduruyoruz
    try:
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        return {"error": str(e)}

    # Sadece tumor var mi yok mu kismina bakiyoruz
    with torch.no_grad():
        binary_out = binary_model(input_tensor)
        binary_probs = torch.softmax(binary_out, dim=1)[0]
        binary_pred = torch.argmax(binary_out, dim=1).item()
        binary_conf = float(binary_probs[binary_pred].item())
        
    has_tumor = bool(binary_pred == 1)
    
    # Cikti sablonumuzu hazirliyoruz
    result = {
        "has_tumor": has_tumor,
        "confidence": round(binary_conf * 100, 2),
        "tumor_type": None,
        "tumor_confidence": None,
        "heatmap_path": None
    }

    # Tumor bulunduysa turunu belirleyip isil harita cizdiriyoruz
    if has_tumor:
        with torch.no_grad():
            multi_out = multi_model(input_tensor)
            multi_probs = torch.softmax(multi_out, dim=1)[0]
            multi_pred = torch.argmax(multi_out, dim=1).item()
            multi_conf = float(multi_probs[multi_pred].item())
            
            result["tumor_type"] = config.MULTICLASS_LABELS[multi_pred]
            result["tumor_confidence"] = round(multi_conf * 100, 2)
            
        # Odaklanilan yeri tam gormek isin isil haritayi coklu modelden aliyoruz
        target_layer = multi_model.layer4[-1] 
        grad_cam = GradCAM(multi_model, target_layer)
        
        # Tespit edilen sinif isin harita uretiyoruz
        cam, _ = grad_cam.generate_heatmap(input_tensor, class_idx=multi_pred)
        
        # Arayuzden okunabilmesi isin haritayi disari kaydediyoruz
        output_heatmap_path = os.path.abspath(os.path.join(config.RESULTS_DIR, 'current_heatmap.png'))
        save_gradcam_image(image_path, cam, output_heatmap_path)
        
        result["heatmap_path"] = output_heatmap_path

    return result

if __name__ == "__main__":
    # Konsoldan resim yolunu disaridan almak isin ayar yapiyoruz
    parser = argparse.ArgumentParser(description="Beyin Tumoru Tespiti ve Grad-CAM")
    parser.add_argument("--image", type=str, required=True, help="Test edilecek resmin dosya yolu")
    
    args = parser.parse_args()
    
    # C sharp gibi dillerle haberlesebilmek isin sonucu JSON seklinde yazdiriyoruz
    output = predict_and_explain(args.image)
    print(json.dumps(output))