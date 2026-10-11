#23071001-RecepTayyipYANIK/230710055-SametPulat/220710093-FurkanKubat
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Ileri ve geri besleme kancalarini takiyoruz
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        # Cikan aktivasyon sonucunu kaydediyoruz
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # Geri yayilimdaki gradyanlari aliyoruz
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_tensor, class_idx=None):
        # Resmi modele veriyoruz
        model_output = self.model(input_tensor)
        
        if class_idx is None:
            # Sinif yoksa en yuksek skorlu olani seciyoruz
            class_idx = torch.argmax(model_output, dim=1).item()
            
        # Eski gradyanlari temizliyoruz
        self.model.zero_grad()
        
        # Sadece hedef sinif skoruna gore turev aliyoruz
        score = model_output[0, class_idx]
        score.backward()
        
        # Gradyan ortalamalari ile agirliklari buluyoruz
        weights = torch.mean(self.gradients, dim=[2, 3])[0]
        
        # Aktivasyonlari agirliklarla carpip topluyoruz
        activations = self.activations[0]
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(activations.device)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Sadece pozitif kisimlari aliyoruz
        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()
        
        # Degerleri sifir ile bir arasina cekiyoruz
        if np.max(cam) != 0:
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            
        return cam, class_idx

def save_gradcam_image(image_path, cam, output_path, img_size=224):
    # Turkce karakter sikintisi cikmamasi icin resmi boyle okuyoruz
    with open(image_path, "rb") as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
    original_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Orijinal resmi yeniden boyutlandiriyoruz
    original_image = cv2.resize(original_image, (img_size, img_size))
    
    # Haritayi resim boyutuna uyarlayalim
    cam = cv2.resize(cam, (img_size, img_size))
    
    # Haritayi renkli yapiyoruz
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Resim ile haritayi ust uste koyuyoruz
    superimposed_img = np.uint8(np.clip(heatmap * 0.4 + original_image, 0, 255))
    
    # Sorunsuz sekilde disari kaydediyoruz
    is_success, im_buf_arr = cv2.imencode(".png", superimposed_img)
    if is_success:
        im_buf_arr.tofile(output_path)