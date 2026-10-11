#23071001-RecepTayyipYANIK/230710055-SametPulat/220710093-FurkanKubat
import os
# PyTorch modellerinin inecegi yeri kendi klasorumuz yapiyoruz
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['TORCH_HOME'] = os.path.join(BASE_DIR, 'torch_cache')
import torch
import torch.nn as nn
from torchvision import models

# Kanallar arasi dikkat mekanizmasi
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # ONNX hatasi yuzunden kaldirildi
        # self.max_pool = nn.AdaptiveMaxPool2d(1) 
            
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
      
        # En yuksek degerleri alarak isleme devam ediyoruz
        max_x = torch.max(x, dim=2, keepdim=True)[0]
        max_x = torch.max(max_x, dim=3, keepdim=True)[0]
        max_out = self.fc(max_x)
        
        out = avg_out + max_out
        return self.sigmoid(out)

# Mekansal dikkat mekanizmasi
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Ortalama ve en yuksek degerleri birlestiriyoruz
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# Iki dikkat modulunu birlestiren ana sinif
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

# ResNet18 mimarisinin her asamasina bu modulleri ekliyoruz
def add_cbam_to_resnet(model):
    model.layer1.add_module("cbam", CBAM(64))
    model.layer2.add_module("cbam", CBAM(128))
    model.layer3.add_module("cbam", CBAM(256))
    model.layer4.add_module("cbam", CBAM(512))
    return model

def get_model(num_classes=2):
    # Dogrulugu artirmak icin onceden egitilmis agirliklari kullaniyoruz
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Tum katmanlari egitime aciyoruz
    for param in model.parameters():
        param.requires_grad = True
        
    # Modele dikkat modulu ekliyoruz
    model = add_cbam_to_resnet(model)
            
    num_ftrs = model.fc.in_features

    # Son katmani kendi sinif sayimiza gore ayarlayip ezberlemeyi onleyen yapi kuruyoruz
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

def get_feature_extractor():
    # Modeli sadece ozellik cikarici olarak yukleyip son katmani iptal ediyoruz
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    model = add_cbam_to_resnet(model)

    # Siniflandirma yapmasini engelleyip sadece veriyi almasini sagliyoruz
    model.fc = nn.Identity()
    
    return model