#23071001-RecepTayyipYANIK/230710055-SametPulat/220710093-FurkanKubat
import os
# Kütüphane çakışmasını önlemek için KMP ayarını yapıyoruz
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Subset
import config
from PIL import Image
import h5py
import numpy as np

# Standart resimleri okuyan veri seti sınıfımız
class TumorDataset(Dataset):
    def __init__(self, root_dir, mode='binary', transform=None):
        """
        mode: 'binary' veya 'multiclass'
        binary -> 0: No Tumor, 1: Tumor
        multiclass -> Sadece tümörlü resimler (glioma, meningioma, pituitary) (0, 1, 2)
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        # Resim yolları ve etiketleri tutacak liste
        self.samples = []
        
        # Ana klasördeki alt sınıfları geziyoruz
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            # Klasör değilse atlıyoruz
            if not os.path.isdir(class_dir):
                continue
                
            # Sınıf içindeki tüm resimleri geziyoruz
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                
                if mode == 'binary':
                    # Klasör adı notumor ise etiket 0 değilse 1
                    label = 0 if class_name == 'notumor' else 1
                    self.samples.append((img_path, label))
                elif mode == 'multiclass':
                    # Sağlıklı beyinleri atlayıp sadece tümör tiplerine odaklanıyoruz
                    if class_name == 'notumor':
                        continue
                    if class_name in config.MULTICLASS_LABELS:
                        # Etiketi config dosyasındaki listeye göre numaralandırıyoruz
                        label = config.MULTICLASS_LABELS.index(class_name)
                        self.samples.append((img_path, label))

    def __len__(self):
        # Toplam veri sayısını dönüyoruz
        return len(self.samples)

    def __getitem__(self, idx):
        # Modele beslenecek veriyi buradan çekiyoruz
        img_path, label = self.samples[idx]
        # Kanal sayısı standart olsun diye RGB formatına çeviriyoruz
        image = Image.open(img_path).convert('RGB')
        
        # Transform işlemi varsa uyguluyoruz
        if self.transform:
            image = self.transform(image)
            
        return image, label

# mat uzantılı MATLAB veri setini okumak için özel sınıf
class MatlabTumorDataset(Dataset):
    def __init__(self, root_dir, mode='binary'):
        self.root_dir = root_dir
        self.mode = mode
        self.samples = []
        
        # Dizin varsa içindeki mat dosyalarını listeye ekliyoruz
        if os.path.exists(root_dir):
            for file_name in os.listdir(root_dir):
                if file_name.endswith('.mat'):
                    self.samples.append(os.path.join(root_dir, file_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mat_path = self.samples[idx]
        
        # h5py ile matlab dosyasının içindeki matrisleri okuyoruz
        with h5py.File(mat_path, 'r') as f:
            cjdata = f['cjdata']
            # MATLAB dosyasından etiket değerini alıyoruz
            mat_label = int(cjdata['label'][0, 0])
            # Görüntüyü numpy matrisi olarak alıyoruz
            img_array = np.array(cjdata['image'])
            
        # Matris değerlerini 0 ile 255 arasına normalize ediyoruz
        img_min = img_array.min()
        img_max = img_array.max()
        if img_max > img_min:
            img_normalized = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_normalized = np.zeros_like(img_array, dtype=np.uint8)
            
        # Normalize matrisi formatlayıp RGB yapıyoruz
        image = Image.fromarray(img_normalized).convert('RGB')
        
        # MATLAB etiketlerini projemizdeki sıraya uyduruyoruz
        if mat_label == 1:
            our_label = config.MULTICLASS_LABELS.index('meningioma')
        elif mat_label == 2:
            our_label = config.MULTICLASS_LABELS.index('glioma')
        elif mat_label == 3:
            our_label = config.MULTICLASS_LABELS.index('pituitary')
        else:
            our_label = 0
            
        # Binary moddaysak tümörlü olduğu için doğrudan 1 yapıyoruz
        final_label = 1 if self.mode == 'binary' else our_label
        return image, final_label

# Veriyi böldükten sonra train ve test için ayrı transform uygulamak için sarmalayıcı sınıf
class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label
        
    def __len__(self):
        return len(self.subset)

def get_dataloaders(mode='binary'):
    # Overfitting'i azaltmak ve modelin tümöre daha iyi odaklanmasını sağlamak için 
    # RandomAffine (Translation) ve GaussianBlur (Gürültü/Bulanıklaştırma) eklendi.
    # Modele veri artırma uygulayıp ezber yapmasını önlüyoruz
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), # Resmi standart boyuta getir
        transforms.RandomHorizontalFlip(), # Rastgele yatay çevir
        transforms.RandomRotation(15), # En fazla 15 derece döndür
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # B maddesi: Kaydırma (Translation)
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Parlaklık ve zıtlığı hafif değiştir
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # B maddesi: Giriş Gürültüsü (Noise/Bulanıklık)
        transforms.ToTensor(), # PyTorch tensör formatına çevir
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standart normalize et
    ])
    
    # Test verisinde artırma yapılmaz sadece boyutlandırıp normalize ediyoruz
    test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"[{mode.upper()}] Veri setleri hazirlaniyor...")

    # Klasik resimli veri setlerini yüklüyoruz
    base_train = TumorDataset(config.TRAIN_DIR, mode=mode, transform=train_transform)
    base_test = TumorDataset(config.TEST_DIR, mode=mode, transform=test_transform)
    
    # MATLAB veri setini yüklüyoruz
    matlab_full = MatlabTumorDataset(config.MATLAB_DIR, mode=mode)
    
    if len(matlab_full) > 0:
        print(f"[{mode.upper()}] {len(matlab_full)} adet MATLAB (.mat) verisi bulundu. Entegre ediliyor...")
        # Veri sızıntısını önlemek için sabit seed kullanarak bölüyoruz
        generator = torch.Generator().manual_seed(42)
        # Verinin yüzde 80 kısmı eğitime kalanı teste
        train_size = int(0.8 * len(matlab_full))
        test_size = len(matlab_full) - train_size
        
        # Rastgele bölüyoruz
        matlab_train_subset, matlab_test_subset = random_split(matlab_full, [train_size, test_size], generator=generator)
        
        # Alt kümeleri ilgili transformlarla sarmalıyoruz
        matlab_train = DatasetWrapper(matlab_train_subset, transform=train_transform)
        matlab_test = DatasetWrapper(matlab_test_subset, transform=test_transform)
        
        # Klasik resim verisi ile MATLAB verisini tek bir veri setinde birleştiriyoruz
        final_train_dataset = ConcatDataset([base_train, matlab_train])
        final_test_dataset = ConcatDataset([base_test, matlab_test])
    else:
        # MATLAB verisi yoksa sadece klasikleri kullanıyoruz
        final_train_dataset = base_train
        final_test_dataset = base_test
        
    print(f"[{mode.upper()}] Toplam Egitim Seti: {len(final_train_dataset)} resim")
    print(f"[{mode.upper()}] Toplam Test Seti: {len(final_test_dataset)} resim")

    # Modeli eğitirken verileri gruplar halinde verecek yapıları oluşturuyoruz
    # Eğitim verisi karıştırılır test verisi karıştırılmaz
    train_loader = DataLoader(final_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(final_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader