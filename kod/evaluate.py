#23071001-RecepTayyipYANIK/230710055-SametPulat/220710093-FurkanKubat
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config

def evaluate_model(model, test_loader, device, mode='binary'):
    # Modeli degerlendirme moduna aliyoruz
    model.eval()
    all_preds = []
    all_labels = []
    
    # Bellek tasarrufu icin gradyan hesaplamayi kapatiyoruz
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Verileri ilgili cihaza gonderiyoruz
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            # En yuksek olasilikli tahmini seciyoruz
            _, preds = torch.max(outputs, 1)
            
            # Tahminleri ve gercek etiketleri listeye ekliyoruz
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    if mode == 'binary':
        # Ikili siniflandirma isimleri
        target_names = ['No Tumor', 'Tumor']
    else:
        # Coklu siniflandirma isimleri
        target_names = config.MULTICLASS_LABELS
        
    print(f"\n--- {mode.upper()} SINIFLANDIRMA RAPORU ---")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Karmasiklik matrisini hesapliyoruz
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    # Matrisi isil harita ile gorsellestiriyoruz
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Sınıf')
    plt.title(f'Karmaşıklık Matrisi - {mode.capitalize()}')
    
    # Cizilen matrisi dizine kaydediyoruz
    plt.savefig(os.path.join(config.RESULTS_DIR, f'confusion_matrix_{mode}.png'))
    plt.close()

def plot_history(history, mode='binary'):
    # Toplam adim sayisini buluyoruz
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Grafigi iki parcaya bolup ilkine kayiplari ciziyoruz
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Grafigin ikinci kismina dogruluk oranlarini ciziyoruz
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Sonuclari kaydedip pencereyi kapatiyoruz
    plt.savefig(os.path.join(config.RESULTS_DIR, f'training_history_{mode}.png'))
    plt.close()