#23071001-RecepTayyipYANIK/230710055-SametPulat/220710093-FurkanKubat
import torch
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import config
from dataset import get_dataloaders
from model import get_feature_extractor

def extract_features(model, dataloader, device):
    """
    Model uzerinden veriyi gecirip ozellikleri ve etiketleri dizi olarak doneriz
    """
    # Degerlendirme modunu aciyoruz
    model.eval()
    all_features = []
    all_labels = []

    print("Özellikler (Features) Çıkarılıyor...")
    # Gradyan hesabini kapatiyoruz
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            
            # Veriyi modelden geciriyoruz
            features = model(inputs)
            
            # Veriyi islemciye alip diziye ceviriyoruz
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
    # Tum parcalari tek bir dizide birlestiriyoruz
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    return X, y

def train_and_evaluate_hybrid(mode='binary'):
    """
    Secilen moda gore hibrit modeli kurup calistiririz
    """
    # Ekran karti varsa onu yoksa islemciyi seciyoruz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{mode.upper()} - HİBRİT MODEL] Cihaz: {device}")
    
    # Veri yukleyicileri cagiriyoruz
    train_loader, test_loader = get_dataloaders(mode=mode)
    
    # Ozellik cikarici modeli hazirliyoruz
    feature_extractor = get_feature_extractor().to(device)
    
    # Egitim verisinin ozelliklerini cikariyoruz
    print("Eğitim seti için:")
    X_train, y_train = extract_features(feature_extractor, train_loader, device)
    
    # Test verisinin ozelliklerini cikariyoruz
    print("Test seti için:")
    X_test, y_test = extract_features(feature_extractor, test_loader, device)
    
    print(f"Çıkarılan Eğitim Özellikleri Boyutu: {X_train.shape}")
    print(f"Çıkarılan Test Özellikleri Boyutu: {X_test.shape}")
    
    # Destek vektor makinesini egitiyoruz
    print("\nDestek Vektör Makinesi (SVM) Eğitiliyor...")
    # Iyi sonuc veren standart ayarlari kullaniyoruz
    clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    
    # Egitilen modeli test ediyoruz
    print("Model Test Ediliyor...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {mode.upper()} HİBRİT MODEL SONUÇLARI ---")
    print(f"Doğruluk (Accuracy): %{acc * 100:.2f}")
    
    target_names = ['No Tumor', 'Tumor'] if mode == 'binary' else config.MULTICLASS_LABELS
    print("\nSınıflandırma Raporu (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\nKarmaşıklık Matrisi (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
    
    # Son uretilen modeli disari kaydediyoruz
    model_path = os.path.join(config.MODEL_DIR, f'hybrid_svm_{mode}.pkl')
    joblib.dump(clf, model_path)
    print(f"Hibrit model {model_path} olarak kaydedildi (Python / scikit-learn için).")
    
    return clf