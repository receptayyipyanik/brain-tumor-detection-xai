#23071001-RecepTayyipYANIK/230710055-SametPulat/220710093-FurkanKubat
import os

# Temel dizin (kod klasörünün mutlak yolu)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Veri seti yolları
DATA_DIR = os.path.join(BASE_DIR, 'archive (1)')
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
TEST_DIR = os.path.join(DATA_DIR, 'Testing')
MATLAB_DIR = os.path.join(BASE_DIR, 'BRAIN_DATA-20260506T064827Z-3-001', 'BRAIN_DATA')

# Model kaydetme yolu
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Hiperparametreler
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_BINARY = 20
EPOCHS_MULTICLASS = 30
LEARNING_RATE = 1e-4

# Sınıflar
MULTICLASS_LABELS = ['glioma', 'meningioma', 'pituitary']

