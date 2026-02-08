import os
from pathlib import Path

# Chemins des fichiers
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"


# Chemins des données
RATINGS_FILE = PROCESSED_DATA_DIR / "ratings_clean.csv"
MOVIES_FILE = PROCESSED_DATA_DIR / "movies_clean.csv"

# Chemins des modeles 
SVD_MODEL_PATH = MODELS_DIR / "svd_model.pkl"
COSINE_SIM_PATH = MODELS_DIR / "cosine_sim_matrix.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
HYBRID_CONFIG_PATH = MODELS_DIR / "hybrid_config.pkl"

## Mlflow 
MLFLOW_TRACKING_URI = 'file:./mlruns'
MLFLOW_EXPERIMENT_NAME = 'recommendation_system' 

SVD_CONFIG = {
    'n_factors': 100,
    'n_epochs': 20,
    'lr_all': 0.005,
    'reg_all': 0.02,
    'random_state': 42
}

# Paramètres du modèle Content-Based
CONTENT_CONFIG = {
    'min_rating': 4,  # Note minimum pour considérer qu'un film est aimé
    'token_pattern': r'[A-Za-z-]+'  # Pattern pour TF-IDF
}

# Paramètres du modèle Hybrid
HYBRID_CONFIG = {
    'alpha': 0.7,  # Poids du Collaborative
    'beta': 0.3    # Poids du Content-Based
}

# Paramètres d'évaluation
EVALUATION_CONFIG = {
    'test_size': 0.2,
    'k_values': [5, 10, 20],
    'random_state': 42
}

# Paramètres de l'API
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True
}
