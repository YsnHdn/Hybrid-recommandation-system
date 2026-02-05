import pandas as pd
from pathlib import Path
from typing import Tuple
from ..config import RATINGS_FILE, MOVIES_FILE , SVD_MODEL_PATH , COSINE_SIM_PATH
import pickle



def load_ratings() -> pd.DataFrame:
    """
    Charge les ratings nettoyés
    
    Returns:
        DataFrame avec les colonnes : user_id, item_id, rating, timestamp, etc.
    """
    
    ratings = pd.read_csv(RATINGS_FILE)
    print(f"Ratings chargés : {len(ratings):,} lignes")

    
    return ratings

def load_movies() -> pd.DataFrame:
    """
    Charge les informations sur les films
    
    Returns:
        DataFrame avec les colonnes : item_id, title, year, genres, etc.
    """
    if not MOVIES_FILE.exists():
        raise FileNotFoundError(f"Fichier movies introuvable : {MOVIES_FILE}")
    
    movies = pd.read_csv(MOVIES_FILE)
    
    print(f"Films chargés : {len(movies):,} lignes")
    
    return movies

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge à la fois les ratings et les films
    
    Returns:
        Tuple (ratings_df, movies_df)
    """
    ratings = load_ratings()
    movies = load_movies()
    
    return ratings, movies

import pickle


def load_svd_model():
    """
    Charge le modèle SVD sauvegardé
    
    Returns:
        Modèle SVD entraîné
    """
    if not SVD_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle SVD introuvable : {SVD_MODEL_PATH}")
    
    with open(SVD_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print("Modèle SVD chargé")
    return model


def load_cosine_similarity():
    """
    Charge la matrice de similarité cosine
    
    Returns:
        Matrice de similarité (numpy array)
    """
    if not COSINE_SIM_PATH.exists():
        raise FileNotFoundError(f"Matrice de similarité introuvable : {COSINE_SIM_PATH}")
    
    with open(COSINE_SIM_PATH, 'rb') as f:
        cosine_sim = pickle.load(f)
    
    print(f"Matrice de similarité chargée : {cosine_sim.shape}")
    return cosine_sim


def load_all_models():
    """
    Charge tous les modèles nécessaires
    
    Returns:
        Tuple (svd_model, cosine_sim, ratings_df, movies_df)
    """
    print("Chargement de tous les modèles et données...\n")
    
    # Charger les données
    ratings, movies = load_data()
    
    # Charger les modèles
    svd_model = load_svd_model()
    cosine_sim = load_cosine_similarity()
    
    print("\nTous les modèles et données chargés avec succès")
    
    return svd_model, cosine_sim, ratings, movies