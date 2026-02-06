"""
Script pour entraîner le modèle Content-Based Filtering
"""
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour importer src
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_data
from src.models.content_based import ContentBasedModel
from src.config import COSINE_SIM_PATH


def main():
    """
    Entraîne et sauvegarde le modèle Content-Based
    """
    print("="*70)
    print("ENTRAÎNEMENT DU MODÈLE CONTENT-BASED FILTERING")
    print("="*70)
    
    # Charger les données
    print("\n1. Chargement des données...")
    ratings, movies = load_data()
    
    # Créer et entraîner le modèle
    print("\n2. Création du modèle...")
    model = ContentBasedModel()
    
    print("\n3. Entraînement...")
    model.fit(movies)
    
    # Sauvegarder le modèle
    print("\n4. Sauvegarde du modèle...")
    model.save(COSINE_SIM_PATH)
    
    print("\n" + "="*70)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("="*70)


if __name__ == "__main__":
    main()