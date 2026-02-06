"""
Script pour entraîner le modèle Collaborative Filtering (SVD)
"""
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour importer src
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_data
from src.models.collaborative import CollaborativeModel
from src.config import SVD_MODEL_PATH


def main():
    """
    Entraîne et sauvegarde le modèle Collaborative
    """
    print("="*70)
    print("ENTRAÎNEMENT DU MODÈLE COLLABORATIVE FILTERING (SVD)")
    print("="*70)
    
    # Charger les données
    print("\n1. Chargement des données...")
    ratings, movies = load_data()
    
    # Créer et entraîner le modèle
    print("\n2. Création du modèle...")
    model = CollaborativeModel()
    
    print("\n3. Entraînement...")
    model.fit(ratings)
    
    # Sauvegarder le modèle
    print("\n4. Sauvegarde du modèle...")
    model.save(SVD_MODEL_PATH)
    
    print("\n" + "="*70)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("="*70)


if __name__ == "__main__":
    main()