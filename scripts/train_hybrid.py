"""
Script pour créer et sauvegarder la configuration du modèle Hybrid
"""
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour importer src
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_data
from src.models.collaborative import CollaborativeModel
from src.models.content_based import ContentBasedModel
from src.models.hybrid import HybridModel
from src.config import SVD_MODEL_PATH, COSINE_SIM_PATH, HYBRID_CONFIG_PATH


def main():
    """
    Charge les modèles Collaborative et Content-Based, crée le Hybrid et sauvegarde la config
    """
    print("="*70)
    print("CRÉATION DU MODÈLE HYBRID")
    print("="*70)
    
    # Charger les données
    print("\n1. Chargement des données...")
    ratings, movies = load_data()
    
    # Charger le modèle Collaborative
    print("\n2. Chargement du modèle Collaborative...")
    collab_model = CollaborativeModel.load(SVD_MODEL_PATH)
    
    # Charger le modèle Content-Based
    print("\n3. Chargement du modèle Content-Based...")
    content_model = ContentBasedModel.load(COSINE_SIM_PATH)
    
    # Créer le modèle Hybrid
    print("\n4. Création du modèle Hybrid...")
    hybrid_model = HybridModel(
        collaborative_model=collab_model,
        content_model=content_model
    )
    
    print(f"   Alpha (Collaborative) : {hybrid_model.alpha}")
    print(f"   Beta (Content-Based) : {hybrid_model.beta}")
    
    # Sauvegarder la configuration
    print("\n5. Sauvegarde de la configuration...")
    hybrid_model.save(HYBRID_CONFIG_PATH)
    
    print("\n" + "="*70)
    print("MODÈLE HYBRID CRÉÉ AVEC SUCCÈS")
    print("="*70)


if __name__ == "__main__":
    main()