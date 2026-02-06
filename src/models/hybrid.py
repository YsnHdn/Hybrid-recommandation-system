"""
Modèle Hybrid combinant Collaborative et Content-Based Filtering
"""
import pandas as pd
from typing import List

from .collaborative import CollaborativeModel
from .content_based import ContentBasedModel
from ..utils.preprocessing import normalize_scores
from ..config import HYBRID_CONFIG


class HybridModel:
    """
    Modèle Hybrid combinant Collaborative et Content-Based avec pondération
    """
    
    def __init__(
        self, 
        collaborative_model: CollaborativeModel,
        content_model: ContentBasedModel,
        alpha: float = None
    ):
        """
        Initialise le modèle Hybrid
        
        Args:
            collaborative_model: Instance du modèle Collaborative
            content_model: Instance du modèle Content-Based
            alpha: Poids du Collaborative (default depuis config)
        """
        self.collaborative_model = collaborative_model
        self.content_model = content_model
        
        # Utiliser alpha depuis config si non fourni
        self.alpha = alpha if alpha is not None else HYBRID_CONFIG['alpha']
        self.beta = 1.0 - self.alpha
        
        # Vérifier que les modèles sont entraînés
        if not self.collaborative_model.is_trained:
            raise ValueError("Le modèle Collaborative doit être entraîné")
        if not self.content_model.is_trained:
            raise ValueError("Le modèle Content-Based doit être entraîné")
    
    
    def recommend(
        self, 
        user_id: int, 
        ratings_df: pd.DataFrame, 
        movies_df: pd.DataFrame, 
        n: int = 10
    ) -> List[int]:
        """
        Génère les top N recommandations hybrides pour un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            ratings_df: DataFrame des ratings
            movies_df: DataFrame des films
            n: Nombre de recommandations
            
        Returns:
            Liste des item_id recommandés (ordonnée par score hybride décroissant)
        """
        # Obtenir les recommandations des deux modèles (top 50 pour avoir du choix)
        collab_items = self.collaborative_model.recommend(
            user_id, ratings_df, movies_df, n=50
        )
        content_items = self.content_model.recommend(
            user_id, ratings_df, movies_df, n=50
        )
        
        # Créer des scores pour chaque approche (basé sur le rang)
        hybrid_scores = {}
        
        # Scores Collaborative (plus le rang est bon, plus le score est élevé)
        for i, item_id in enumerate(collab_items):
            hybrid_scores[item_id] = self.alpha * (50 - i)
        
        # Scores Content-Based (ajouter ou créer)
        for i, item_id in enumerate(content_items):
            if item_id in hybrid_scores:
                hybrid_scores[item_id] += self.beta * (50 - i)
            else:
                hybrid_scores[item_id] = self.beta * (50 - i)
        
        # Trier par score décroissant et prendre le top N
        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = [item[0] for item in sorted_hybrid[:n]]
        
        return top_items
    
    def save(self, filepath: str):
        """
        Sauvegarde la configuration du modèle hybrid
        
        Args:
            filepath: Chemin où sauvegarder la configuration
        """
        import pickle
        
        config = {
            'alpha': self.alpha,
            'beta': self.beta
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Configuration Hybrid sauvegardée : {filepath}")
        
        
        @classmethod
    def load(
        cls, 
        filepath: str,
        collaborative_model: CollaborativeModel,
        content_model: ContentBasedModel
    ):
        """
        Charge une configuration Hybrid sauvegardée
        
        Args:
            filepath: Chemin de la configuration
            collaborative_model: Instance du modèle Collaborative
            content_model: Instance du modèle Content-Based
            
        Returns:
            Instance de HybridModel avec la configuration chargée
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        instance = cls(
            collaborative_model=collaborative_model,
            content_model=content_model,
            alpha=config['alpha']
        )
        
        print(f"Configuration Hybrid chargée : {filepath}")
        
        return instance