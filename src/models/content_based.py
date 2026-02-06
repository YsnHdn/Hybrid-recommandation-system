"""
Modèle de Content-Based Filtering basé sur TF-IDF et similarité cosine
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

from ..config import CONTENT_CONFIG


class ContentBasedModel:
    """
    Modèle de Content-Based Filtering utilisant TF-IDF sur les genres
    """
    
    def __init__(self, **kwargs):
        """
        Initialise le modèle Content-Based
        
        Args:
            **kwargs: Paramètres pour TF-IDF
        """
        # Fusionner les paramètres par défaut avec ceux fournis
        params = {**CONTENT_CONFIG, **kwargs}
        
        self.tfidf = TfidfVectorizer(
            token_pattern=params['token_pattern'],
            lowercase=True
        )
        
        self.min_rating = params['min_rating']
        self.cosine_sim = None
        self.is_trained = False
        
    def fit(self, movies_df: pd.DataFrame):
        """
        Entraîne le modèle en calculant la matrice de similarité
        
        Args:
            movies_df: DataFrame avec colonnes item_id, genres
        """
        print("Entraînement du modèle Content-Based...")
        
        # Vérifier que la colonne genres existe
        if 'genres' not in movies_df.columns:
            raise ValueError("Le DataFrame doit contenir une colonne 'genres'")
        
        # Créer la matrice TF-IDF
        tfidf_matrix = self.tfidf.fit_transform(movies_df['genres'])
        
        # Calculer la matrice de similarité cosine
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        self.is_trained = True
        print(f"Entraînement terminé - Matrice de similarité : {self.cosine_sim.shape}")
    
    
    def get_similar_items(
        self, 
        item_id: int, 
        movies_df: pd.DataFrame, 
        n: int = 10
    ) -> List[int]:
        """
        Trouve les films les plus similaires à un film donné
        
        Args:
            item_id: ID du film
            movies_df: DataFrame des films
            n: Nombre de films similaires à retourner
            
        Returns:
            Liste des item_id similaires (ordonnée par similarité décroissante)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de trouver des films similaires")
        
        # Trouver l'index du film
        try:
            idx = movies_df[movies_df['item_id'] == item_id].index[0]
        except IndexError:
            raise ValueError(f"Film avec item_id={item_id} introuvable")
        
        # Récupérer les scores de similarité
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Trier par similarité décroissante
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Prendre les N films les plus similaires (en excluant le film lui-même)
        sim_scores = sim_scores[1:n+1]
        
        # Récupérer les item_id
        similar_indices = [i[0] for i in sim_scores]
        similar_items = movies_df.iloc[similar_indices]['item_id'].tolist()
        
        return similar_items
    
    
    def recommend(
        self, 
        user_id: int, 
        ratings_df: pd.DataFrame, 
        movies_df: pd.DataFrame, 
        n: int = 10
    ) -> List[int]:
        """
        Génère les top N recommandations pour un utilisateur basé sur ses films aimés
        
        Args:
            user_id: ID de l'utilisateur
            ratings_df: DataFrame des ratings
            movies_df: DataFrame des films
            n: Nombre de recommandations
            
        Returns:
            Liste des item_id recommandés (ordonnée par score décroissant)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des recommandations")
        
        # Films aimés par l'utilisateur
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        liked_films = user_ratings[user_ratings['rating'] >= self.min_rating]
        
        # Films déjà vus
        seen_items = user_ratings['item_id'].tolist()
        
        # Dictionnaire pour accumuler les scores
        recommendation_scores = {}
        
        # Pour chaque film aimé
        for _, row in liked_films.iterrows():
            item_id = row['item_id']
            
            # Trouver l'index du film
            try:
                idx = movies_df[movies_df['item_id'] == item_id].index[0]
            except IndexError:
                continue
            
            # Récupérer les similarités
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            
            # Accumuler les scores
            for film_idx, sim_score in sim_scores:
                film_id = movies_df.iloc[film_idx]['item_id']
                
                if film_id not in seen_items:
                    if film_id not in recommendation_scores:
                        recommendation_scores[film_id] = 0
                    recommendation_scores[film_id] += sim_score
        
        # Trier par score décroissant et prendre le top N
        sorted_recs = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = [item[0] for item in sorted_recs[:n]]
        
        return top_items
    
    def save(self, filepath: str):
        """
        Sauvegarde le modèle entraîné
        
        Args:
            filepath: Chemin où sauvegarder le modèle
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant d'être sauvegardé")
        
        import pickle
        
        model_data = {
            'tfidf': self.tfidf,
            'cosine_sim': self.cosine_sim,
            'min_rating': self.min_rating
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modèle sauvegardé : {filepath}")
        
    @classmethod
    def load(cls, filepath: str):
        """
        Charge un modèle sauvegardé
        
        Args:
            filepath: Chemin du modèle sauvegardé
            
        Returns:
            Instance de ContentBasedModel avec le modèle chargé
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Créer une instance
        instance = cls()
        instance.tfidf = model_data['tfidf']
        instance.cosine_sim = model_data['cosine_sim']
        instance.min_rating = model_data['min_rating']
        instance.is_trained = True
        
        print(f"Modèle chargé : {filepath}")
        
        return instance