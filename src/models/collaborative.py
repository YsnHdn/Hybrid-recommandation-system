"""
Modèle de Collaborative Filtering basé sur SVD
"""
import pandas as pd
from surprise import SVD, Dataset, Reader, Trainset
from typing import List

from ..config import SVD_CONFIG


class CollaborativeModel:
    """
    Modèle de Collaborative Filtering utilisant SVD (Surprise)
    """
    
    def __init__(self, **kwargs):
        """
        Initialise le modèle SVD
        
        Args:
            **kwargs: Paramètres pour SVD (n_factors, n_epochs, lr_all, reg_all)
        """
        # Fusionner les paramètres par défaut avec ceux fournis
        params = {**SVD_CONFIG, **kwargs}
        
        self.algo = SVD(
            n_factors=params['n_factors'],
            n_epochs=params['n_epochs'],
            lr_all=params['lr_all'],
            reg_all=params['reg_all'],
            random_state=params['random_state']
        )
        
        self.is_trained = False
        
    
    def fit(self, ratings_df: pd.DataFrame):
        """
        Entraîne le modèle sur les données de ratings
        
        Args:
            ratings_df: DataFrame avec colonnes user_id, item_id, rating
        """
        
        reader = Reader(rating_scale=(1,5))
        data = Dataset.load_from_df(ratings_df[['user_id' , 'item_id' , 'ratings']] , reader = reader)
        
        trainset = data.build_full_trainset()
        
        # Entraîner
        print("Entraînement du modèle Collaborative (SVD)...")
        self.algo.fit(trainset)
        self.is_trained = True
        print("Entraînement terminé")
        
    
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Prédit la note qu'un utilisateur donnerait à un film
        
        Args:
            user_id: ID de l'utilisateur
            item_id: ID du film
            
        Returns:
            Note prédite (entre 1 et 5)
        """
        
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        prediction = self.algo.predict(user_id, item_id)
        return prediction.est
    
    def recommend(
        self, 
        user_id: int, 
        ratings_df: pd.DataFrame, 
        movies_df: pd.DataFrame, 
        n: int = 10
    ) -> List[int]:
        """
        Génère les top N recommandations pour un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            ratings_df: DataFrame des ratings (pour savoir quels films sont déjà vus)
            movies_df: DataFrame des films (pour connaître tous les films disponibles)
            n: Nombre de recommandations
            
        Returns:
            Liste des item_id recommandés (ordonnée par score décroissant)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des recommandations")
        
        # Films déjà notés par l'utilisateur
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        rated_items = user_ratings['item_id'].tolist()
        
        # Tous les films disponibles
        all_items = movies_df['item_id'].tolist()
        
        # Films non notés
        items_to_predict = [item for item in all_items if item not in rated_items]
        
        # Prédire les notes pour tous les films non notés
        predictions = []
        for item_id in items_to_predict:
            pred = self.algo.predict(user_id, item_id)
            predictions.append({
                'item_id': item_id,
                'score': pred.est
            })
        
        # Trier par score décroissant et prendre le top N
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('score', ascending=False).head(n)
        
        return predictions_df['item_id'].tolist()
    
    def save(self, filepath: str):
        """
        Sauvegarde le modèle entraîné
        
        Args:
            filepath: Chemin où sauvegarder le modèle
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant d'être sauvegardé")
        
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.algo, f)
        
        print(f"Modèle sauvegardé : {filepath}")
    
    
    ## class method permet à la methode d'instancier directement et appliquer la methode
    ## Cas Simple : Problème : On crée une instance vide d'abord, c'est bizarre.
            #     class CollaborativeModel:
            #     def load(self, filepath):  # Méthode normale
            #         algo = pickle.load(...)
            #         self.algo = algo
            #         self.is_trained = True
            #         # Pas de return

            # # Usage
            # model = CollaborativeModel()  # On doit d'abord créer une instance vide
            # model.load("model.pkl")       # Puis la remplir
            # model.recommend(...)          # Maintenant on peut l'utiliser
    ## avec classmethod : 
            #     class CollaborativeModel:
            #     @classmethod
            #     def load(cls, filepath):  # Méthode de classe
            #         algo = pickle.load(...)
            #         instance = cls()  # Crée une nouvelle instance
            #         instance.algo = algo
            #         instance.is_trained = True
            #         return instance  # Retourne l'instance
            
            # # Usage
            # model = CollaborativeModel.load("model.pkl")  # Directement !
            # model.recommend(...)  # Prêt à utiliser
            # model.recommend(...)  # Prêt à utiliser
    @classmethod
    def load(cls , file_path):
        """
        Charge un modèle sauvegardé
        
        Args:
            filepath: Chemin du modèle sauvegardé
            
        Returns:
            Instance de CollaborativeModel avec le modèle chargé
        """
        import pickle
        
        with open(file_path , 'rb') as f:
            algo = pickle.load(f)
        
        instance = cls()
        instance.algo = algo
        instance.is_trained = True
        
        return instance 
            
            