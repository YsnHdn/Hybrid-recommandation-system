"""
Module de prétraitement des données
"""
import numpy as np
import pandas as pd
from typing import Tuple


def create_user_train_test_split(
    ratings_df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crée un train/test split pour chaque utilisateur
    
    Pour chaque utilisateur, on sépare ses ratings en train (80%) et test (20%)
    
    Args:
        ratings_df: DataFrame avec les ratings
        test_size: Proportion du test set (default: 0.2)
        random_state: Seed pour reproductibilité
        
    Returns:
        Tuple (train_df, test_df)
    """
    np.random.seed(random_state)
    
    train_list = []
    test_list = []
    
    for user_id in ratings_df['user_id'].unique():
        user_data = ratings_df[ratings_df['user_id'] == user_id]
        
        # Mélanger les ratings de l'utilisateur
        user_data = user_data.sample(frac=1, random_state=random_state)
        
        # Calculer le split
        n_test = max(1, int(len(user_data) * test_size))
        
        test_data = user_data.iloc[:n_test]
        train_data = user_data.iloc[n_test:]
        
        train_list.append(train_data)
        test_list.append(test_data)
    
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    return train_df, test_df

def get_relevant_items(
    user_id: int, 
    test_df: pd.DataFrame, 
    min_rating: int = 4
) -> list:
    """
    Retourne les films que l'utilisateur a aimés dans le test set
    
    Args:
        user_id: ID de l'utilisateur
        test_df: DataFrame de test
        min_rating: Note minimum pour être considéré comme pertinent (default: 4)
        
    Returns:
        Liste des item_id pertinents
    """
    user_test = test_df[test_df['user_id'] == user_id]
    relevant = user_test[user_test['rating'] >= min_rating]
    
    return relevant['item_id'].tolist()

def normalize_scores(
    scores_df: pd.DataFrame, 
    score_column: str
) -> pd.DataFrame:
    """
    Normalise les scores entre 0 et 1 avec Min-Max Scaling
    
    Args:
        scores_df: DataFrame avec les scores
        score_column: Nom de la colonne à normaliser
        
    Returns:
        DataFrame avec une colonne 'normalized_score' ajoutée
    """
    min_score = scores_df[score_column].min()
    max_score = scores_df[score_column].max()
    
    # Éviter la division par zéro
    if max_score == min_score:
        scores_df['normalized_score'] = 0.5
    else:
        scores_df['normalized_score'] = (
            (scores_df[score_column] - min_score) / (max_score - min_score)
        )
    
    return scores_df


