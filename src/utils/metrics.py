"""
Module d'évaluation pour les systèmes de recommandation
"""
import numpy as np
from typing import List


def precision_at_k(
    recommended_items: List[int], 
    relevant_items: List[int], 
    k: int
) -> float:
    """
    Calcule la Precision@K
    
    Precision@K = (Nombre de films pertinents dans le top K) / K
    
    Args:
        recommended_items: Liste des items recommandés (ordonnée)
        relevant_items: Liste des items pertinents
        k: Nombre de recommandations à considérer
        
    Returns:
        Precision@K (entre 0 et 1)
    """
    if k == 0:
        return 0.0
    
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    
    hits = len([item for item in recommended_at_k if item in relevant_set])
    
    return hits / k

def recall_at_k(
    recommended_items: List[int], 
    relevant_items: List[int], 
    k: int
) -> float:
    """
    Calcule le Recall@K
    
    Recall@K = (Nombre de films pertinents dans le top K) / (Total de films pertinents)
    
    Args:
        recommended_items: Liste des items recommandés (ordonnée)
        relevant_items: Liste des items pertinents
        k: Nombre de recommandations à considérer
        
    Returns:
        Recall@K (entre 0 et 1)
    """
    if len(relevant_items) == 0:
        return 0.0
    
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    
    hits = len([item for item in recommended_at_k if item in relevant_set])
    
    return hits / len(relevant_items)

def ndcg_at_k(
    recommended_items: List[int], 
    relevant_items: List[int], 
    k: int
) -> float:
    """
    Calcule le NDCG@K (Normalized Discounted Cumulative Gain)
    
    Plus un film pertinent est haut dans la liste, meilleur est le score
    
    Args:
        recommended_items: Liste des items recommandés (ordonnée)
        relevant_items: Liste des items pertinents
        k: Nombre de recommandations à considérer
        
    Returns:
        NDCG@K (entre 0 et 1)
    """
    if len(relevant_items) == 0 or k == 0:
        return 0.0
    
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    
    # DCG : Discounted Cumulative Gain
    dcg = 0.0
    for i, item in enumerate(recommended_at_k):
        if item in relevant_set:
            # Formule : relevance / log2(position + 2)
            dcg += 1.0 / np.log2(i + 2)
    
    # IDCG : Ideal DCG (si tous les items pertinents étaient en haut)
    idcg = 0.0
    for i in range(min(len(relevant_items), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    # NDCG
    if idcg == 0:
        return 0.0
    
    return dcg / idcg