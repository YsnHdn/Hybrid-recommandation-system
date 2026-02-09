"""
Script pour évaluer les trois modèles de recommandation
"""
import sys
from pathlib import Path
import pandas as pd

# Ajouter le dossier parent au path pour importer src
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_data
from src.models.collaborative import CollaborativeModel
from src.models.content_based import ContentBasedModel
from src.models.hybrid import HybridModel
from src.utils.preprocessing import create_user_train_test_split, get_relevant_items
from src.utils.metrics import precision_at_k, recall_at_k, ndcg_at_k
from src.config import SVD_CONFIG, SVD_MODEL_PATH, COSINE_SIM_PATH, HYBRID_CONFIG_PATH, EVALUATION_CONFIG , MLFLOW_EXPERIMENT_NAME , MLFLOW_TRACKING_URI


import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def evaluate_model(model, model_name, train_df, test_df, movies_df, k_values, sample_users=100):
    """
    Évalue un modèle sur plusieurs utilisateurs
    """
    print(f"\nÉvaluation de {model_name}...")
    
    users = train_df['user_id'].unique()[:sample_users]
    results = []
    
    for i, user_id in enumerate(users):
        if (i + 1) % 25 == 0:
            print(f"  Progression : {i+1}/{len(users)} utilisateurs")
        
        # Films pertinents
        relevant = get_relevant_items(user_id, test_df, min_rating=4)
        
        if len(relevant) == 0:
            continue
        
        # Générer recommandations
        try:
            recommendations = model.recommend(user_id, train_df, movies_df, n=20)
        except:
            continue
        
        # Calculer métriques
        for k in k_values:
            precision = precision_at_k(recommendations, relevant, k)
            recall = recall_at_k(recommendations, relevant, k)
            ndcg = ndcg_at_k(recommendations, relevant, k)
            
            results.append({
                'model': model_name,
                'k': k,
                'precision': precision,
                'recall': recall,
                'ndcg': ndcg
            })
    
    return pd.DataFrame(results)


def main():
    """
    Évalue les trois modèles et affiche les résultats
    """
    print("="*70)
    print("ÉVALUATION DES MODÈLES DE RECOMMANDATION")
    print("="*70)
    
    # Charger les données
    print("\n1. Chargement des données...")
    ratings, movies = load_data()
    
    # Créer train/test split
    print("\n2. Création du train/test split...")
    train_ratings, test_ratings = create_user_train_test_split(
        ratings, 
        test_size=EVALUATION_CONFIG['test_size'],
        random_state=EVALUATION_CONFIG['random_state']
    )
    
    print(f"   Train : {len(train_ratings):,} ratings")
    print(f"   Test : {len(test_ratings):,} ratings")
    
    # Charger les modèles
    print("\n3. Chargement des modèles...")
    collab_model = CollaborativeModel.load(SVD_MODEL_PATH)
    content_model = ContentBasedModel.load(COSINE_SIM_PATH)
    hybrid_model = HybridModel.load(
        HYBRID_CONFIG_PATH,
        collaborative_model=collab_model,
        content_model=content_model
    )
    
    # Évaluer les modèles
    print("\n4. Évaluation des modèles (100 utilisateurs)...")
    k_values = EVALUATION_CONFIG['k_values']
    
    all_results = []
    
    # Collaborative
    
    with mlflow.start_run(run_name="Evaluation Collaborative"):
        
        mlflow.log_params(SVD_CONFIG , {
            "model_type" : 'Collaborative',
            "Sample users" : 100,
            "k_values" : str(k_values)
        })
        
        collab_results = evaluate_model(
            collab_model, "Collaborative", 
            train_ratings, test_ratings, movies, k_values
        )
        all_results.append(collab_results)

        for k in k_values :
            subset = collab_results[collab_results['k'] == k]
            mlflow.log_metric(f"precision_at_{k}", subset['precision'].mean())
            mlflow.log_metric(f"recall_at_{k}", subset['recall'].mean())
            mlflow.log_metric(f"ndcg_at_{k}", subset['ndcg'].mean())
        
    
    
    # Content-Based
    with mlflow.start_run(run_name="Evaluation_ContentBased"):
        mlflow.log_param("model_type", "Content-Based")
        mlflow.log_param("sample_users", 100)
        mlflow.log_param("k_values", str(k_values))
        
        content_results = evaluate_model(
            content_model, "Content-Based",
            train_ratings, test_ratings, movies, k_values
        )
        all_results.append(content_results)
        
        # Logger les métriques moyennes
        for k in k_values:
            subset = content_results[content_results['k'] == k]
            mlflow.log_metric(f"precision_at_{k}", subset['precision'].mean())
            mlflow.log_metric(f"recall_at_{k}", subset['recall'].mean())
            mlflow.log_metric(f"ndcg_at_{k}", subset['ndcg'].mean())
    
    # Hybrid
    with mlflow.start_run(run_name="Evaluation_Hybrid"):
        mlflow.log_param("model_type", "Hybrid")
        mlflow.log_param("sample_users", 100)
        mlflow.log_param("k_values", str(k_values))
        
        hybrid_results = evaluate_model(
            hybrid_model, "Hybrid",
            train_ratings, test_ratings, movies, k_values
        )
        all_results.append(hybrid_results)
        
        # Logger les métriques moyennes
        for k in k_values:
            subset = hybrid_results[hybrid_results['k'] == k]
            mlflow.log_metric(f"precision_at_{k}", subset['precision'].mean())
            mlflow.log_metric(f"recall_at_{k}", subset['recall'].mean())
            mlflow.log_metric(f"ndcg_at_{k}", subset['ndcg'].mean())
    
    # Combiner et afficher résultats
    print("\n" + "="*70)
    print("RÉSULTATS")
    print("="*70)
    
    all_df = pd.concat(all_results, ignore_index=True)
    summary = all_df.groupby(['model', 'k']).agg({
        'precision': 'mean',
        'recall': 'mean',
        'ndcg': 'mean'
    }).round(4)
    
    print("\n", summary)
    
    # Meilleur modèle
    print("\n" + "="*70)
    print("MEILLEUR MODÈLE PAR MÉTRIQUE")
    print("="*70)
    
    for k in k_values:
        subset = summary.loc[(slice(None), k), :]
        print(f"\n@K={k} :")
        
        best_precision = subset['precision'].idxmax()
        best_recall = subset['recall'].idxmax()
        best_ndcg = subset['ndcg'].idxmax()
        
        print(f"  Precision : {best_precision[0]} ({subset.loc[best_precision, 'precision']:.4f})")
        print(f"  Recall : {best_recall[0]} ({subset.loc[best_recall, 'recall']:.4f})")
        print(f"  NDCG : {best_ndcg[0]} ({subset.loc[best_ndcg, 'ndcg']:.4f})")


if __name__ == "__main__":
    main()