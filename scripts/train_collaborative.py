"""
Script pour entraîner le modèle Collaborative Filtering (SVD)
"""
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour importer src
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_data
from src.models.collaborative import CollaborativeModel
from src.config import SVD_MODEL_PATH , MLFLOW_EXPERIMENT_NAME , MLFLOW_TRACKING_URI , SVD_CONFIG
from surprise.model_selection import train_test_split
import mlflow
import mlflow.sklearn


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)



def main():
    """
    Entraîne et sauvegarde le modèle Collaborative
    """
    print("="*70)
    print("ENTRAÎNEMENT DU MODÈLE COLLABORATIVE FILTERING (SVD)")
    print("="*70)
    
    ## Démarer mlflow :
    
    with mlflow.start_run(run_name='Collaborative_SVD') :
    
        # Charger les données
        print("\n1. Chargement des données...")
        ratings, movies = load_data()
        
        ## Logger les parametres du Modele sur Mlflow :
        
        mlflow.log_params(
            SVD_CONFIG , {'algorithme' : 'SVD' , 'test_size' : 0.2}
        )
        
        
        # Créer et entraîner le modèle
        print("\n2. Création du modèle...")
        model = CollaborativeModel()
        
        print("\n3. Entraînement...")
        metrics = model.fit_and_evaluate(ratings , test_size=0.2)
        
        # Logger les métriques dans MLflow
        print(f"\n4. Métriques :")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        
        mlflow.log_metric("rmse", metrics['rmse'])
        mlflow.log_metric("mae", metrics['mae'])
        # Sauvegarder le modèle
        print("\n4. Sauvegarde du modèle...")
        model.save(SVD_MODEL_PATH)
        
        mlflow.sklearn.log_model(
            sk_model=model.algo,
            artifact_path='collaborative_model',
            registered_model_name='CollaborativeModel'
        )
        
        
        print("\n" + "="*70)
        print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
        print("="*70)


if __name__ == "__main__":
    main()