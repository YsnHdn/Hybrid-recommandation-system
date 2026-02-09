"""
Schémas Pydantic pour la validation des données de l'API
"""
from pydantic import BaseModel, Field
from typing import List, Optional

from sqlalchemy import desc


class HealthResponse(BaseModel):
    """
    Réponse du endpoint health
    """
    status : str
    message : str
    models_loaded : dict = {}
    

class RecommendationRequest(BaseModel):
    
    """
    Requêtes pour obtenir des recommandations
    """
    
    user_id : int = Field(... , description="ID de l'utilisateur" , ge = 1)
    n : int = Field(... , description="Nombre de recommandation" , ge = 1 , le = 50)
    model_type : str = Field("hybrid" , description="type du modèle : collaborative , content , hybrid" )
    
    
    class Config :
        json_schema_extra = {
            "example" : {
                'user_id' : 10,
                'n' : 5 ,
                'model_type' : 'hybrid'  
            }
            
        }
    
class MovieRecommendation(BaseModel):
    """
    Une recommandation de film
    """
    item_id: int
    title: str
    score: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "item_id": 50,
                "title": "Star Wars (1977)",
                "score": 4.5
            }
        }
    
class RecommendationResponse(BaseModel):
    """
    Réponse contenant les recommandations
    """
    user_id: int
    model_type: str
    recommendations: List[MovieRecommendation]
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "model_type": "hybrid",
                "recommendations": [
                    {"item_id": 50, "title": "Star Wars (1977)", "score": 4.5},
                    {"item_id": 181, "title": "Return of the Jedi (1983)", "score": 4.3}
                ]
            }
        }

class SimilarItemsRequest(BaseModel):
    """
    Requête pour obtenir des films similaires
    """
    item_id: int = Field(..., description="ID du film de référence", ge=1)
    n: int = Field(10, description="Nombre de films similaires", ge=1, le=50)
    
    class Config:
        json_schema_extra = {
            "example": {
                "item_id": 50,
                "n": 10
            }
        }