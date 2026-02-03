import pandas as pd
import pytest
from model_training import train_and_predict

def test_train_and_predict_flow():
    # 1. Création de données de test minimales
    train_df = pd.DataFrame({
        "Survived": [1, 0, 1, 0],
        "Pclass": [1, 3, 1, 3],
        "Sex": ["female", "male", "female", "male"],
        "SibSp": [0, 1, 0, 1],
        "Parch": [0, 0, 0, 0]
    })
    
    test_df = pd.DataFrame({
        "PassengerId": [892, 893],
        "Pclass": [3, 1],
        "Sex": ["male", "female"],
        "SibSp": [0, 0],
        "Parch": [0, 0]
    })
    
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    
    # 2. Exécution
    predictions = train_and_predict(train_df, test_df, features)
    
    # 3. Vérifications
    assert len(predictions) == 2  # On attend 2 prédictions pour 2 passagers test
    assert set(predictions).issubset({0, 1})  # Les prédictions doivent être 0 ou 1