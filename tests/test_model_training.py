"""
Tests unitaires pour le module model_training.
"""

import pandas as pd

from model_training import train_and_predict


def test_train_and_predict_flow():
    """
    Vérifie le flux complet d'entraînement et de prédiction du modèle.
    """
    train_df = pd.DataFrame(
        {
            "Survived": [1, 0, 1, 0],
            "Pclass": [1, 3, 1, 3],
            "Sex": ["female", "male", "female", "male"],
            "SibSp": [0, 1, 0, 1],
            "Parch": [0, 0, 0, 0],
        }
    )

    test_df = pd.DataFrame(
        {
            "PassengerId": [892, 893],
            "Pclass": [3, 1],
            "Sex": ["male", "female"],
            "SibSp": [0, 0],
            "Parch": [0, 0],
        }
    )

    features = ["Pclass", "Sex", "SibSp", "Parch"]

    predictions = train_and_predict(train_df, test_df, features)

    assert len(predictions) == 2
    assert set(predictions).issubset({0, 1})
