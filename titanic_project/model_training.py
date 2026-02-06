"""
Modèle d'entraînement de notre projet Titanic.

Contient les fonctions nécessaires à l'entraînement d'un
modèle de classification et à la génération des prédictions
sur notre jeu de données de test.
"""

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]):
    """
    Entraîne un modèle de classification et génère des prédictions.

    Prépare les variables explicatives à l'aide de l'encodage one-hot,
    entraîne un modèle RandomForest puis génère les prédictions sur le
    jeu de données de test.

    Parameters
    ----------
    train_df : pd.DataFrame
        Jeu de données d'entraînement contenant la variable cible 'Survived'.
    test_df : pd.DataFrame
        Jeu de données de test sans la variable cible.
    features : list[str]
        Liste des variables explicatives à utiliser.
    """
    y = train_df["Survived"]
    x_train = pd.get_dummies(train_df[features])
    x_test = pd.get_dummies(test_df[features])

    # S'assurer que x_train et x_test ont les mêmes colonnes après get_dummies
    x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(x_train, y)
    predictions = model.predict(x_test)

    return predictions


if __name__ == "__main__":
    # Chemins avec ".." pour sortir de titanic_project
    train_data = pd.read_csv("../data/interim/train_clean.csv")
    test_data = pd.read_csv("../data/interim/test_clean.csv")

    features_ = ["Pclass", "Sex", "SibSp", "Parch"]
    preds = train_and_predict(train_data, test_data, features_)

    # Sauvegarde
    output_path = "../reports/submission.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": preds})
    output.to_csv(output_path, index=False)
    print("Your submission was successfully saved!")
