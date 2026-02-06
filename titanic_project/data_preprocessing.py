"""
Module de prétraitement des données pour notre projet "Titanic".

Ici on charge nos données brutes CSV et on le sauvegarde pour pouvoir éxécuter les étapes suivantes.
"""

import pandas as pd
import os


def prepare_data(input_path, output_path):
    """Charge un CSV et le sauvegarde dans le dossier interim.
    Cette fonction lit un fichier CSV depuis le chemin d'entrée,
    crée le dossier de sortie s'il n'existe pas, puis enregistre
    le jeu de données dans le dossier interim qui est notre dossier
    de sauvegarde des tables.

    On charge nos données sous forme de DataFrame pandas."""

    # Lecture du fichier
    df = pd.read_csv(input_path)

    # Création du dossier de destination s'il n'existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sauvegarde
    df.to_csv(output_path, index=False, encoding="utf-8")

    return df


# Exécution pour tes fichiers réels
if __name__ == "__main__":
    prepare_data("./data/raw/train.csv", "./data/interim/train_clean.csv")
    prepare_data("./data/raw/test.csv", "./data/interim/test_clean.csv")
