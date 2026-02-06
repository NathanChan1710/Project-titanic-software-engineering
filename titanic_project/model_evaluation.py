"""
Model d’évaluation du projet Titanic.

Contient les fonctions nécessaires au calcul des
taux de survie à partir des données.
"""

import pandas as pd
import os

def calculate_survival_rates(df):
    """
    Calcule les taux de survie en fonction du genre.

    On utilise les colonnes 'Sex' et 'Survived'.
    """


    women = df.loc[df.Sex == "female"]["Survived"]
    rate_women = sum(women) / len(women) if len(women) > 0 else 0

    men = df.loc[df.Sex == "male"]["Survived"]
    rate_men = sum(men) / len(men) if len(men) > 0 else 0
    
    return rate_women, rate_men

# CE BLOC EMPECHE L'ERREUR PENDANT LE TEST
if __name__ == "__main__":
    train_data = pd.read_csv("../data/interim/train_clean.csv")
    
    r_women, r_men = calculate_survival_rates(train_data)
    
    # Sauvegarde
    output_path = "../data/processed/survival_rates.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = pd.DataFrame({"group": ["women", "men"], "survival_rate": [r_women, r_men]})
    results.to_csv(output_path, index=False)
    print("Calcul terminé et sauvegardé.")
