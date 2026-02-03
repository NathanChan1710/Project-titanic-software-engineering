import pandas as pd
import os

def prepare_data(input_path, output_path):
    """Charge un CSV et le sauvegarde dans le dossier interim."""
    # Lecture du fichier
    df = pd.read_csv(input_path)
    
    # Création du dossier de destination s'il n'existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sauvegarde selon tes critères
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    return df

# Exécution pour tes fichiers réels
if __name__ == "__main__":
    prepare_data("./data/raw/train.csv", "./data/interim/train_clean.csv")
    prepare_data("./data/raw/test.csv", "./data/interim/test_clean.csv")