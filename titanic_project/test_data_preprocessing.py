import os
import pandas as pd
import pytest
from data_preprocessing import prepare_data 

def test_prepare_data_logic():
    # Chemins : on utilise ".." pour remonter d'un cran
    tmp_input = "tmp_raw.csv"
    # Cela va créer/utiliser le dossier data à la racine de Project-titanic-software-engineering
    tmp_output = "../data/interim/tmp_test_clean.csv" 
    
    # 1. Création du mini CSV de test (localement dans titanic_project)
    df_test = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df_test.to_csv(tmp_input, index=False)
    
    # 2. Appel de la fonction
    result = prepare_data(tmp_input, tmp_output)
    
    # 3. Assertions
    assert os.path.exists(tmp_output)  # Vérifie si le fichier est bien remonté dans ../data/
    assert result.shape == (2, 2)
    
    # 4. Nettoyage
    if os.path.exists(tmp_input): os.remove(tmp_input)
    if os.path.exists(tmp_output): os.remove(tmp_output)