import pandas as pd
import pytest
from model_evaluation import calculate_survival_rates

def test_calculate_survival_rates():
    # 1. Création d'un mini DataFrame de test
    # 2 femmes (1 survit), 2 hommes (0 survit)
    test_df = pd.DataFrame({
        "Sex": ["female", "female", "male", "male"],
        "Survived": [1, 0, 0, 0]
    })
    
    # 2. Appel de la fonction
    rate_women, rate_men = calculate_survival_rates(test_df)
    
    # 3. Vérifications (Assertions)
    assert rate_women == 0.5  # 1 sur 2
    assert rate_men == 0.0    # 0 sur 2
    assert isinstance(rate_women, float)

def test_calculate_survival_empty():
    """Test de sécurité si le DataFrame est vide"""
    empty_df = pd.DataFrame(columns=["Sex", "Survived"])
    rate_women, rate_men = calculate_survival_rates(empty_df)
    assert rate_women == 0
    assert rate_men == 0