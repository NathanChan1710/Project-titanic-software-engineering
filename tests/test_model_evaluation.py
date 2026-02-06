"""
Tests unitaires pour le module model_evaluation.
"""

import pandas as pd

from titanic_project.model_evaluation import calculate_survival_rates


def test_calculate_survival_rates():
    """
    Vérifie le calcul des taux de survie pour les femmes et les hommes.
    """
    test_df = pd.DataFrame(
        {
            "Sex": ["female", "female", "male", "male"],
            "Survived": [1, 0, 0, 0],
        }
    )

    rate_women, rate_men = calculate_survival_rates(test_df)

    assert rate_women == 0.5
    assert rate_men == 0.0
    assert isinstance(rate_women, float)


def test_calculate_survival_empty():
    """
    Vérifie le comportement de la fonction avec un DataFrame vide.
    """
    empty_df = pd.DataFrame(columns=["Sex", "Survived"])

    rate_women, rate_men = calculate_survival_rates(empty_df)

    assert rate_women == 0
    assert rate_men == 0
