"""
Tests unitaires pour le module data_preprocessing.
"""

import os
import pandas as pd

from titanic_project.data_preprocessing import prepare_data



def test_prepare_data_logic():
    """
    Vérifie le chargement et la sauvegarde des données par prepare_data.
    """
    tmp_input = "tmp_raw.csv"
    tmp_output = "../data/interim/tmp_test_clean.csv"

    df_test = pd.DataFrame(
        {
            "col1": [1, 2],
            "col2": ["a", "b"],
        }
    )
    df_test.to_csv(tmp_input, index=False)

    result = prepare_data(tmp_input, tmp_output)

    assert os.path.exists(tmp_output)
    assert result.shape == (2, 2)

    if os.path.exists(tmp_input):
        os.remove(tmp_input)

    if os.path.exists(tmp_output):
        os.remove(tmp_output)
