import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra

train_data = pd.read_csv("./data/external/train.csv")
train_data.head()

test_data = pd.read_csv("./data/external/test.csv")
test_data.head()

# on a deja fait, on a arrangé le chemin

train_data.to_csv("./data/processed/train_clean.csv", index=False, encoding="utf-8")


test_data.to_csv("./data/processed/test_clean.csv", index=False, encoding="utf-8")