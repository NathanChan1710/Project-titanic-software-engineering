import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv("./data/interim/train_clean.csv")
test_data = pd.read_csv("./data/interim/test_clean.csv")

women = train_data.loc[train_data.Sex == "female"]["Survived"]
rate_women = sum(women) / len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == "male"]["Survived"]
rate_men = sum(men) / len(men)

print("% of men who survived:", rate_men)

# --- Sauvegarde CSV ---
results = pd.DataFrame({"group": ["women", "men"], "survival_rate": [rate_women, rate_men]})

results.to_csv("./data/processed/survival_rates.csv", index=False)
