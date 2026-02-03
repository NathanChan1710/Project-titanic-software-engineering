import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

def train_and_predict(train_df, test_df, features):
    """Logique d'entraînement et de prédiction."""
    y = train_df["Survived"]
    X = pd.get_dummies(train_df[features])
    X_test = pd.get_dummies(test_df[features])
    
    # S'assurer que X et X_test ont les mêmes colonnes après get_dummies
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)
    
    return predictions

if __name__ == "__main__":
    # Chemins avec ".." pour sortir de titanic_project
    train_data = pd.read_csv("../data/interim/train_clean.csv")
    test_data = pd.read_csv("../data/interim/test_clean.csv")

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    preds = train_and_predict(train_data, test_data, features)

    # Sauvegarde
    output_path = "../data/reports/submission.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": preds})
    output.to_csv(output_path, index=False)
    print("Your submission was successfully saved!")