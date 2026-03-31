
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = "Dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

def load_data(file_path):
    df = pd.read_csv(file_path)

    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    return X, y

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1008015354, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test

def train_models(X_train, X_train_scaled, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state = 1008015354),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state = 1008015354),
        "Gradient Boosting": GradientBoostingClassifier(random_state = 1008015354)
    }
    models["Logistic Regression"].fit(X_train_scaled, y_train)
    models["KNN"].fit(X_train_scaled, y_train)
    models["Random Forest"].fit(X_train, y_train)
    models["Gradient Boosting"].fit(X_train, y_train)
    return models

def evaluate_models(models, X_test, X_test_scaled, y_test):
    results = {}
    for name, model in models.items():
        if name in ["Logistic Regression", "KNN"]:
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"\n{name}")
        print("-" * 40)
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    return results

def plot_results(results):
    model_names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, accuracies)
    plt.ylabel("Accuracy")
    plt.title("Model Comparison on Diabetes Dataset")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X, y = load_data(file_path)

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(X, y)

    models = train_models(X_train, X_train_scaled, y_train)

    results = evaluate_models(models, X_test, X_test_scaled, y_test)

    plot_results(results)