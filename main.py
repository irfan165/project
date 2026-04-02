
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

file_path = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"


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
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=1008015354),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=1008015354),
        "Gradient Boosting": GradientBoostingClassifier(random_state=1008015354)
    }

    models["Logistic Regression"].fit(X_train_scaled, y_train)
    models["KNN"].fit(X_train_scaled, y_train)
    models["Random Forest"].fit(X_train, y_train)
    models["Gradient Boosting"].fit(X_train, y_train)

    return models


def evaluate_models(models, X_test, X_test_scaled, y_test):
    results = []

    for name, model in models.items():
        if name in ["Logistic Regression", "KNN"]:
            X_eval = X_test_scaled
        else:
            X_eval = X_test

        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"\n{name}")
        print("-" * 40)
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": report["macro avg"]["precision"],
            "Recall": report["macro avg"]["recall"],
            "F1": report["macro avg"]["f1-score"],
            "ROC-AUC": roc_auc
        })

    results_df = pd.DataFrame(results)
    return results_df


def plot_model_comparison(results_df):
    metrics_to_plot = ["Accuracy", "F1", "ROC-AUC"]

    plot_df = results_df.set_index("Model")[metrics_to_plot]

    plot_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Comparison of Model Performance")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models, X_test, X_test_scaled, y_test):
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        if name in ["Logistic Regression", "KNN"]:
            X_eval = X_test_scaled
        else:
            X_eval = X_test

        y_prob = model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)

        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Diabetes Prediction Models")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_random_forest_importance(model, feature_names):
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = load_data(file_path)

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(X, y)

    models = train_models(X_train, X_train_scaled, y_train)

    results_df = evaluate_models(models, X_test, X_test_scaled, y_test)

    print("\nOverall Results Table")
    print(results_df.round(4))
    plot_model_comparison(results_df)    
    plot_roc_curves(models, X_test, X_test_scaled, y_test)
    plot_random_forest_importance(models["Random Forest"], X.columns)


