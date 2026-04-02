
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree


from sklearn.model_selection import train_test_split, GridSearchCV
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

def tune_models(X_train, X_train_scaled, y_train):
    tuned_models = {}

    #Log Reg
    lr_params = {
        "C": [0.01, 0.1, 1, 10, 100]
    }
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=1008015354),
        lr_params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    lr_grid.fit(X_train_scaled, y_train)
    tuned_models["Logistic Regression"] = lr_grid.best_estimator_
    print("Best Logistic Regression params:", lr_grid.best_params_)

    #KNN
    knn_params = {
        "n_neighbors": [3, 5, 7, 9, 11, 15]
    }
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    knn_grid.fit(X_train_scaled, y_train)
    tuned_models["KNN"] = knn_grid.best_estimator_
    print("Best KNN params:", knn_grid.best_params_)
    
    #dts
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=1008015354),
        {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        },
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )
    dt_grid.fit(X_train, y_train)
    tuned_models["Decision Tree"] = dt_grid.best_estimator_
    print("Best Decision Tree params:", dt_grid.best_params_)

    #rfs
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=1008015354),
        rf_params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    tuned_models["Random Forest"] = rf_grid.best_estimator_
    print("Best Random Forest params:", rf_grid.best_params_)

    #G boost
    gb_params = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5]
    }
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=1008015354),
        gb_params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    gb_grid.fit(X_train, y_train)
    tuned_models["Gradient Boosting"] = gb_grid.best_estimator_
    print("Best Gradient Boosting params:", gb_grid.best_params_)

    return tuned_models

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
    
def plot_confusion_matrices(models, X_test, X_test_scaled, y_test):
    for name, model in models.items():
        if name in ["Logistic Regression", "KNN"]:
            X_eval = X_test_scaled
        else:
            X_eval = X_test

        y_pred = model.predict(X_eval)
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix: {name}")
        plt.tight_layout()
        plt.show()

def plot_decision_tree(model, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        filled=True,
        feature_names=feature_names,
        class_names=["No Diabetes", "Diabetes"],
        rounded=True,
        fontsize=8
    )
    plt.title("Optimized Decision Tree for Diabetes Prediction")
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

    models = tune_models(X_train, X_train_scaled, y_train)

    results_df = evaluate_models(models, X_test, X_test_scaled, y_test)

    print("\nOverall Results Table")
    print(results_df.round(4))

    plot_roc_curves(models, X_test, X_test_scaled, y_test)
    plot_model_comparison(results_df)
    plot_confusion_matrices(models, X_test, X_test_scaled, y_test)
    #plot_decision_tree_importance(models["Decision Tree"], X.columns)
    plot_decision_tree(models["Decision Tree"], X.columns)
    plot_random_forest_importance(models["Random Forest"], X.columns)