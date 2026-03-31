# ==============================
# CSCC11 Group 6
# Decision Tree for Diabetes Risk Prediction
# Member: Irfan Ahmed; Yixi Li; Yuyao Wu; Zain Rizvi
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns

# ----------------------
# Load Dataset
# ----------------------
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1008015354, stratify=y
)

dt_base = DecisionTreeClassifier(random_state=1008015354)
dt_base.fit(X_train, y_train)
y_pred_base = dt_base.predict(X_test)

# ----------------------
# Hyperparameter Tuning (GridSearchCV)
# ----------------------
param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=1008015354),
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# Best model
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)
y_pred_proba = best_dt.predict_proba(X_test)[:, 1]

def evaluate(y_true, y_pred, y_pred_proba, model_name):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"=== {model_name} ===")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"ROC AUC:         {auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("-" * 50)

# Evaluate both base and tuned model
evaluate(y_test, y_pred_base, dt_base.predict_proba(X_test)[:,1], "Base Decision Tree")
evaluate(y_test, y_pred, y_pred_proba, "Tuned Decision Tree")

# ----------------------
# Visualization 
# ----------------------
plt.figure(figsize=(20, 10))
plot_tree(
    best_dt,
    filled=True,
    feature_names=X.columns,
    class_names=["No Diabetes", "Diabetes"],
    rounded=True,
    fontsize=9
)
plt.title("Optimized Decision Tree for Diabetes Prediction", fontsize=16)
plt.savefig("decision_tree_plot.png", dpi=300, bbox_inches="tight")
plt.close()

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_dt.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10))
plt.title("Top 10 Important Features (Decision Tree)", fontsize=14)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

print("Best Hyperparameters from Grid Search:")
print(grid_search.best_params_)