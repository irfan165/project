import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier, plot_importance

# 1. Load dataset
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

print("\nStatistical Summary:")
print(df.describe())

# 2. Features and target
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# 3. Split into train+val and test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1008015354,
    stratify=y
)

# 4. Split train and val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,   # 0.25 * 0.8 = 0.2 total
    random_state=1008015354,
    stratify=y_temp
)

# 5. tuning
depth_values = [4, 5, 6]
n_estimators_values = [200, 300, 500]
learning_rate_values = [0.03, 0.05, 0.1]
subsample_values = [0.8, 0.9]
colsample_values = [0.8, 0.9]

best_val_acc = -1
best_params = None
best_model = None

results = []

for depth in depth_values:
    for n_estimators in n_estimators_values:
        for lr in learning_rate_values:
            for subsample in subsample_values:
                for colsample in colsample_values:
                    model = XGBClassifier(
                        objective="binary:logistic",
                        max_depth=depth,
                        n_estimators=n_estimators,
                        learning_rate=lr,
                        subsample=subsample,
                        colsample_bytree=colsample,
                        random_state=1008015354,
                        eval_metric="logloss"
                    )

                    model.fit(X_train, y_train)

                    y_train_pred = model.predict(X_train)
                    y_val_pred = model.predict(X_val)

                    train_acc = accuracy_score(y_train, y_train_pred)
                    val_acc = accuracy_score(y_val, y_val_pred)

                    results.append({
                        "max_depth": depth,
                        "n_estimators": n_estimators,
                        "learning_rate": lr,
                        "subsample": subsample,
                        "colsample_bytree": colsample,
                        "train_acc": train_acc,
                        "val_acc": val_acc
                    })

                    print(
                        f"depth={depth}, n_estimators={n_estimators}, "
                        f"learning_rate={lr}, subsample={subsample}, "
                        f"colsample={colsample}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
                    )

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_params = {
                            "max_depth": depth,
                            "n_estimators": n_estimators,
                            "learning_rate": lr,
                            "subsample": subsample,
                            "colsample_bytree": colsample
                        }
                        best_model = model

print("\nBest Parameters:")
print(best_params)
print("Best Validation Accuracy:", best_val_acc)

# 6. Retrain final model on train + val
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

final_model = XGBClassifier(
    objective="binary:logistic",
    max_depth=best_params["max_depth"],
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    random_state=1008015354,
    eval_metric="logloss"
)

final_model.fit(X_trainval, y_trainval)

# 7. Test evaluation
y_test_pred = final_model.predict(X_test)

test_acc = accuracy_score(y_test, y_test_pred)
print("\nXGBoost Test Accuracy:", test_acc)

print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# 8. Confusion matrix heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9. Validation accuracy plot (optional: grouped by depth)
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
for depth in depth_values:
    subset = results_df[
        (results_df["max_depth"] == depth) &
        (results_df["learning_rate"] == 0.05)
    ].sort_values("n_estimators")
    plt.plot(
        subset["n_estimators"],
        subset["val_acc"],
        marker="o",
        label=f"max_depth={depth}, lr=0.05"
    )

plt.xlabel("n_estimators")
plt.ylabel("Validation Accuracy")
plt.title("XGBoost Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# 10. Feature importance by gain
plt.figure(figsize=(8, 6))
plot_importance(final_model, importance_type="gain", max_num_features=10)
plt.title("XGBoost Feature Importance (gain)")
plt.show()