import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1008015354,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,   # 0.25 x 0.8 = 0.2 of total data
    random_state=1008015354,
    stratify=y_temp
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# tune C 
C_values = np.logspace(-3, 3, 20)
lr_train_acc = []
lr_val_acc = []

for C in C_values:
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=1008015354
    )
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    lr_train_acc.append(accuracy_score(y_train, y_train_pred))
    lr_val_acc.append(accuracy_score(y_val, y_val_pred))

best_lr_idx = np.argmax(lr_val_acc)
best_C = C_values[best_lr_idx]

print("Logistic Regression")
print("Best C =", best_C)
print("Train acc =", lr_train_acc[best_lr_idx])
print("Val acc =", lr_val_acc[best_lr_idx])

# plot train/val accuracy
plt.figure(figsize=(8, 5))
plt.semilogx(C_values, lr_train_acc, marker='o', label="Train Accuracy")
plt.semilogx(C_values, lr_val_acc, marker='o', label="Validation Accuracy")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.title("Logistic Regression: Accuracy vs C")
plt.legend()
plt.grid(True)
plt.show()

# retrain final model on train + val
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

scaler_final = StandardScaler()
X_trainval_scaled = scaler_final.fit_transform(X_trainval)
X_test_scaled_final = scaler_final.transform(X_test)

final_model = LogisticRegression(
    C=best_C,
    max_iter=1000,
    random_state=1008015354
)
final_model.fit(X_trainval_scaled, y_trainval)

y_test_pred = final_model.predict(X_test_scaled_final)

print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_test_pred))

# feature coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": final_model.coef_[0]
})

coef_df = coef_df.sort_values(by="Coefficient", ascending=False)

print("\nTop Positive Coefficients:")
print(coef_df.head(10))

print("\nTop Negative Coefficients:")
print(coef_df.tail(10))