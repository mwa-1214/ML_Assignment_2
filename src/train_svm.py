import numpy as np
import pandas as pd
from sklearn.svm import SVC
import joblib

# Load Set R training and validation data
train_data = pd.read_csv("results/train.csv")
val_data = pd.read_csv("results/validation.csv")

features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

X_train = train_data[features].to_numpy()
y_train = train_data["fault"].to_numpy()

X_val = val_data[features].to_numpy()
y_val = val_data["fault"].to_numpy()

C_values = [0.1, 1, 10, 100]

results = []
best_linear_model = None
best_linear_accuracy = -1
best_linear_C = None

# Train and evaluate Linear SVM models
for C in C_values:
    model = SVC(kernel="linear", C=C)
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    accuracy = np.mean(predictions == y_val) * 100
    support_vectors = model.n_support_.sum()

    results.append({
        "Kernel": "linear",
        "C": C,
        "gamma": "-",
        "Validation Accuracy (%)": round(accuracy, 2),
        "Support Vectors": int(support_vectors)
    })

    print(
        f"Linear SVM | C={C} | "
        f"Validation Accuracy={accuracy:.2f}% | "
        f"Support Vectors={support_vectors}"
    )

    if accuracy > best_linear_accuracy:
        best_linear_accuracy = accuracy
        best_linear_model = model
        best_linear_C = C

# Save the best Linear SVM
joblib.dump(best_linear_model, "results/best_linear_svm.pkl")

print("\nBest Linear SVM")
print("C:", best_linear_C)
print("Validation accuracy:", round(best_linear_accuracy, 2), "%")

# Save the linear SVM results table
pd.DataFrame(results).to_csv(
    "results/svm_validation.csv", index=False
)

print("\nLinear SVM results saved.")
# RBF SVM: test all C and gamma combinations
gamma_values = [0.1, 1]

best_rbf_model = None
best_rbf_accuracy = -1
best_rbf_C = None
best_rbf_gamma = None

for C in C_values:
    for gamma in gamma_values:
        model = SVC(kernel="rbf", C=C, gamma=gamma)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        accuracy = np.mean(predictions == y_val) * 100
        support_vectors = model.n_support_.sum()

        results.append({
            "Kernel": "rbf",
            "C": C,
            "gamma": gamma,
            "Validation Accuracy (%)": round(accuracy, 2),
            "Support Vectors": int(support_vectors)
        })

        print(
            f"RBF SVM | C={C}, gamma={gamma} | "
            f"Validation Accuracy={accuracy:.2f}% | "
            f"Support Vectors={support_vectors}"
        )

        if accuracy > best_rbf_accuracy:
            best_rbf_accuracy = accuracy
            best_rbf_model = model
            best_rbf_C = C
            best_rbf_gamma = gamma

# Save the best RBF SVM model
joblib.dump(best_rbf_model, "results/best_rbf_svm.pkl")

# Save results for both Linear and RBF SVM
pd.DataFrame(results).to_csv(
    "results/svm_validation.csv", index=False
)

print("\nBest RBF SVM")
print("C:", best_rbf_C)
print("Gamma:", best_rbf_gamma)
print("Validation accuracy:", round(best_rbf_accuracy, 2), "%")

print("\nBoth SVM models and validation results saved.")