import numpy as np
import pandas as pd
import joblib

from logistic_regression import predict
from metrics import accuracy, confusion_matrix, missed_faults


# Load the test datasets
test_R = pd.read_csv("results/test.csv")
test_Q = pd.read_csv("results/test_Q.csv")

features_R = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

features_Q = features_R + [
    "Ia_sq", "Ib_sq", "Ic_sq",
    "Va_sq", "Vb_sq", "Vc_sq"
]


def evaluate_model(name, X, y, model_type, model):
    if model_type == "logistic":
        X = np.column_stack((np.ones(len(X)), X))
        predictions = predict(X, model)
    else:
        predictions = model.predict(X)

    acc = accuracy(y, predictions)
    cm = confusion_matrix(y, predictions)
    missed = missed_faults(y, predictions)

    tn, fp = cm[0]
    fn, tp = cm[1]

    print(f"\n{name}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print(f"True Negatives: {tn}")
    print(f"False Alarms: {fp}")
    print(f"Missed Faults: {missed}")
    print(f"True Positives: {tp}")

    return {
        "Model": name,
        "Accuracy (%)": round(acc * 100, 2),
        "TN": int(tn),
        "False Alarms": int(fp),
        "Missed Faults": int(missed),
        "TP": int(tp)
    }


results = []

# Logistic Regression - Set R
X_test_R = test_R[features_R].to_numpy()
y_test_R = test_R["fault"].to_numpy()
theta_R = np.load("results/theta_R.npy")

results.append(
    evaluate_model(
        "Logistic Regression, Set R",
        X_test_R, y_test_R, "logistic", theta_R
    )
)

# Logistic Regression - Set Q
X_test_Q = test_Q[features_Q].to_numpy()
y_test_Q = test_Q["fault"].to_numpy()
theta_Q = np.load("results/theta_Q.npy")

results.append(
    evaluate_model(
        "Logistic Regression, Set Q",
        X_test_Q, y_test_Q, "logistic", theta_Q
    )
)

# Linear SVM
linear_svm = joblib.load("results/best_linear_svm.pkl")

results.append(
    evaluate_model(
        "Linear SVM",
        X_test_R, y_test_R, "svm", linear_svm
    )
)

# RBF SVM
rbf_svm = joblib.load("results/best_rbf_svm.pkl")

results.append(
    evaluate_model(
        "RBF SVM",
        X_test_R, y_test_R, "svm", rbf_svm
    )
)

# Save the final test results
results_df = pd.DataFrame(results)
results_df.to_csv("results/test_results.csv", index=False)

print("\nFinal test results saved to results/test_results.csv")