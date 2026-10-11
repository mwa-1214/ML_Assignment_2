
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import fit_logistic, predict


alphas = [0.01, 0.1, 1, 10]
n_iters = 2000

# Load standardized training data for Set R
train_R = pd.read_csv("results/train.csv")

# Load standardized training data for Set Q
train_Q = pd.read_csv("results/train_Q.csv")

features_R = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

features_Q = features_R + [
    "Ia_sq", "Ib_sq", "Ic_sq",
    "Va_sq", "Vb_sq", "Vc_sq"
]


def run_experiment(data, features, set_name):
    X = data[features].to_numpy()
    y = data["fault"].to_numpy()

    # Add intercept column
    X = np.column_stack((np.ones(len(X)), X))

    results = {}

    for alpha in alphas:
        theta, cost_history = fit_logistic(
            X, y, alpha=alpha, n_iters=n_iters
        )

        results[alpha] = cost_history
        costs = np.array(cost_history)

        decreases_every_time = np.all(np.diff(costs) < 0)

        print(
            f"{set_name}, alpha={alpha}: "
            f"cost decreases every iteration: "
            f"{decreases_every_time}"
        )

        print(
            f"{set_name}, alpha={alpha}: "
            f"initial cost={cost_history[0]:.6f}, "
            f"final cost={cost_history[-1]:.6f}"
        )

    plt.figure(figsize=(9, 6))

    for alpha, history in results.items():
        plt.plot(history, label=f"alpha={alpha}")

    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.title(f"Learning Rate Comparison - {set_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"results/cost_{set_name}.png")
    plt.show()


run_experiment(train_R, features_R, "Set_R")
run_experiment(train_Q, features_Q, "Set_Q")

# Selected learning rates
best_alpha_R = 1
best_alpha_Q = 10

print("\nSelected learning rates:")
print("Set R:", best_alpha_R)
print("Set Q:", best_alpha_Q)

# Load validation datasets
val_R = pd.read_csv("results/validation.csv")
val_Q = pd.read_csv("results/validation_Q.csv")


def evaluate_model(train_data, val_data, features, alpha, set_name):
    # Prepare training data
    X_train = train_data[features].to_numpy()
    y_train = train_data["fault"].to_numpy()

    X_train = np.column_stack((np.ones(len(X_train)), X_train))

    # Train final model
    theta, cost_history = fit_logistic(
        X_train, y_train, alpha=alpha, n_iters=2000
    )

    # Training accuracy
    train_predictions = predict(X_train, theta)
    train_accuracy = np.mean(train_predictions == y_train) * 100

    # Prepare validation data
    X_val = val_data[features].to_numpy()
    y_val = val_data["fault"].to_numpy()

    X_val = np.column_stack((np.ones(len(X_val)), X_val))

    val_predictions = predict(X_val, theta)
    val_accuracy = np.mean(val_predictions == y_val) * 100

    print(f"\n{set_name} Final Results")
    print("Learning rate:", alpha)
    print("Final cost:", cost_history[-1])
    print("Training accuracy:", round(train_accuracy, 2), "%")
    print("Validation accuracy:", round(val_accuracy, 2), "%")
    print("Theta:", theta)

    return theta


theta_R = evaluate_model(
    train_R, val_R, features_R, best_alpha_R, "Set R"
)

theta_Q = evaluate_model(
    train_Q, val_Q, features_Q, best_alpha_Q, "Set Q"
)

# Save final model parameters
np.save("results/theta_R.npy", theta_R)
np.save("results/theta_Q.npy", theta_Q)

print("\nFinal model parameters saved successfully.")
# Load test datasets for final evaluation
test_R = pd.read_csv("results/test.csv")
test_Q = pd.read_csv("results/test_Q.csv")


def evaluate_test(data, features, theta, set_name):
    X_test = data[features].to_numpy()
    y_test = data["fault"].to_numpy()

    # Add intercept column
    X_test = np.column_stack((np.ones(len(X_test)), X_test))

    # Predict using the saved final model parameters
    predictions = predict(X_test, theta)

    test_accuracy = np.mean(predictions == y_test) * 100

    print(f"\n{set_name} Test Results")
    print("Test accuracy:", round(test_accuracy, 2), "%")


evaluate_test(test_R, features_R, theta_R, "Set R")
evaluate_test(test_Q, features_Q, theta_Q, "Set Q")