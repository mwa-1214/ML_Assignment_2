
import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def hypothesis(X, theta):
    return sigmoid(X @ theta)


def cost(X, y, theta):
    m = len(y)
    h = hypothesis(X, theta)

    # Clip predictions before taking logarithms
    h = np.clip(h, 1e-12, 1 - 1e-12)

    total_cost = -np.sum(
        y * np.log(h) + (1 - y) * np.log(1 - h)
    )

    return total_cost / m

def fit_logistic(X, y, alpha, n_iters):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for _ in range(n_iters):
        h = hypothesis(X, theta)
        gradient = (X.T @ (h - y)) / m
        theta = theta - alpha * gradient
        cost_history.append(cost(X, y, theta))

    return theta, cost_history
def predict(X, theta):
    probabilities = hypothesis(X, theta)
    return (probabilities >= 0.5).astype(int)

if __name__ == "__main__":
    X_test = np.array([
        [1, 0.5, 1.2],
        [1, 1.0, 0.8],
        [1, 1.5, 0.4],
        [1, 2.0, 1.0]
    ])

    y_test = np.array([0, 0, 1, 1])
    theta_test = np.zeros(3)

    print("Sigmoid test:", sigmoid(np.array([0.0])))
    print("Hypothesis test:", hypothesis(X_test, theta_test))
    print("Cost test:", cost(X_test, y_test, theta_test))
    theta_result, cost_history = fit_logistic(
        X_test, y_test, alpha=0.1, n_iters=100
    )

    print("\nGradient descent test:")
    print("Final theta:", theta_result)
    print("Initial cost:", cost_history[0])
    print("Final cost:", cost_history[-1])
    predictions = predict(X_test, theta_result)

    print("\nPrediction test:")
    print("Predictions:", predictions)
    print("Actual labels:", y_test)
