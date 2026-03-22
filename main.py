import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Naresh\PycharmProjects\MachineLearningScratch\Salary Data.csv")
X = df['YearsExperience'].values
y = df['Salary'].values
m = len(y)

#cost function

def compute_cost(X, y, w, b):
    predictions =  np.dot(X, w) + b
    errors = predictions - y
    cost = (1/(2 * m)) * np.sum(np.square(errors))
    return cost

#gradient computation

def compute_gradient(X, y, w, b):
    predictions = np.dot(X, w) + b
    errors = predictions - y
    dw = (1/m)*np.dot(X, errors)
    db = (1/m)*np.sum(errors)
    return dw,db

#gradient descent

def gradient_descent(X, y, w, b, alpha, epochs):
    cost_history = []

    for epoch in range(epochs):
        dw, db = compute_gradient(X, y, w, b)

        w = w - alpha * dw
        b = b - alpha * db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Cost: {cost}, w: {w}, b: {b}")

    return w, b, cost_history

#train
w, b = 0.0, 0.0
alpha = 0.01
epochs = 1000

w_final, b_final, cost_history = gradient_descent(X, y, w, b, alpha, epochs)

print(f"Final w: {w_final}, b: {b_final}")

def predict(X, w, b):
    predictions = np.dot(X, w) + b
    return predictions

y_pred = predict(X, w_final, b_final)

#evaluation
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)

r2_score = 1 - (ss_res/ss_tot)
print(f"R2:  {r2_score}")

# ── 8. Plot Results ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Regression line
axes[0].scatter(X, y, color="steelblue", label="Actual")
axes[0].plot(X, y_pred, color="tomato", label=f"Fit: y = {w_final:.0f}x + {b_final:.0f}")
axes[0].set_xlabel("Years of Experience")
axes[0].set_ylabel("Salary")
axes[0].set_title("Linear Regression Fit")
axes[0].legend()

# Cost curve
axes[1].plot(cost_history, color="green")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Cost (MSE)")
axes[1].set_title("Cost vs Epochs")

plt.tight_layout()
plt.savefig("linear_regression_result.png", dpi=150)
plt.show()

