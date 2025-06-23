import numpy as np
import pandas as pd

# Values for the single training example
# Inputs: 
x1, x2 = 0.04, 0.20
y_true = 0.50
learning_rate = 0.4

# Initial weights w1 through w6
weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(w, x1, x2):
    # Weighted sum inputs
    z1 = w[0]*x1 + w[1]*x2
    z2 = w[2]*x1 + w[3]*x2

    # Nodes in the hidden layer
    h1 = sigmoid(z1)
    h2 = sigmoid(z2)

    # Weighted sum inputs
    z3 = w[4]*h1 + w[5]*h2

    # Activation funtion in output layer (y_hat = prediction)
    y_hat = sigmoid(z3)
    return y_hat, z1, z2, z3, h1, h2

# Squared error E=1/2(y-ÿ)^2 (ÿ = y hat)
def compute_error(y, y_hat):
    return 0.5 * (y - y_hat)**2

# Training loop
errors = []
max_iterations = 10000
threshold = 1e-6
w = weights.copy()

for i in range(max_iterations):
    y_hat, z1, z2, z3, h1, h2 = forward_pass(w, x1, x2)
    error = compute_error(y_true, y_hat)
    errors.append(error)
    
    # Check stopping condition (10^-6)
    if i > 0 and abs(errors[i] - errors[i-1]) < threshold:
        break
    
    # Backpropagation

    # Output‐layer error signal
    delta3 = -(y_true - y_hat) * y_hat * (1 - y_hat)

    # Gradients for the output‐weights
    dw5 = delta3 * h1
    dw6 = delta3 * h2

    # Hidden‐layer error signals
    delta1 = delta3 * w[4] * h1 * (1 - h1)
    delta2 = delta3 * w[5] * h2 * (1 - h2)

    # Gradients for the hidden‐weights
    dw1 = delta1 * x1
    dw2 = delta1 * x2
    dw3 = delta2 * x1
    dw4 = delta2 * x2
    
    # Update weights
    w[0] -= learning_rate * dw1
    w[1] -= learning_rate * dw2
    w[2] -= learning_rate * dw3
    w[3] -= learning_rate * dw4
    w[4] -= learning_rate * dw5
    w[5] -= learning_rate * dw6

# Build DataFrame of errors
rounds = list(range(1, len(errors) + 1))
df = pd.DataFrame({"Round": rounds, "Error": errors})

# Select first 10 rounds and the final round
indices = list(range(min(10, len(df)))) + [len(df)-1]
df_selected = df.iloc[indices].reset_index(drop=True)

print(df_selected)
