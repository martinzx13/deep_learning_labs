import numpy as np

# Program to decrease the MSE function implementing
# a linear Regression model.

def compute_cost(X, y, theta):
  m = len(y)
  prediction = (X.dot(theta))
  errors = prediction - y

  cost_j0 = (1/(2 * m) * errors.T.dot(errors))

  return (cost_j0.item())

def gradient_descent(X, y , theta, alpha, iterations):
  m = len(y)
  cost_history = np.zeros(iterations)
  for i in range(iterations):
    prediction = (X.dot(theta))
    errors = prediction - y

    gradient = (1/m)*X.T.dot(errors)
    theta = theta - alpha * gradient
    cost_history[i] = compute_cost(X, y, theta)
  return (theta, cost_history)

