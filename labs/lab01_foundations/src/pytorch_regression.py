import torch

def gradient_descent_pytorch(X_b_tensor, y_tensor, theta, alpha, iterations):
  m = len(y_tensor)
  costs = torch.zeros(iterations)

  for i in range(iterations):
    prediction = X_b_tensor@theta
    error = prediction - y_tensor
    loss = (1/(2 * m))*torch.sum(error**2)

    loss.backward()

    with torch.no_grad():
      # We substract the gradient
      theta -= alpha * theta.grad

      # Clean the slope gradient.
      theta.grad.zero_()

    # Store the current value
    costs[i] = loss.item()
  return (theta, costs)