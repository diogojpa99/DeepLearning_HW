import numpy as np

# Define the number of input units, hidden units, and output units
input_units = 784
hidden_units = 200
output_units = 10

# Initialize the weight matrices and bias vectors for the hidden and output layers
W1 = np.random.normal(0.1, 0.1**2, (input_units, hidden_units))
b1 = np.zeros(hidden_units)
W2 = np.random.normal(0.1, 0.1**2, (hidden_units, output_units))
b2 = np.zeros(output_units)

# Define the ReLU activation function
def relu(x):
  return np.maximum(0, x)

# Define the multinomial logistic loss function
def softmax(x):
  exps = np.exp(x - np.max(x))
  return exps / np.sum(exps)

# Define the forward propagation function
def forward_prop(x, W1, b1, W2, b2):
  # Compute the output of the hidden layer
  hidden = relu(np.dot(x, W1) + b1)
  # Compute the output of the output layer
  output = softmax(np.dot(hidden, W2) + b2)
  return output

def gradient_backprop(x, y, W1, b1, W2, b2):
  # Compute the output of the hidden layer
  hidden = relu(np.dot(x, W1) + b1)
  # Compute the output of the output layer
  output = softmax(np.dot(hidden, W2) + b2)

  # Compute the error between the model output and the ground truth labels
  error = output - y

  # Compute the gradients of the model parameters with respect to the loss
  dW2 = np.dot(hidden.T, error)
  db2 = np.sum(error, axis=0)
  dhidden = np.dot(error, W2.T)
  dhidden[hidden <= 0] = 0
  dW1 = np.dot(x.T, dhidden)
  db1 = np.sum(dhidden, axis=0)

  return dW1, db1, dW2, db2

# Set the learning rate
learning_rate = 0.001

# Iterate over the training data
for x, y in training_data:
  # Compute the gradients of the model parameters
  dW1, db1, dW2, db2 = gradient_backprop(x, y, W1, b1, W2, b2)

  # Update the model parameters
  W1 -= learning_rate * dW1
  b1 -= learning_rate * db1
  W2 -= learning_rate * dW2
  b2 -= learning_rate * db2
  
  
  ###################################################################
  
import numpy as np

# define the sigmoid activation function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# define the derivative of the sigmoid activation function
def sigmoid_derivative(x):
  return x * (1 - x)

# define the MLP class
class MLP:
  def __init__(self, input_size, hidden_size, output_size):
    # initialize the weights and biases of the MLP
    self.weights1 = np.random.rand(input_size, hidden_size)
    self.bias1 = np.random.rand(1, hidden_size)
    self.weights2 = np.random.rand(hidden_size, output_size)
    self.bias2 = np.random.rand(1, output_size)

  def forward(self, x):
    # perform a forward pass through the MLP
    self.hidden = sigmoid(np.dot(x, self.weights1) + self.bias1)
    self.output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)

  def backward(self, x, y, learning_rate):
    # compute the error at the output layer
    output_error = y - self.output

    # compute the gradient of the error with respect to the output
    output_gradient = sigmoid_derivative(self.output) * output_error

    # compute the error at the hidden layer
    hidden_error = np.dot(output_gradient, self.weights2.T)

    # compute the gradient of the error with respect to the hidden layer
    hidden_gradient = sigmoid_derivative(self.hidden) * hidden_error

    # update the weights and biases
    self.weights2 += np.dot(self.hidden.T, output_gradient) * learning_rate
    self.bias2 += np.sum(output_gradient, axis=0) * learning_rate
    self.weights1 += np.dot(x.T, hidden_gradient) * learning_rate
    self.bias1 += np.sum(hidden_gradient, axis=0) * learning_rate

# define the training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# create an instance of the MLP
mlp = MLP(2, 2, 1)

# set the learning rate
learning_rate = 0.1

# train the MLP using SGD
for epoch in range(10000):
  for x, y in zip(X, Y):
    mlp.forward(x)
    mlp.backward(x, y, learning_rate)

# test the MLP
print(mlp.output)

