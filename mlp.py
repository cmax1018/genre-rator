#multi-layer perceptron
import numpy as np
from random import random

class MLP:
  def __init__(self, num_inputs=3, num_hidden=[3, 3], num_outputs=2):
    self.num_inputs = num_inputs
    self.num_hidden = num_hidden
    self.num_outputs = num_outputs

    layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

    #initiate random weights
    self.weights = []
    for i in range(len(layers)-1):
      w = np.random.rand(layers[i], layers[i+1])
      self.weights.append(w)

    activations = []
    #list of arrays where each array in the list represents activations for that layer
    for i in range(len(layers)):
      a = np.zeros(layers[i])
      activations.append(a)
    self.activations = activations
    derivatives = []
    #list of arrays where each array in the list represents activations for that layer
    for i in range(len(layers)-1):
      d = np.zeros((layers[i], layers[i+1]))
      derivatives.append(d)
    self.derivatives = derivatives

  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def forward_propagate(self, inputs):
    activations = inputs
    self.activations[0] = inputs
    for i, w in enumerate(self.weights):
      #calculate net input
      net_input = np.dot(activations, w)
      #calculate the activations
      activations = self._sigmoid(net_input)
      self.activations[i+1] = activations

    return activations

  def back_propagate(self, error, verbose=False):
    for i in reversed(range(len(self.derivatives))):
      activations = self.activations[i+1]
      delta = error * self._sigmoid_derivative(activations)
      #turn delta into approp. matrix
      delta_reshaped = delta.reshape(delta.shape[0], -1).T
      current_activations = self.activations[i]
      #need to turn current activations into a column vector for the math later
      current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
      self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
      error = np.dot(delta, self.weights[i].T)

      if verbose:
        print("Deritatives for W{}: {}".format(1, self.derivatives[i]))
    return error

  def gradient_descent(self, learning_rate):
    for i in range(len(self.weights)):
      weights = self.weights[i]
      derivatives = self.derivatives[i]
      weights += derivatives * learning_rate

  def train(self, inputs, targets, epochs, learning_rate):
    for i in range(epochs):
      sum_error = 0
      for input, target in zip(inputs, targets):
        output = self.forward_propagate(input)
        error = target - output
        self.back_propagate(error)
        self.gradient_descent(learning_rate)

        sum_error += self._mse(target, output)
      print("Training... error is {} at epoch {}".format(sum_error / len(inputs), i))

  def _mse(self, target, output):
    return np.average((target - output)**2)

  def _sigmoid_derivative(self, x):
    return x * (1.0 - x)

if __name__ == '__main__':
  # create an MLP
  mlp = MLP(2,[5],1)
  # create some inputs for training
  inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
  targets = np.array([[i[0] + i[1]] for i in inputs])


  mlp.train(inputs, targets, 200, 1)

  # create dummy data
  input = np.array([.2, .3])
  target = np.array([.5])
  output = mlp.forward_propagate(input)
  print()
  print()
  print("Albert: I believe {} + {} is {}".format(input[0], input[1], output[0]))
