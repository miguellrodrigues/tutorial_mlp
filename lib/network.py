import numpy as np

from lib.layer import Layer
from lib.matrix import Matrix, array_to_matrix, matrix_to_array
from lib.numbers import random_int, random_double


class Network:
  def __init__(self, topology):
    self.topology = topology
    self.topology_size = len(topology)
    
    self.layers = []
    self.weight_matrices = []
    
    self.bias = .01
    
    self.errors = []
    self.derived_errors = []
    
    self.global_error = .0
    
    self.learning_rate = 1e-3
    
    self.output_index = self.topology_size - 1
    
    for i in range(self.topology_size):
      self.layers.append(Layer(topology[i]))
    
    for i in range(self.topology_size - 1):
      weight_matrix = Matrix(topology[i+1], topology[i], True)
      
      self.weight_matrices.append(weight_matrix)
    
    for i in range(topology[self.topology_size - 1]):
      self.errors.append(.0)
      self.derived_errors.append(.0)
  
  def set_current_input(self, _in):
    for i in range(_in.rows):
      self.layers[0].set_neuron_value(i, _in.get_value(i, 0))
  
  def feed_forward(self):
    for i in range(self.topology_size - 1):
      if i == 0:
        left = self.layers[i].convert_to_matrix(Layer.VALUES)
      else:
        left = self.layers[i].convert_to_matrix(Layer.ACTIVATED_VALUES)
      
      right = self.weight_matrices[i]
      
      r = right.multiply(left)
      
      for j in range(r.rows):
        self.layers[i + 1].set_neuron_value(j, (r.get_value(j, 0) + self.bias))
  
  def back_propagation(self):
    weights = []
  
    gradients = Matrix(self.topology[self.output_index], 1)
  
    derived_output_values = self.layers[self.output_index].convert_to_matrix(Layer.DERIVED_VALUES)
  
    for i in range(self.topology[self.output_index]):
      error = self.derived_errors[i]
      output = derived_output_values.get_value(i, 0)
    
      gradient = error * output
    
      gradients.set_value(i, 0, gradient)
  
    last_hidden_layer_activated = self.layers[self.output_index - 1].convert_to_matrix(Layer.ACTIVATED_VALUES)
  
    delta_weights_last_hidden = gradients.multiply(last_hidden_layer_activated.transpose())
  
    temp_weights = Matrix(
      self.topology[self.output_index],
      self.topology[self.output_index - 1]
    )
  
    for i in range(temp_weights.rows):
      for j in range(temp_weights.cols):
        original_value = self.weight_matrices[self.output_index - 1].get_value(i, j)
        delta_value = delta_weights_last_hidden.get_value(i, j)
      
        delta_value *= self.learning_rate
      
        temp_weights.set_value(i, j, (original_value - delta_value))
  
    weights.append(temp_weights)
  
    i = self.output_index - 1
  
    # # # # # # # #
  
    while i > 0:
      _gradients = Matrix(gradients.rows, gradients.cols)
    
      for j in range(gradients.rows):
        for k in range(gradients.cols):
          _gradients.set_value(j, k, gradients.get_value(j, k))
    
      transposed_weights = self.weight_matrices[i].transpose()
    
      gradients = transposed_weights.multiply(_gradients)
    
      # # # # #
    
      derived_values = self.layers[i].convert_to_matrix(Layer.DERIVED_VALUES)
    
      layer_gradients = derived_values.hadamard(gradients)
    
      for j in range(layer_gradients.rows):
        for k in range(layer_gradients.cols):
          gradients.set_value(j, k, layer_gradients.get_value(j, k))
    
      if i == 1:
        layer_values = self.layers[0].convert_to_matrix(Layer.VALUES)
      else:
        layer_values = self.layers[i - 1].convert_to_matrix(Layer.ACTIVATED_VALUES)
    
      delta_weights = gradients.multiply(layer_values.transpose())
    
      _temp_weights = Matrix(
        self.weight_matrices[i - 1].rows,
        self.weight_matrices[i - 1].cols
      )
    
      for j in range(_temp_weights.rows):
        for k in range(_temp_weights.cols):
          original_value = self.weight_matrices[i - 1].get_value(j, k)
          delta_value = delta_weights.get_value(j, k)
        
          delta_value *= self.learning_rate
        
          _temp_weights.set_value(j, k, (original_value - delta_value))
    
      weights.append(_temp_weights)
    
      i -= 1
  
    self.weight_matrices = []
  
    for matrix in reversed(weights):
      self.weight_matrices.append(matrix)

  def test(self):
    weights = []
  
    gradients = Matrix(self.topology[self.output_index], 1)
  
    derived_output_values = self.layers[self.output_index].convert_to_matrix(Layer.DERIVED_VALUES)

    gradients.data = derived_output_values.data * np.array(self.derived_errors)
  
    last_hidden_layer_activated = self.layers[self.output_index - 1].convert_to_matrix(Layer.ACTIVATED_VALUES)
    delta_weights_last_hidden = gradients.multiply(last_hidden_layer_activated.transpose()).scalar(self.learning_rate)
  
    w = self.weight_matrices[self.output_index - 1].subtract(delta_weights_last_hidden)

    weights.append(w)
  
    i = self.output_index - 1
  
    # # # # # # # #
  
    while i > 0:
      _gradients = gradients.copy()
    
      transposed_weights = self.weight_matrices[i].transpose()
    
      gradients = transposed_weights.multiply(_gradients)
    
      # # # # #
    
      derived_values = self.layers[i].convert_to_matrix(Layer.DERIVED_VALUES)
    
      layer_gradients = derived_values.hadamard(gradients)
      gradients.data = layer_gradients.data
    
      if i == 1:
        layer_values = self.layers[0].convert_to_matrix(Layer.VALUES)
      else:
        layer_values = self.layers[i - 1].convert_to_matrix(Layer.ACTIVATED_VALUES)
    
      delta_weights = gradients.multiply(layer_values.transpose()).scalar(self.learning_rate)
    
      w = self.weight_matrices[i - 1].subtract(delta_weights)

      weights.append(w)
    
      i -= 1
  
    self.weight_matrices = []
  
    for weight in reversed(weights):
      self.weight_matrices.append(weight)
  
  def set_errors(self, target):
    if target.rows == 0:
      print('Invalid Target')
      return
    
    if target.rows != self.layers[self.output_index].neurons_size:
      print('Invalid Target')
      return
    
    self.global_error = .0
    
    output_neurons = self.layers[self.output_index].neurons
    for i in range(target.rows):
      t = target.get_value(i, 0)
      y = output_neurons[i].get_derived_value()
      
      self.errors[i] = .5 * np.power((t - y), 2.0)
      self.derived_errors[i] = (y - t)
      
      self.global_error += self.errors[i]
  
  def train(self, _in, target):
    self.set_current_input(array_to_matrix(_in))
    
    self.feed_forward()
    self.set_errors(array_to_matrix(target))
    self.test()

  def predict(self, _in):
    self.set_current_input(array_to_matrix(_in))
    self.feed_forward()
    
    return matrix_to_array(self.layers[self.output_index].convert_to_matrix(Layer.VALUES))
  
  def mutate(self, rate):
    for weight_matrix in self.weight_matrices:
      count = int(rate * weight_matrix.cols)
      
      random_row = random_int(0, weight_matrix.rows - 1)
      
      for _ in range(count):
        random_col = random_int(0, weight_matrix.cols - 1)
        
        value = weight_matrix.get_value(random_row, random_col)
        
        weight_matrix.set_value(random_row, random_col, value + random_double(-1, 1))
