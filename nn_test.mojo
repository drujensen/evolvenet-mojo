from neuralnetwork import NeuralNetwork, Neuron, Layer
from organism import Organism
from tensor import Tensor, TensorShape, TensorSpec, rand

def main():
  var layer = Layer("input", 1, "none")
  #var layer2 = Layer("hidden", 1, "none")
  
  var spec = TensorSpec(DType.float64, 1)
  var tensor = Tensor[DType.float64](spec)
  tensor[0] = 4.0
  layer.set(tensor)

  print(layer.neurons[0].synapses[0])
  print(layer.neurons[0].bias)
  print(layer.neurons[0].activation)

  var layer2 = Layer("hidden", 1, "relu")
  layer2.neurons[0].synapses = rand[DType.float64](spec)
  layer2.activate(layer)

  print(layer2.neurons[0].synapses[0])
  print(layer2.neurons[0].bias)
  print(layer2.neurons[0].activation)

