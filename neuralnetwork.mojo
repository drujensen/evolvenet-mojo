import math
from random import random_float64
from math import round, min, max, pow
from tensor import Tensor, TensorShape, TensorSpec, rand

@value
struct Neuron:
    var synapses: List[Float64]
    var function: String
    var activation: Float64
    var bias: Float64

    fn __init__(inout self, function: String):
        self.function = function
        self.activation = 0.0
        self.bias = 0.0
        self.synapses = List[Float64]()

    fn clone(self) -> Neuron:
        var neuron: Neuron = Neuron(self.function)
        neuron.activation.__copyinit__(self.activation)
        neuron.bias.__copyinit__(self.bias)
        neuron.synapses.__copyinit__(self.synapses)
        return neuron

    fn randomize(inout self):
        self.bias = random_float64(-1.0, 1.0)
        var size = len(self.synapses)
        for idx in range(size):
          self.synapses[idx] = random_float64(-1.0, 1.0)

    fn mutate(inout self, rate: Float64):
        self.bias += random_float64(-rate, rate)
        var size = len(self.synapses)
        for idx in range(size):
            self.synapses[idx] = random_float64(-rate, rate)

    fn punctuate(inout self, pos: Int):
        var pow10 = pow[DType.float64, 1](10, pos)
        self.bias = round(pow10 * self.bias) / pow10
        var size = len(self.synapses)
        for idx in range(size):
            self.synapses[idx] = round(pow10 * self.synapses[idx]) / pow10

    fn set(inout self, value: Float64):
        self.activation = self.none(value)

    fn activate(inout self, parent: Layer):
        var total = 0.0
        var size = len(self.synapses)
        for idx in range(size):
            total += self.synapses[idx] * parent.neurons[idx].activation

        if self.function == "none":
            self.activation = self.none(total + self.bias)
        elif self.function == "relu":
            self.activation = self.relu(total + self.bias)
        elif self.function == "sigmoid":
            self.activation = self.sigmoid(total + self.bias)
        elif self.function == "tanh":
            self.activation = self.tanh(total + self.bias)
        else:
            print("Activation function " + self.function + " is not supported")

    fn none(self, value: Float64) -> Float64:
        return value

    fn relu(self, value: Float64) -> Float64:
        return max(0.0, value)

    fn sigmoid(self, value: Float64) -> Float64:
        return 1.0 / (1.0 + math.exp(-value))

    fn tanh(self, value: Float64) -> Float64:
        return math.tanh(value)


@value
struct Layer:
    var name: String
    var function: String
    var neurons: List[Neuron]

    fn __init__(inout self, name: String, size: Int = 0, function: String = "sigmoid"):
        self.name = name
        self.function = function
        self.neurons = List[Neuron]()
        for _ in range(size):
            self.neurons.append(Neuron(function))

    fn clone(inout self) -> Layer:
        var layer = Layer(self.name, 0, self.function)
        layer.neurons.__copyinit__(self.neurons)
        return layer

    fn randomize(inout self):
        for idx in range(self.neurons.size):
            self.neurons[idx].randomize()

    fn mutate(self, rate: Float64):
        var neuron_rate = Float64(rate / self.neurons.size)
        for idx in range(self.neurons.size):
            var neuron = self.neurons[idx]
            neuron.mutate(neuron_rate)

    fn punctuate(inout self, pos: Int):
        for idx in range(self.neurons.size):
            self.neurons[idx].punctuate(pos)

    fn set(inout self, values: List[Float64]):
        for idx in range(self.neurons.size):
            self.neurons[idx].set(Float64(values[idx]))

    fn activate(inout self, parent: Layer):
        for idx in range(self.neurons.size):
            self.neurons[idx].activate(parent)


@value
struct NeuralNetwork:
    var layers: List[Layer]
    var error: Float64

    fn __init__(inout self):
        self.layers = List[Layer]()
        self.error = 1.0

    fn add_layer(inout self, name: String, size: Int, function: String = "sigmoid"):
        self.layers.append(Layer(name=name, size=size, function=function))

    fn clone(self) -> NeuralNetwork:
        var network: NeuralNetwork = NeuralNetwork()
        network.layers.__copyinit__(self.layers)
        return network

    fn randomize(inout self):
        self.error = 1.0
        for idx in range(self.layers.size):
            if idx == 0:
                continue
            self.layers[idx].randomize()

    fn mutate(inout self):
        for idx in range(self.layers.size):
            if idx == 0:
                continue
            self.layers[idx].mutate(self.error)

    fn punctuate(inout self, pos: Int):
        for idx in range(self.layers.size):
            if idx == 0:
                continue
            self.layers[idx].punctuate(idx)

    fn run(inout self, data: List[Float64]) -> List[Float64]:
        for idx in range(self.layers.size):
            if idx == 0:
                self.layers[idx].set(data[idx])
            else:
                self.layers[idx].activate(self.layers[idx - 1])
        var size = len(self.layers[-1].neurons)
        var results = List[Float64](size)
        for n_idx in range(size):
            results.append(self.layers[-1].neurons[n_idx].activation)

        return results

    fn evaluate(inout self, inputs: List[List[Float64]], outputs: List[List[Float64]]):
        var sum_error: Float64 = 0.0
        var rows = len(inputs)
        for row in range(rows):
            var actuals = self.run(inputs[row])
            var n_size = len(self.layers[-1].neurons)
            for n_idx in range(n_size):
                sum_error += (self.layers[-1].neurons[n_idx].activation - outputs[row][n_idx]) ** 2
        self.error = sum_error / (2 * rows)

    fn fully_connect(inout self):
        for l_idx in range(self.layers.size):
            if l_idx == 0:
                continue
            var parent = self.layers[l_idx - 1]
            var size = len(self.layers[l_idx].neurons)
            for n_idx in range(size):
                self.layers[l_idx].neurons[n_idx].synapses = List[Float64](len(parent.neurons))
