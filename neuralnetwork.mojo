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
        neuron.activation = self.activation
        neuron.bias = self.bias
        for synapse in self.synapses:
            neuron.synapses.append(synapse[])
        return neuron

    fn randomize(inout self):
        self.bias = random_float64(-1.0, 1.0)
        for synapse in self.synapses:
            synapse[] = random_float64(-1.0, 1.0)

    fn mutate(inout self, rate: Float64):
        self.bias += random_float64(-rate, rate)
        for synapse in self.synapses:
            synapse[] += random_float64(-rate, rate)

    fn punctuate(inout self, pos: Int):
        var pow10 = pow[DType.float64, 1](10, pos)
        self.bias = round(pow10 * self.bias) / pow10
        for synapse in self.synapses:
            synapse[] = round(pow10 * synapse[]) / pow10

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

    fn clone(self) -> Layer:
        var layer = Layer(self.name, 0, self.function)
        for neuron in self.neurons:
          layer.neurons.append(neuron[].clone())
        return layer

    fn randomize(inout self):
        for neuron in self.neurons:
            neuron[].randomize()

    fn mutate(inout self, rate: Float64):
        var neuron_rate = Float64(rate / len(self.neurons))
        for neuron in self.neurons:
            neuron[].mutate(neuron_rate)

    fn punctuate(inout self, pos: Int):
        for neuron in self.neurons:
            neuron[].punctuate(pos)

    fn set(inout self, values: List[Float64]):
        for idx in range(len(values)):
            self.neurons[idx].set(values[idx])

    fn connect(inout self, parent: Layer):
        for neuron in self.neurons:
          for _ in range(len(parent.neurons)):
            neuron[].synapses.append(random_float64(-1.0, 1.0))

    fn activate(inout self, parent: Layer):
        for neuron in self.neurons:
            neuron[].activate(parent)

    fn inspect(self):
        for idx in range(len(self.neurons)):
            print("n:", idx)
            print(" - w:")
            for s_idx in range(len(self.neurons[idx].synapses)):
                print("   -", self.neurons[idx].synapses[s_idx])
            print(" - f:", self.neurons[idx].function)
            print(" - b:", self.neurons[idx].bias)
            print(" - a:", self.neurons[idx].activation)


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
        for layer in self.layers:
            network.layers.append(layer[].clone())
        return network

    fn randomize(inout self):
        self.error = 1.0
        for idx in range(len(self.layers)):
            if idx == 0:
                continue
            self.layers[idx].randomize()

    fn mutate(inout self):
        for idx in range(len(self.layers)):
            if idx == 0:
                continue
            self.layers[idx].mutate(self.error)

    fn punctuate(inout self, pos: Int):
        for idx in range(len(self.layers)):
            if idx == 0:
                continue
            self.layers[idx].punctuate(idx)

    fn run(inout self, data: List[Float64]) -> List[Float64]:
        for idx in range(len(self.layers)):
            if idx == 0:
                self.layers[idx].set(data)
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
            var cols = len(actuals)
            for col in range(cols):
                sum_error += (actuals[col] - outputs[row][col]) ** 2
        self.error = sum_error / (2 * rows)

    fn fully_connect(inout self):
        for idx in range(len(self.layers)):
            if idx == 0:
                continue
            var parent = self.layers[idx - 1]
            self.layers[idx].connect(parent)

    fn inspect(self):
        print("#############################################")
        for idx in range(len(self.layers)):
            print("---------")
            print("l:", idx, self.layers[idx].name)
            print("---------")
            self.layers[idx].inspect()
        print("---------")
        print("error: ", self.error)
