from neuralnetwork import NeuralNetwork, Neuron, Layer
from organism import Organism
from tensor import Tensor, TensorShape, TensorSpec, rand
from algorithm.sort import sort

def main():
    var nn = NeuralNetwork()
    nn.add_layer("input", size=2)
    nn.add_layer("hidden", size=2)
    nn.add_layer("output", size=1)
    nn.fully_connect()

    var networks = List[NeuralNetwork]()
    networks.append(nn.clone())
    networks.append(nn.clone())
    networks.append(nn.clone())
    networks.append(nn.clone())

    var inputs = List[List[Float64]]()
    inputs.append(List[Float64](0.0, 0.0))
    inputs.append(List[Float64](0.0, 1.0))
    inputs.append(List[Float64](1.0, 0.0))
    inputs.append(List[Float64](1.0, 1.0))

    var outputs = List[List[Float64]]()
    outputs.append(List[Float64](0.0))
    outputs.append(List[Float64](1.0))
    outputs.append(List[Float64](1.0))
    outputs.append(List[Float64](0.0))

    for idx in range(len(networks)):
        networks[idx].randomize()

    for idx in range(len(networks)):
        networks[idx].mutate()

    for idx in range(len(networks)):
        networks[idx].evaluate(inputs, outputs)

    fn cmp_err(a: NeuralNetwork, b: NeuralNetwork, /) capturing -> Bool:
        return a.error < b.error

    sort[NeuralNetwork, cmp_err](networks)

    for idx in range(len(networks)):
        networks[idx].inspect()
