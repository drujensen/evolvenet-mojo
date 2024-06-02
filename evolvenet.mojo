from neuralnetwork import NeuralNetwork
from organism import Organism
from tensor import Tensor, TensorShape, TensorSpec
from utils.index import Index

def main():
    var nn = NeuralNetwork()
    nn.add_layer("input", size=2)
    nn.add_layer("hidden", size=4)
    nn.add_layer("output", size=1)
    nn.fully_connect()

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

    var organism = Organism(nn)
    var network = organism.evolve(inputs, outputs, generations=10000)

    network.inspect()

    var error = String(network.error)
    print("Final error: " + error)
    confusion_matrix(network, inputs, outputs)


def confusion_matrix(model: NeuralNetwork, inputs: List[List[Float64]], outputs: List[List[Float64]]):
    t_n = t_p = f_n = f_p = ct = 0

    var rows = len(inputs)
    for row in range(rows):
        var actuals = model.run(inputs[row])
        var cols = len(actuals[row])
        for col in range(cols):
            ct += 1
            if outputs[row][col] > 0.5:
                if actuals[row][col] > 0.5:
                    t_p += 1
                else:
                    f_p += 1
            else:
                if actuals[row][col] < 0.5:
                    t_n += 1
                else:
                    f_n += 1

    print("Test size: " + String(rows))
    print("----------------------")
    print("TN: " + String(t_n) + " | " + " FP: " + String(f_p))
    print("----------------------")
    print("FN: " + String(f_n) + " | " + " TP: " + String(t_p))
    print("----------------------")
    accuracy = (t_n + t_p) / ct
    print("Accuracy: " + String(accuracy))


