from neuralnetwork import NeuralNetwork
from organism import Organism
from tensor import Tensor, TensorShape, TensorSpec

def confusion_matrix(model: NeuralNetwork, data: Tensor[DType.float64]):
    t_n = t_p = f_n = f_p = ct = 0

    var size = data.dim(0)
    for row in range(size):
        var inputs = Tensor(data[row][0])
        var outputs = Tensor(data[row][1])
        var actuals = model.run(inputs)
        for idx in range(actuals.dim(0)):
            ct += 1
            if outputs[idx] > 0.5:
                if actuals[idx] > 0.5:
                    t_p += 1
                else:
                    f_p += 1
            else:
                if actuals[idx] < 0.5:
                    t_n += 1
                else:
                    f_n += 1

    print("Test size: " + String(size))
    print("----------------------")
    print("TN: " + String(t_n) + " | " + " FP: " + String(f_p))
    print("----------------------")
    print("FN: " + String(f_n) + " | " + " TP: " + String(t_p))
    print("----------------------")
    accuracy = (t_n + t_p) / ct
    print("Accuracy: " + String(accuracy))


def main():
    var nn = NeuralNetwork()
    nn.add_layer("input", size=2)
    nn.add_layer("hidden", size=2)
    nn.add_layer("output", size=1)
    nn.fully_connect()

    var data = Tensor[DType.float64]([
        [[0.0, 0.0], [0.0]],
        [[0.0, 1.0], [1.0]],
        [[1.0, 0.0], [1.0]],
        [[1.0, 1.0], [0.0]]
    ])

    var organism = Organism(nn)
    var network = organism.evolve(data)
    
    var error = String(network.error)
    print("Final error: " + error)
    confusion_matrix(network, data)
