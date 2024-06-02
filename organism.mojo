from neuralnetwork import NeuralNetwork
from tensor import Tensor, TensorShape, TensorSpec
from algorithm.sort import sort

@value
struct Organism:
    var networks: List[NeuralNetwork]
    var size: Int
    var one_forth: Int
    var two_forth: Int
    var three_forth: Int

    def __init__(inout self, network: NeuralNetwork, size: Int = 16):
        if size < 16:
            raise Error("size needs to be greater than 16")

        self.networks = List[NeuralNetwork]()

        for idx in range(size):
            self.networks.append(network.clone())

        for idx in range(size):
            self.networks[idx].randomize()

        self.size = size
        self.one_forth = int(size * 0.25)
        self.two_forth = self.one_forth * 2
        self.three_forth = self.one_forth * 3

    def evolve(self,
               inputs: List[List[Float64]],
               outputs: List[List[Float64]],
               generations: Int = 10000,
               error_threshold: Float64 = 0.0,
               log_each: Int = 1000) -> NeuralNetwork:

        fn cmp_err(a: NeuralNetwork, b: NeuralNetwork, /) capturing -> Bool:
            return a.error < b.error

        for gen in range(generations):
            for idx in range(len(self.networks)):
                self.networks[idx].evaluate(inputs, outputs)

            sort[NeuralNetwork, cmp_err](self.networks)

            error = self.networks[0].error
            if error <= error_threshold:
                print("generation: " + String(gen) + " error: " + error + ". below threshold. breaking.")
                break
            elif gen % log_each == 0:
                print("generation: " + String(gen) + " error: " + error)

            # kill the bottom quarter
            self.networks.resize(self.three_forth)

            # clone the top quarter
            for n in range(self.one_forth):
                self.networks.append(self.networks[n].clone())

            # mutate all but the best one
            for n in range(1, self.size):
                self.networks[n].mutate()

            # punctuate top quarter
            for n in range(1, self.one_forth):
                self.networks[n].punctuate(n)

        sort[NeuralNetwork, cmp_err](self.networks)

        return self.networks[0]
