from neuralnetwork import NeuralNetwork
from tensor import Tensor, TensorShape, TensorSpec
from algorithm.sort import sort

@value
struct Organism:
    var networks: List[NeuralNetwork]
    var one_forth: Int
    var two_forth: Int
    var three_forth: Int

    def __init__(inout self, network: NeuralNetwork, size: Int = 16):
        if size < 16:
            raise Error("size needs to be greater than 16")
        self.networks = List[NeuralNetwork]()
        for _ in range(size):
            var nn = network.clone()
            nn.randomize()
            self.networks.append(nn)
        one_forth = int(size * 0.25)
        self.one_forth = one_forth
        self.two_forth = one_forth * 2
        self.three_forth = one_forth * 3

    def evolve(self,
               inputs: List[List[Float64]],
               outputs: List[List[Float64]],
               generations: Int = 10000,
               error_threshold: Float64 = 0.0,
               log_each: Int = 1000) -> NeuralNetwork:

        fn cmp_err(a: NeuralNetwork, b: NeuralNetwork, /) capturing -> Bool:
            return a.error < b.error
        
        for gen in range(generations):
            for idx in range(self.networks.size):
                self.networks[idx].evaluate(inputs, outputs)

            sort[NeuralNetwork, cmp_err](self.networks)

            error = self.networks[0].error
            if error <= error_threshold:
                print("generation: " + String(gen) + " error: " + error + ". below threshold. breaking.")
                break
            elif gen % log_each == 0:
                print("generation: " + String(gen) + " error: " + error)

            self.networks = self.networks[:self.three_forth]
            top_quarter = self.networks[:self.one_forth]
            for n in range(top_quarter.size):
                self.networks.append(top_quarter[n].clone())

            for i in range(3):
                self.networks[i+1].punctuate(i)

            for j in range(11):
                self.networks[j+4].mutate()

        sort[NeuralNetwork, cmp_err](self.networks)
        
        return self.networks[0]
