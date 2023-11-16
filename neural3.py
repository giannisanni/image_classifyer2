import numpy as np

np.random.seed(0)

X = [[1, 2, -4, 1.2],
     [-2, 3, 4, 1.8],
     [3, 2, -1, 3.2]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,3)
layer2 = Layer_Dense(3,5)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)