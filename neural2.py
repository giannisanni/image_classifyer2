import numpy as np

X = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]]

weights1 = [[4.1, 1.2, 6.3, 3.2],
            [3.1, 2.2, 4.3, 1.2],
            [5.1, 7.2, 1.3, 3.2]]

biases1 = [1.1, 2.2, 3.3]

weights2 = [[1.1, 2.2, 3.3, 4.4],
            [5.1, 6.2, 7.3, 8.4],
            [9.1, 10.2, 11.3, 12.4]]

biases2 = [1.4, 2.7, 6.3]
layer1_output = np.dot(X, np.array(weights1).T) + biases1
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

