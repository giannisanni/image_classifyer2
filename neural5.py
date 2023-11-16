import numpy as np
from PIL import Image

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network architecture
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, hidden_layers=2):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers

        # Initialize weights and biases with smaller values (Xavier/Glorot Initialization)
        self.weights = [np.random.randn(input_nodes, hidden_nodes) / np.sqrt(input_nodes)] + \
                        [np.random.randn(hidden_nodes, hidden_nodes) / np.sqrt(hidden_nodes) for _ in range(hidden_layers - 1)] + \
                        [np.random.randn(hidden_nodes, output_nodes) / np.sqrt(hidden_nodes)]
        self.biases = [np.random.randn(hidden_nodes) / np.sqrt(hidden_nodes) for _ in range(hidden_layers)] + \
                        [np.random.randn(output_nodes) / np.sqrt(output_nodes)]

    # Forward pass - Added safe_sigmoid function to keep values under a threshold and avoid OverflowError
    def feedforward(self, inputs):

        def safe_sigmoid(x):
            return sigmoid(np.clip(x, -100, 100))

        self.layers = [inputs]
        for i in range(len(self.weights)):
            self.layers.append(safe_sigmoid(np.dot(self.layers[-1], self.weights[i]) + self.biases[i]))
        return self.layers[-1]

    # Backward pass
    def backpropagation(self, inputs, target_outputs, learning_rate=0.5):
        deltas = [None] * (self.hidden_layers + 1)

        # Calculate output error
        output_error = target_outputs - self.layers[-1]
        deltas[-1] = output_error * sigmoid_derivative(self.layers[-1])

        # Calculate hidden error
        for i in range(len(deltas) - 2, -1, -1):
            hidden_error = np.dot(deltas[i+1], self.weights[i+1].T)
            deltas[i] = hidden_error * sigmoid_derivative(self.layers[i+1])

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * np.dot(self.layers[i].T, deltas[i])
            self.biases[i] += learning_rate * np.sum(deltas[i], axis=0)

    # Train the neural network
    def train(self, inputs, target_outputs, epochs, learning_rate):
        for epoch in range(epochs):
            predicted_outputs = self.feedforward(inputs)
            self.backpropagation(inputs, target_outputs, learning_rate)

    # Evaluate the neural network performance
    def evaluate(self, test_inputs, test_outputs):
        test_predictions = self.feedforward(test_inputs)
        predicted_classes = np.argmax(test_predictions, axis=1)
        true_classes = np.argmax(test_outputs, axis=1)
        accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
        return accuracy

# Create a function to convert images to input data accepted by the network
def process_image(img_path, flatten=True):
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print(f"The image file '{img_path}' was not found. Please check the file path and try again.")
        return None

    img = img.resize((32, 32))
    img_array = np.array(img, dtype=np.float32) / 255
    if flatten:
        img_array = img_array.flatten()
    return img_array

if __name__ == "__main__":
    neural_network = NeuralNetwork(3072, 150, 4, hidden_layers=3)

    # Define training data
    image_paths = ['cat1.jpg', 'cat2.jpg', 'cat3.jpg', 'cat4.jpg','cat5.jpg',
               'dog1.jpg', 'dog2.jpg', 'dog3.jpg', 'dog4.jpg','dog5.jpg',
               'car1.jpg', 'car2.jpg', 'car3.jpg', 'car4.jpg','car5.jpg',
               'house1.jpg', 'house2.jpg', 'house3.jpg', 'house4.jpg','house5.jpg']

    inputs = [process_image(image_path) for image_path in image_paths]
    inputs = np.array([input for input in inputs if input is not None])

    if not inputs.any():
        print("No valid images found for training")
    else:
        # Define target outputs
        target_outputs = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                                   [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],
                                   [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])

        # Train the neural network
        neural_network.train(inputs, target_outputs, epochs=2500, learning_rate=0.5)

        # Test the neural network with more images and evaluate its performance
        test_image_paths = ['car4.jpg']

        test_inputs = [process_image(test_image_path) for test_image_path in test_image_paths]
        test_inputs = np.array([test_input for test_input in test_inputs if test_input is not None])
        if not test_inputs.any():
            print("No valid test images found")
        else:
            test_outputs = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
            test_predictions = neural_network.feedforward(test_inputs)
            print("Test image predictions: ", test_predictions)

            # Evaluate the neural network performance
            accuracy = neural_network.evaluate(test_inputs, test_outputs)
            print(f"Neural network accuracy: {accuracy:.2%}")

        def classify_image(predictions):
            labels = ['cat', 'dog', 'car', 'house']

            max_index = predictions.argmax()

            return labels[max_index]

        # Example usage
        predictions = test_predictions
        for i in range(len(test_image_paths)):
            label = classify_image(predictions[i])
            print(f'The image {test_image_paths[i]} looks most like a {label}.')