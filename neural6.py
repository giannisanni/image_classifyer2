import numpy as np
from PIL import Image
import streamlit as st
import joblib
import os

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
    # Check if a pre-trained model exists
    model_path = "neural_network_model.joblib"
    if os.path.exists(model_path):
        # Load the pre-trained model
        neural_network = joblib.load(model_path)
    else:
        # Create a neural network with increased input nodes, hidden nodes, and a higher number of output nodes for handling more classes
        neural_network = NeuralNetwork(3072, 150, 2, hidden_layers=3)  # Updated output_nodes to 4

    # Define training data and add more image samples
    cat_paths = ['cat1.jpg', 'cat2.jpg', 'cat3.jpg', 'cat4.jpg', 'cat5.jpg', 'cat6.jpg', 'cat7.jpg', 'cat8.jpg', 'cat9.jpg', 'cat10.jpg']
    dog_paths = ['dog1.jpg', 'dog2.jpg', 'dog3.jpg', 'dog4.jpg', 'dog5.jpg', 'dog6.jpg', 'dog7.jpg', 'dog8.jpg', 'dog9.jpg', 'dog10.jpg']

    cat_inputs = [process_image(cat_path) for cat_path in cat_paths]
    dog_inputs = [process_image(dog_path) for dog_path in dog_paths]

    inputs = np.concatenate([cat_inputs, dog_inputs], axis=0)

    if not inputs.any():
        print("No valid images found for training")
    else:
        # Define target outputs with an additional class
        cat_targets = np.array([[1, 0] for _ in cat_inputs])
        dog_targets = np.array([[0, 1] for _ in dog_inputs])

        target_outputs = np.concatenate([cat_targets, dog_targets], axis=0)

        # Train the neural network
        neural_network.train(inputs, target_outputs, epochs=2000, learning_rate=0.02)
        # Save the trained model
        joblib.dump(neural_network, model_path)
        # Test the neural network with more images and evaluate its performance
        test_image_paths = ['dog14.jpg']
        test_inputs = [process_image(test_image_path) for test_image_path in test_image_paths]
        test_inputs = np.array([test_input for test_input in test_inputs if test_input is not None])
        if not test_inputs.any():
            print("No valid test images found")
        else:
            test_outputs = np.array([[1, 0]])
            test_predictions = neural_network.feedforward(test_inputs)
            print("Test image predictions: ", test_predictions)

            # Evaluate the neural network performance
            accuracy = neural_network.evaluate(test_inputs, test_outputs)
            print(f"Neural network accuracy: {accuracy:.2%}")


        def classify_image(predictions):
            # Create a list of labels
            labels = ['cat', 'dog']

            # Find the index of the highest prediction
            max_index = predictions.argmax()

            # Return the label with the corresponding index
            return labels[max_index]


        # Example usage
        predictions = test_predictions
        label = classify_image(predictions[0])
        print(f'This image looks most like a {label}.')

def main():
    st.title("Image Classification with Neural Network")

    uploaded_file = st.file_uploader("Choose an image of a dog ore a cat...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Process the uploaded image
        img_array = process_image(uploaded_file, flatten=True)
        if img_array is not None:
            img_array = img_array.reshape(1, -1)  # Reshape for the neural network input
            prediction = neural_network.feedforward(img_array)

            # Display prediction result
            st.write(f"Prediction Result: {classify_image(prediction[0])}")
            # Show test image predictions
            st.write("Test Image Predictions:")
            st.write(prediction)

if __name__ == "__main__":
    main()
