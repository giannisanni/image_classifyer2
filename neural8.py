import numpy as np
import pandas as pd

rg = np.random.default_rng()


def generate_data(n_features, n_values):
    features = rg.random((n_features, n_values))
    weights = rg.random((1, n_values))[0]
    targets = np.random.choice([0, 1], n_features)
    data = pd.DataFrame(features, columns=["x0", "x1", "x2"])
    data["targets"] = targets
    return data, weights


data, weights = generate_data(4, 3)
print(data)

bias = 0.5
l_rate = 0.1
epochs = 50
epoch_loss = []


def get_weighted_sum(feature, weights, bias):
    return np.dot(feature, weights) + bias


def sigmoid(w_sum):
    return 1 / (1 + np.exp(-w_sum))


def cross_entropy(target, prediction):
    return -(target * np.log10(prediction) + (1 - target) * np.log10(1 - prediction))


def update_weights(weights, l_rate, target, prediction, feature):
    new_weights = []
    for x, w in zip(feature, weights):
        new_w = w + l_rate * (target - prediction) * x
        new_weights.append(new_w)
    return new_weights


def update_bias(bias, l_rate, target, prediction):
    return bias + l_rate * (target - prediction)


def train_model(data, weights, bias, l_rate, epochs):
    for e in range(epochs):
        individual_loss = []
        for i in range(len(data)):
            feature = data.loc[i][:-1]
            target = data.loc[i][-1]
            w_sum = get_weighted_sum(feature, weights, bias)
            prediction = sigmoid(w_sum)
            loss = cross_entropy(target, prediction)
            individual_loss.append(loss)
            # gradient descent
            weights = update_weights(weights, l_rate, target, prediction, feature)
            bias = update_bias(bias, l_rate, target, prediction)
        average_loss = sum(individual_loss) / len(individual_loss)
        epoch_loss.append(average_loss)
        print("**************************")
        print("epoch", e)
        print(average_loss)


train_model(data, weights, bias, l_rate, epochs)


df = pd.DataFrame(epoch_loss)
df_plot = df.plot(kind="line", grid=True)
df_plot

