import numpy as np
import matplotlib.pyplot as plt

def init_params(layers_dims, initialization="default"):
    """
    Initializes the parameters (weights and biases) for a neural network.

    Parameters:
    layers_dims (list): List of integers representing the dimensions of each layer in the neural network.
    initialization (str): Initialization method to use for parameter initialization.
        - "default": Random initialization using a uniform distribution between -0.5 and 0.5.
        - "xavier": Xavier/Glorot initialization for better training convergence.
        - "he": He initialization for training deeper networks.

    Returns:
    params (dict): A dictionary containing the initialized parameters for each layer.
    """
    params = {}

    for layer in range(1, len(layers_dims)):
        input_dim = layers_dims[layer - 1]
        output_dim = layers_dims[layer]

        # Xavier weight initialization
        if initialization == "xavier":
            params["W" + str(layer)] = np.random.uniform(
                -np.sqrt(6 / (input_dim + output_dim)),
                np.sqrt(6 / (input_dim + output_dim)),
                (output_dim, input_dim)
            )
            params["b" + str(layer)] = np.random.uniform(
                -np.sqrt(6 / (input_dim + output_dim)),
                np.sqrt(6 / (input_dim + output_dim)),
                (output_dim, 1)
            )
        # He weight initialization
        elif initialization == "he":
            params["W" + str(layer)] = np.random.normal(
                0, np.sqrt(2 / input_dim), (output_dim, input_dim)
            )
            params["b" + str(layer)] = np.random.normal(
                0, np.sqrt(2 / input_dim), (output_dim, 1)
            )
        # Default initialization uniform distribution
        else:
            params["W" + str(layer)] = np.random.uniform(-0.5, 0.5, (output_dim, input_dim))
            params["b" + str(layer)] = np.random.uniform(-0.5, 0.5, (output_dim, 1))

    return params

def forward_prop(X, params, activation="relu", dropout_prob=0):
    """
    Perform forward propagation through the neural network to compute activations.

    Parameters:
    X (numpy.ndarray): Input data of shape (input_size, m), where input_size is the number of features and m is the number of examples.
    params (dict): Dictionary containing network parameters including weights (W) and biases (b) for each layer.
    dropout_prob (float): Dropout probability for applying dropout regularization to hidden layers (default 0).

    Returns:
    activations (dict): Dictionary containing computed activations for each layer.
    dropout_masks (dict): Dictionary containing dropout masks applied to hidden layers.
    """
    # Get the number of layers directly from the length of W parameters
    L = len(params) // 2
    activations = {}
    activations["A0"] = X  # Input activations
    dropout_masks = {}  # Dictionary to store dropout masks

    for l in range(1, L):
        # Calculate Z and A for intermediate layers using ReLU activation
        activations["Z" + str(l)] = np.dot(params["W" + str(l)], activations["A" + str(l - 1)]) + params["b" + str(l)]

        if activation == 'tanh':
            activations["A" + str(l)] = _tanh(activations["Z" + str(l)])  # Tanh activation
        elif activation == 'sigmoid':
            activations["A" + str(l)] = 1 / (1 + np.exp(-activations["Z" + str(l)])  # Sigmoid activation
        else:
            activations["A" + str(l)] = _relu(activations["Z" + str(l)])  # ReLU activation

        if l < L:
            dropout_mask = np.random.rand(*activations["A" + str(l)].shape) < (1 - dropout_prob)  # Inverted dropout mask
            activations["A" + str(l)] *= dropout_mask
            activations["A" + str(l)] /= (1 - dropout_prob)  # Scale to maintain expected value
            dropout_masks["D" + str(l)] = dropout_mask

    # Calculate Z and A for the output layer using softmax activation
    activations["Z" + str(L)] = np.dot(params["W" + str(L)], activations["A" + str(L - 1)]) + params["b" + str(L)]
    exp_scores = np.exp(activations["Z" + str(L)])
    activations["A" + str(L)] = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)  # Softmax activation

    return activations, dropout_masks

def back_prop(activations, params, Y, dropout_masks, activation="relu"):
    """
    Perform backpropagation to compute gradients of the loss with respect to parameters.

    Parameters:
    activations (dict): Dictionary containing computed activations for each layer during forward propagation.
    params (dict): Dictionary containing network parameters including weights (W) and biases (b) for each layer.
    Y (numpy.ndarray): True labels (ground truth) of shape (1, m), where m is the number of examples.
    dropout_masks (dict): Dictionary containing dropout masks applied to hidden layers.

    Returns:
    grads (dict): Dictionary containing computed gradients of the loss with respect to parameters.
    """
    L = len(params) // 2
    one_hot_Y = one_hot_encode(Y)
    m = one_hot_Y.shape[1]

    derivatives = {}
    grads = {}

    # for layer L
    derivatives["dZ" + str(L)] = activations["A" + str(L)] - one_hot_Y
    grads["dW" + str(L)] = (1 / m) * np.dot(derivatives["dZ" + str(L)], activations["A" + str(L - 1)].T)
    grads["db" + str(L)] = (1 / m) * np.sum(derivatives["dZ" + str(L)])

    # for layers L-1 to 1
    for l in reversed(range(1, L)):
        if activation == 'relu':
            _activation_derivative = _relu_derivative
        elif activation == 'tanh':
            _activation_derivative = _tanh_derivative
        elif activation == 'sigmoid':
            _activation_derivative = _sigmoid_derivative

        derivatives["dZ" + str(l)] = np.dot(params["W" + str(l + 1)].T, derivatives["dZ" + str(l + 1)]) * _activation_derivative(activations["Z" + str(l)])

        # apply dropout mask
        if l < L:
            derivatives["dZ" + str(l)] *= dropout_masks["D" + str(l)]

        grads["dW" + str(l)] = (1 / m) * np.dot(derivatives["dZ" + str(l)], activations["A" + str(l - 1)].T)
        grads["db" + str(l)] = (1 / m) * np.sum(derivatives["dZ" + str(l)], axis=1, keepdims=True)

    return grads

import numpy as np

def update_params(params, grads, alpha):
    """
    Update network parameters using gradient descent optimization.
    
    Parameters:
    params (dict): Dictionary containing network parameters including weights (W) and biases (b) for each layer.
    grads (dict): Dictionary containing gradients of the loss with respect to parameters.
    alpha (float): Learning rate, controlling the step size of parameter updates.
    
    Returns:
    params_updated (dict): Dictionary containing updated network parameters.
    """
    
    # number of layers
    L = len(params) // 2
    
    params_updated = {}
    
    for l in range(1, L + 1):
        params_updated["W" + str(l)] = params["W" + str(l)] - alpha * grads["dW" + str(l)]
        params_updated["b" + str(l)] = params["b" + str(l)] - alpha * grads["db" + str(l)]
    
    return params_updated

def train(X, Y, params, max_iter=10, learning_rate=0.1, dropout_prob=0, activation="relu"):
    """
    Trains a neural network model using gradient descent optimization.
    
    Parameters:
    X (numpy.ndarray): Input data of shape (input_size, num_samples).
    Y (numpy.ndarray): Ground truth labels of shape (1, num_samples).
    params (dict): Dictionary containing the parameters of the neural network.
    Keys are "W1", "b1", ..., "WL", "bL" where L is the number of layers.
    max_iter (int, optional): Maximum number of training iterations. Default is 10.
    learning_rate (float, optional): Learning rate for gradient descent. Default is 0.1.
    dropout_prob (float, optional): Dropout probability for regularization. Default is 0.
    activation (str, optional): Activation function to use in hidden layers. Default is "relu".
    
    Returns:
    tuple: A tuple containing updated parameters, list of accuracies per iteration, and list of losses per iteration.
    """
    
    # Initialize parameters Wl, bl for layers l = 1, ..., L
    L = len(params) // 2
    accuracies = []
    losses = []
    
    for iteration in range(1, max_iter + 1):
        # Forward propagation
        activations, dropout_mask = forward_prop(X, params, activation, dropout_prob)
        
        # Make predictions
        Y_hat = get_predictions(activations["A" + str(L)])
        
        # Compute accuracy
        accuracy = get_accuracy(Y_hat, Y)
        accuracies.append(accuracy)
        
        # Compute loss (cross-entropy)
        loss = cross_entropy(one_hot_encode(Y), activations["A" + str(L)])
        losses.append(loss)
        
        # Backpropagation
        gradients = back_prop(activations, params, Y, dropout_mask, activation)
        
        # Update parameters
        params = update_params(params, gradients, learning_rate)
        
        # Print progress
        if iteration == 1 or (iteration % 5) == 0:
            print("Iteration {}: Accuracy = {}, Loss = {}".format(iteration, accuracy, loss))
    
    return params, accuracies, losses

def test(X_test, Y_test, params, activation):
    """
    Evaluates the trained neural network model on test data.
    
    Parameters:
    X_test (numpy.ndarray): Test input data of shape (input_size, num_samples).
    Y_test (numpy.ndarray): Ground truth labels for test data of shape (1, num_samples).
    params (dict): Dictionary containing the parameters of the neural network.
    Keys are "W1", "b1", ..., "WL", "bL" where L is the number of layers.
    activation (str): Activation function used in the hidden layers during forward propagation.
    
    Returns:
    tuple: A tuple containing test accuracy and loss.
    """
    
    L = len(params) // 2
    
    # Forward propagation
    activations, _ = forward_prop(X_test, params, activation, 0)
    
    # Make predictions
    Y_hat = get_predictions(activations["A" + str(L)])
    
    # Compute test accuracy
    test_accuracy = get_accuracy(Y_hat, Y_test)
    loss = cross_entropy(one_hot_encode(Y_test), activations["A" + str(L)]
    
    return test_accuracy, loss

def one_hot_encode(Y):
    Y_one_hot = np.zeros((Y.shape[0], Y.max() + 1))
    Y_one_hot[np.arange(Y.shape[0]), Y] = 1
    Y_one_hot = Y_one_hot.T
    return Y_one_hot

def cross_entropy(Y_one_hot, Y_hat, epsilon=1e-10):
    Y_hat = np.clip(Y_hat, epsilon, 1.0 - epsilon)
    cross_entropy = -np.mean(np.sum(Y_one_hot * np.log(Y_hat), axis=0))
    return cross_entropy

def shuffle_rows(data):
    data = np.array(data)
    np.random.shuffle(data)
    return data

def normalize_pixels(data):
    return data / 255.0

def _relu(Z):
    return np.maximum(Z, 0)

def _softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def _relu_derivative(Z):
    return Z > 0

def _softmax_derivative(Z):
    dZ = np.exp(Z) / sum(np.exp(Z)) * (1.0 - np.exp(Z) / sum(np.exp(Z)))
    return dZ

def _tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def _tanh_derivative(x):
    return 1 - np.tanh(x) ** 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _sigmoid_derivative(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid * (1 - sigmoid)

def get_predictions(AL):
    return np.argmax(AL, axis=0)

def get_accuracy(Y_hat, Y):
    return np.sum(Y_hat == Y) / Y.size

def plot_accuracy_and_loss(accuracies, losses, max_iter):
    # Plot training accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iter + 1), accuracies, marker='o')
    plt.title("Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iter + 1), losses, marker='o', color='r')
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

def shuffle_rows(df):
    return df.sample(frac=1).reset_index(drop=True)

def normalize_pixels(X):
    return X / 255.0

def train(X, y, params, max_iter, alpha, dropout_prob, activation):
    # Implementation of your training function here
    pass

def test(X, y, trained_params, activation):
    # Implementation of your testing function here
    pass

# Load training data
df_train = pd.read_csv("mnist_train.csv")

# Shuffle the data
df_train = shuffle_rows(df_train)

# Split train and validation set
train_val_split = 0.8
train_size = round(df_train.shape[0] * train_val_split)
data_train = df_train[:train_size].T
data_val = df_train[train_size:].T

# Divide input features and target feature
X_train = data_train[1:]
y_train = data_train[0]
X_test = data_val[1:]
y_test = data_val[0]

# Normalize training and validation sets
X_train = normalize_pixels(X_train)
X_test = normalize_pixels(X_test)

print(X_test.shape)
print(y_test.shape)

# Set network and optimizer parameters
layers_dims = [784, 128, 10]
max_iter = 80
alpha = 0.1
dropout_prob = 0
accuracies = []
losses = []
activation = ["relu", "tanh", "sigmoid"]

for a in activation:
    params = init_params(layers_dims, 'xavier')

    # Train the network
    trained_params, train_acc, train_loss = train(X_train, y_train, params, max_iter, alpha, dropout_prob, a)
    plot_accuracy_and_loss(train_acc, train_loss, max_iter)
    accuracy, loss = test(X_test, y_test, trained_params, a)
    accuracies.append(accuracy)
    losses.append(loss)

plt.figure(figsize=(8, 6))
plt.bar(activation, accuracies)
plt.title('Accuracy by Activation Function')
plt.xlabel('Activation Function')
plt.ylabel('Accuracy')

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', color='black', fontweight='bold')

plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

test_accuracy, test_loss = test(X_test, y_test, trained_params)
print(f"Test Accuracy: {test_accuracy * 100:.2f}, Test loss: {test_loss:.2f}")

hidden_neurons_range = range(10, 201, 5)
accuracies = []
losses = []

for hidden_neurons in hidden_neurons_range:
    layers_dims = [784, hidden_neurons, 128, 10]
    params = init_params(layers_dims, 'xavier')
    trained_params, _, __ = train(X_train, y_train, params, max_iter, alpha, dropout_prob)
    accuracy, loss = test(X_test, y_test, trained_params)
    accuracies.append(accuracy)
    losses.append(loss)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(hidden_neurons_range, accuracies)
plt.xlabel('Number of Hidden Neurons')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Hidden Neurons')

plt.subplot(1, 2, 2)
plt.bar(hidden_neurons_range, losses)
plt.xlabel('Number of Hidden Neurons')
plt.ylabel('Loss')
plt.title('Loss vs. Number of Hidden Neurons')

plt.tight_layout()
plt.show()

# Deeper networks
m_layers_dims = [784, 512, 224, 128, 10]
m_params = init_params(layers_dims, 'xavier')
max_iter = 100
m_trained_params, m_train_accuracy, m_train_loss = train(X_train, y_train, m_params, max_iter, alpha, dropout_prob)
m_accuracy, m_loss = test(X_test, y_test, trained_params)
plot_accuracy_and_loss(m_train_accuracy, m_train_loss, max_iter)

s_layers_dims = [784, 128, 10]
s_params = init_params(layers_dims, 'xavier')
max_iter = 100
s_trained_params, s_train_accuracy, s_train_loss = train(X_train, y_train, params, max_iter, alpha, dropout_prob)
s_accuracy, s_loss = test(X_test, y_test, trained_params)
plot_accuracy_and_loss(s_train_accuracy, s_train_loss, max_iter)

# Compare initialization methods
layers_dims = [784, 256, 10]
max_iter = 50
alpha = 0.1
dropout_prob = 0.0

# Define the initialization methods you want to compare
initialization_methods = ['default', 'he', 'xavier']
activation = ['default', 'relu', 'sigmoid']

# Initialize an empty dictionary to store accuracies and losses for each method
results = {}

# Loop through each initialization method
for a in activation:
    for method in initialization_methods:
        # Initialize parameters using the current method
        params = init_params(layers_dims, method)
        
        # Train the network and collect accuracies and losses
        updated_params, accuracies, losses = train(X_train, y_train, params, max_iter, alpha, dropout_prob, a)
        
        # Store the results for the current method
        results[method] = {'accuracies': accuracies, 'losses': losses}

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot accuracy for each method
plt.subplot(1, 2, 1)
for method, data in results.items():
    plt.plot(range(1, max_iter + 1), data['accuracies'], label=method)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.grid()
plt.legend()

# Plot loss for each method
plt.subplot(1, 2, 2)
for method, data in results.items():
    plt.plot(range(1, max_iter + 1), data['losses'], label=method)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Comparison')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

dropout_prob = [num * 0.1 for num in range(0, int(1 / 0.1) + 1)]
for alpha in dropout_prob:
    print(alpha)

# Dropout during training
# set network and optimizer parameters
layers_dims = [784, 224, 10]
max_iter = 50
alpha = 0.1

dropout_prob = [0, 0.1, 0.2, 0.5, 0.8]
accuracies = []
losses = []

for d in dropout_prob:
    params = init_params(layers_dims, 'xavier')
    # train the network
    trained_params, train_acc, train_loss = train(X_train, y_train, params, max_iter, alpha, d, "relu")
    accuracy, loss = test(X_test, y_test, trained_params, a)
    plot_accuracy_and_loss(train_acc, train_loss, max_iter)
    accuracies.append(accuracy)
    losses.append(loss)

# Dropout during training
# set network and optimizer parameters
layers_dims = [784, 512, 10]
max_iter = 150
alpha = 0.1

dropout_prob = [0, 0.1, 0.2]
accuracies = []
losses = []

for d in dropout_prob:
    params = init_params(layers_dims, 'xavier')
    # train the network
    trained_params, train_acc, train_loss = train(X_train, y_train, params, max_iter, alpha, d, "relu")
    accuracy, loss = test(X_test, y_test, trained_params, "relu")
    plot_accuracy_and_loss(train_acc, train_loss, max_iter)
    accuracies.append(accuracy)
    losses.append(loss)
print(accuracies)
print(losses)
print(dropout_prob)

plt.figure(figsize=(10, 6))
# Bar plot for accuracies
plt.subplot(1, 2, 1)
plt.bar(['0', '0.1', '0.2'], accuracies, color='blue')
plt.title('Accuracy vs. Dropout Probability')
plt.xlabel('Dropout Probability')
plt.ylabel('Accuracy')

# Bar plot for losses
plt.subplot(1, 2, 2)
plt.bar(['0', '0.1', '0.2'], losses, color='red')
plt.title('Loss vs. Dropout Probability')
plt.xlabel('Dropout Probability')
plt.ylabel('Loss')
plt.show()

# Define hyperparameter grid
layer_options = [
    [784, 256, 10],
    [784, 256, 256, 10],
    [784, 256, 128, 64, 10],
]
alpha = 0.1
init_methods = ['default', 'xavier', 'he']
max_iter = 50
best_accuracy = 0
best_params = None

for layers_dims in layer_options:
    for initialization_method in init_methods:
        # Initialize Neural network
        params = init_params(layers_dims, initialization_method)

        # Train the neural network
        updated_param, accuracy, loss = train(X_train, y_train, params, max_iter, alpha)

        # Check if this set of hyperparameters is the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                'layers_dims': layers_dims,
                'alpha': alpha,
                'initialization_method': initialization_method
            }          
print("Best hyperparameters:", best_params)
print("Best validation accuracy:", best_accuracy)

# Test best model on test data
test_accuracy = evaluate_model(model, X_test, y_test)
print("Test accuracy:", test_accuracy)
