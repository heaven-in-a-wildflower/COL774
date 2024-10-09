import numpy as np
import os
import pickle
from preprocessor import CustomImageDataset, DataLoader, numpy_transform
import argparse

def adaptive_learning_rate(epoch, initial_lr=0.001, decay_rate=0.01):
    return initial_lr * np.exp(-decay_rate * epoch)

class MultiClassNN:
    def __init__(self, layer_sizes, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        np.random.seed(0)

        self.weights = [np.random.randn(y, x) * np.sqrt(2. / x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]

        # Adam parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize Adam variables
        self.m_w = [np.zeros_like(w) for w in self.weights]  # first moment vector
        self.v_w = [np.zeros_like(w) for w in self.weights]  # second moment vector
        self.m_b = [np.zeros_like(b) for b in self.biases]  # first moment vector for biases
        self.v_b = [np.zeros_like(b) for b in self.biases]  # second moment vector for biases
        self.t = 0  # timestep

    def save_weights(self, filename):
        weights_and_biases = {
            'weights': {f'fc{i+1}': w for i, w in enumerate(self.weights)},
            'bias': {f'fc{i+1}': b for i, b in enumerate(self.biases)}
        }
        with open(filename, 'wb') as f:
            pickle.dump(weights_and_biases, f)

    def cross_entropy_loss(self, y_pred, y_true):
        return -np.sum(y_true * np.log(np.clip(y_pred, 1e-10, 1.0)))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500), dtype=np.float64))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward_pass(self, X):
        a = X
        activations = [X]
        zs = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, a) + b
            zs.append(z)
            if i == self.num_layers - 2:
                a = self.softmax(z)
            else:
                a = self.sigmoid(z)
            activations.append(a)
        return activations, zs

    def backward_pass(self, X, Y):
        m = X.shape[1]
        activations, zs = self.forward_pass(X)

        Y_one_hot = np.eye(self.layer_sizes[-1])[Y.astype(int).flatten()].T

        delta = activations[-1] - Y_one_hot

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True) / m
        nabla_w[-1] = np.dot(delta, activations[-2].T) / m

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].T, delta) * self.sigmoid_prime(z)
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True) / m
            nabla_w[-l] = np.dot(delta, activations[-l-1].T) / m

        return nabla_b, nabla_w

    def update_mini_batch(self, X_batch, Y_batch,epoch,save_path):
        nabla_b, nabla_w = self.backward_pass(X_batch, Y_batch)

        self.t += 1  # increment timestep

        # Update weights and biases with Adam
        for i in range(len(self.weights)):
            # Update first moment estimate
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * nabla_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * nabla_b[i]

            # Update second moment estimate
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (nabla_w[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (nabla_b[i] ** 2)

            # Bias correction
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            learning_rate = adaptive_learning_rate(epoch)
            # Update weights and biases
            self.weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            self.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

            if epoch % 20 == 0:
                self.save_weights(save_path)

        activations, _ = self.forward_pass(X_batch)
        Y_one_hot = np.eye(self.layer_sizes[-1])[Y_batch.astype(int).flatten()].T
        return self.cross_entropy_loss(activations[-1], Y_one_hot) / X_batch.shape[1]


    def train(self, save_path, X_train, Y_train, epochs, mini_batch_size):
        n = X_train.shape[1]
        
        for epoch in range(epochs):
            # Shuffle the training data
            # permutation = np.random.permutation(n)
            # X_train_shuffled = X_train[:, permutation]
            # Y_train_shuffled = Y_train[:, permutation]

            # Create mini-batches
            mini_batches_X = [X_train[:, k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            mini_batches_Y = [Y_train[:, k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            total_loss = 0
            for X_mini, Y_mini in zip(mini_batches_X, mini_batches_Y):
                total_loss += self.update_mini_batch(X_mini, Y_mini,epoch,save_path)

            # if X_train is not None and Y_train is not None:
            #     accuracy = self.evaluate(X_train, Y_train)
            #     print(f"Epoch {epoch + 1}: Training Accuracy: {accuracy:.4f}, Loss: {total_loss/len(mini_batches_X):.4f}")
            # else:
            #     print(f"Epoch {epoch + 1} complete, Loss: {total_loss/len(mini_batches_X):.4f}")

    def predict(self, X):
        return np.argmax(self.forward_pass(X)[0][-1], axis=0)

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        correct_predictions = np.sum(predictions == Y.flatten())
        return correct_predictions / X.shape[1]
    
# Argument parser for command-line options
parser = argparse.ArgumentParser(description='Binary Classification Training and Evaluation')

# Adding command-line arguments
parser.add_argument('--dataset_root', type=str, required=True, help='Path to the training dataset')
parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the model weights')

args = parser.parse_args()

# Dataset paths
train_data_path = args.dataset_root
save_path=args.save_weights_path

# Data loading and preprocessing
root_dir = "./dataset_for_A2/dataset_for_A2/multi_dataset"
train_dataset = CustomImageDataset(root_dir=root_dir, csv=os.path.join(root_dir,"train.csv"), transform=numpy_transform)
#val_dataset = CustomImageDataset(root_dir=root_dir, csv=os.path.join(root_dir,"val.csv"), transform=numpy_transform)


train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))
#val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))

X_train, Y_train = next(iter(train_dataloader))
#X_val = next(iter(val_dataloader))

X_train = X_train.reshape(X_train.shape[0], -1).T
Y_train = Y_train.reshape(1, -1)
#X_val = X_val.reshape(X_val.shape[0], -1).T
#Y_val = Y_val.reshape(1, -1)

# Initialize and train the model
layer_sizes = [625, 512,256, 128,32, 8]
nn_model = MultiClassNN(layer_sizes)
nn_model.train(save_path, X_train, Y_train, epochs=2000, mini_batch_size=32)

# # Evaluate on validation set
# accuracy = nn_model.evaluate(X_val, Y_val)
# print(f"Final Validation Accuracy: {accuracy:.4f}")

# # Print some predictions
# print("\nSample predictions:")
# sample_predictions = nn_model.predict(X_val[:, :10])
# print("Predicted:", sample_predictions)
# print("Actual:   ", Y_val[0, :10])