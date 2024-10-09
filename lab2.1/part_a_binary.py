import numpy as np
import os
import pickle
import argparse
from preprocessor import CustomImageDataset, DataLoader, resize, to_tensor, numpy_transform

class BinaryClassificationNN:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        np.random.seed(0)
        self.weights = [np.random.randn(x, y).T * np.sqrt(2. / x) 
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]

    def save_weights(self, filename):
        weights_and_biases = {
            'weights': {f'fc{i+1}': np.array(w).transpose() for i, w in enumerate(self.weights)},
            'bias': {f'fc{i+1}': np.array(b).ravel() for i, b in enumerate(self.biases)}
        }
        with open(filename, 'wb') as f:
            pickle.dump(weights_and_biases, f)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_pass(self, x):
        a = x
        activations = [x]
        zs = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.sigmoid(z)
            activations.append(a)
        return activations, zs

    def backward_pass(self, x, y):
        activations, zs = self.forward_pass(x)
        delta = activations[-1] - y
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].T, delta) * self.sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        
        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backward_pass(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        m = len(mini_batch)
        self.weights = [w - learning_rate/m * nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - learning_rate/m * nb 
                       for b, nb in zip(self.biases, nabla_b)]

    def train(self, training_data, epochs, mini_batch_size, learning_rate, save_path=None):
        n = len(training_data)
        
        # Save initial weights
        #self.save_weights(os.path.join(save_path, "init_weights.pkl"))
        
        for epoch in range(epochs):
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            
            # Save weights after each epoch
            self.save_weights(save_path)

            # if validation_data:
            #     accuracy = self.evaluate(validation_data)
            #     print(f"Epoch {epoch + 1}: Validation Accuracy: {accuracy:.2f}")
            # else:
            #     print(f"Epoch {epoch + 1} complete")

    def predict(self, x):
        return self.forward_pass(x)[0][-1]

    def evaluate(self, test_data):
        test_results = [(self.predict(x) > 0.5).astype(int) for (x, y) in test_data]
        return sum(int(pred.item() == y.item()) for pred, (_, y) in zip(test_results, test_data)) / len(test_data)

# Argument parser for command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Train a binary classification neural network.")
    parser.add_argument('--dataset_for_root', type=str, required=True, help="Path to the dataset root.")
    parser.add_argument('--save_weights_path', type=str, required=True, help="Path to save the weights.")
    args = parser.parse_args()

    # Load and preprocess data
    root_dir = args.dataset_for_root
    save_path = args.save_weights_path

    train_dataset = CustomImageDataset(root_dir=root_dir, csv=os.path.join(root_dir, "train.csv"), transform=numpy_transform)
    #val_dataset = CustomImageDataset(root_dir=root_dir, csv=os.path.join(root_dir, "val.csv"), transform=numpy_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=1)
    #val_dataloader = DataLoader(val_dataset, batch_size=1)

    training_data = [(images.reshape(-1, 1), labels.reshape(-1, 1)) for images, labels in train_dataloader]
    #validation_data = [(images.reshape(-1, 1), labels.reshape(-1, 1)) for images, labels in val_dataloader]

    # Initialize and train the model
    layer_sizes = [625, 512, 256, 128, 1]
    nn_model = BinaryClassificationNN(layer_sizes)
    nn_model.train(training_data, epochs=15, mini_batch_size=256, learning_rate=0.001, save_path=save_path)

    # Evaluate on validation set
    # accuracy = nn_model.evaluate(validation_data)
    # print(f"Final Validation Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
