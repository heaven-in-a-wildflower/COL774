import numpy as np
import os
import pickle
import argparse
from part_d_trainloader import TrainDataLoader,TrainImageDataset, numpy_transform
from part_d_testloader import TestDataLoader,TestImageDataset, numpy_transform
from scipy.signal import convolve2d
import cv2

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

    def focal_loss(self, y_pred, y_true, alpha=1.0, gamma=2.0):
        # Focal loss implementation
        epsilon = 1e-10  # to avoid log(0)
        y_true = np.eye(self.layer_sizes[-1])[y_true.astype(int).flatten()].T
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy_loss = -np.sum(y_true * np.log(y_pred), axis=0)
        focal_loss_value = alpha * np.power(1 - y_pred, gamma) * cross_entropy_loss
        return np.mean(focal_loss_value)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500), dtype=np.float64))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def swish(self, z):
        return z * self.sigmoid(z)

    def swish_prime(self, z):
        sigmoid_z = self.sigmoid(z)
        return sigmoid_z + z * sigmoid_z * (1 - sigmoid_z)

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
                a = self.swish(z)
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
            delta = np.dot(self.weights[-l+1].T, delta) * self.swish_prime(z)
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True) / m
            nabla_w[-l] = np.dot(delta, activations[-l-1].T) / m

        return nabla_b, nabla_w

    def update_mini_batch(self, X_batch, Y_batch,epoch):
        nabla_b, nabla_w = self.backward_pass(X_batch, Y_batch)

        self.t += 1  # increment timestep

        learning_rate = adaptive_learning_rate(epoch)

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

            # Update weights and biases
            self.weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            self.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        activations, _ = self.forward_pass(X_batch)
        return self.focal_loss(activations[-1], Y_batch)  # Use focal loss here

    def train(self, save_path, X_train, Y_train, epochs, mini_batch_size):
        n = X_train.shape[1]
        
        for epoch in range(epochs):
            # Shuffle the training data
            permutation = np.random.permutation(n)
            X_train_shuffled = X_train[:, permutation]
            Y_train_shuffled = Y_train[:, permutation]

            # Create mini-batches
            mini_batches_X = [X_train_shuffled[:, k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            mini_batches_Y = [Y_train_shuffled[:, k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            total_loss = 0
            for X_mini, Y_mini in zip(mini_batches_X, mini_batches_Y):
                total_loss += self.update_mini_batch(X_mini, Y_mini,epoch)

            if epoch>400 and epoch%20==0:
                self.save_weights(save_path)

            # if X_val is not None and Y_val is not None:
            #     accuracy = self.evaluate(X_val, Y_val)
            #     print(f"Epoch {epoch + 1}: Validation Accuracy: {accuracy:.4f}, Loss: {total_loss/len(mini_batches_X):.4f}")
            # else:
            #     print(f"Epoch {epoch + 1} complete, Loss: {total_loss/len(mini_batches_X):.4f}")

    def predict(self, X):
        return np.argmax(self.forward_pass(X)[0][-1], axis=0)

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        correct_predictions = np.sum(predictions == Y.flatten())
        return correct_predictions / X.shape[1]

def reconstruct_image(flat_image, size=(25, 25)):
    return flat_image.reshape(size)

def apply_convolution(image, kernel):
    # Get dimensions
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    
    # Prepare output
    output = np.zeros_like(image)

    # Apply convolution
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded_image[i:i + kernel_size, j:j + kernel_size] * kernel)

    return output

def apply_morphological_transformations(image):
    # Define kernels for horizontal, vertical, left diagonal, and right diagonal
    kernels = {
        'horizontal': np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]),
        'vertical': np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        'left_diagonal': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        'right_diagonal': np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    }
    
    transformed_images = []
    for kernel in kernels.values():
        transformed_image = apply_convolution(image, kernel)
        transformed_images.append(transformed_image)
    
    return transformed_images

def pad_to_even(image):
    """Pad the image to even dimensions if necessary."""
    rows, cols = image.shape
    if rows % 2 == 1:
        image = np.pad(image, ((0, 1), (0, 0)), mode='constant')
    if cols % 2 == 1:
        image = np.pad(image, ((0, 0), (0, 1)), mode='constant')
    return image

def haar_wavelet_transform(image):
    """
    Perform 2D Haar Wavelet Transform on the input image.
    Works with both even and odd dimensions.
    """
    image = pad_to_even(image)
    rows, cols = image.shape
    output = np.zeros_like(image, dtype=float)

    # Compute averages and differences for rows
    output[:rows//2, :] = (image[0::2, :] + image[1::2, :]) / np.sqrt(2)
    output[rows//2:, :] = (image[0::2, :] - image[1::2, :]) / np.sqrt(2)

    # Compute averages and differences for columns
    temp = output.copy()
    output[:, :cols//2] = (temp[:, 0::2] + temp[:, 1::2]) / np.sqrt(2)
    output[:, cols//2:] = (temp[:, 0::2] - temp[:, 1::2]) / np.sqrt(2)

    # Separate coefficients
    cA = output[:rows//2, :cols//2]
    cH = output[:rows//2, cols//2:]
    cV = output[rows//2:, :cols//2]
    cD = output[rows//2:, cols//2:]

    return cA, (cH, cV, cD)

def wavelet_transform(image, wavelet='haar'):
    """
    Perform 2D Discrete Wavelet Transform.
    Currently only supports Haar wavelet.
    """
    if wavelet.lower() != 'haar':
        raise ValueError("Only Haar wavelet is supported in this implementation.")
    
    return haar_wavelet_transform(image)

def extract_wavelet_features(coeffs):
    """
    Extract features from wavelet coefficients.
    """
    cA, (cH, cV, cD) = coeffs
    features = np.hstack((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))
    return features

# Argument parser for command-line options
parser = argparse.ArgumentParser(description='Binary Classification Training and Evaluation')

# Adding command-line arguments
parser.add_argument('--dataset_root', type=str, required=True, help='Path to the training dataset')
parser.add_argument('--test_dataset_root', type=str, required=True, help='Path to the test/validation dataset')
parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the model weights')
parser.add_argument('--save_predictions_path', type=str, required=True, help='Path to save predictions')

args = parser.parse_args()

# Dataset paths
train_data_path = args.dataset_root
test_data_path = args.test_dataset_root
save_path=args.save_weights_path

train_dataset = TrainImageDataset(root_dir=train_data_path, csv=os.path.join(train_data_path, "train.csv"), transform=numpy_transform)
val_dataset = TestImageDataset(root_dir=test_data_path, csv=os.path.join(test_data_path, "val.csv"), transform=numpy_transform)

train_dataloader = TrainDataLoader(train_dataset, batch_size=len(train_dataset))
val_dataloader = TestDataLoader(val_dataset, batch_size=len(val_dataset))

X_train, Y_train = next(iter(train_dataloader))
X_val = next(iter(val_dataloader))

X_train = X_train.reshape(X_train.shape[0], -1).T  # Shape (625, 3200)
Y_train = Y_train.reshape(1, -1)  # Shape (1, 3200)
X_val = X_val.reshape(X_val.shape[0], -1).T
#Y_val = Y_val.reshape(1, -1)

# # Check the shape of X_train and Y_val
# print("X_train shape:", X_train.shape)
# print("Y_train shape:", Y_train.shape)

X_train_transformed = []
Y_train_transformed = []

# Apply transformations to training set
for i, flat_image in enumerate(X_train.T):  # Iterate over each flattened image with index
    image = reconstruct_image(flat_image)  # Reconstruct the 25x25 image
    
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    transformed_images = apply_morphological_transformations(blurred_image)
    
    original_label = Y_train[0, i]  # Access the correct label for the current image

    for transformed_image in transformed_images:
        wavelet_coeffs = wavelet_transform(transformed_image)  # Perform wavelet transform
        wavelet_features = extract_wavelet_features(wavelet_coeffs)  # Extract features
        
        X_train_transformed.append(wavelet_features)
        Y_train_transformed.append(original_label)  # Append the original label

# Convert to numpy arrays and reshape
X_train_transformed = np.array(X_train_transformed).T  # Transpose to match expected input shape
Y_train_transformed = np.array(Y_train_transformed).reshape(1, -1)

# Final shape check
# print("Transformed training data shape:", X_train_transformed.shape)
# print("Transformed labels shape:", Y_train_transformed.shape)

X_val_transformed = []
#Y_val_transformed = []

for i, flat_image in enumerate(X_val.T):  # Iterate over each flattened image with index
    image = reconstruct_image(flat_image)  # Reconstruct the 25x25 image
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply morphological transformations
    transformed_images = apply_morphological_transformations(blurred_image)
    
    # Get the original label for this image
    #original_label = Y_val[0, i]  # Access the correct label for the current image

    # Apply wavelet transform and extract wavelet features
    for transformed_image in transformed_images:
        wavelet_coeffs = wavelet_transform(transformed_image)  # Perform wavelet transform
        wavelet_features = extract_wavelet_features(wavelet_coeffs)  # Extract features
        
        # Append the wavelet features to the transformed dataset
        X_val_transformed.append(wavelet_features)
        #Y_val_transformed.append(original_label)  # Append the original label

# Convert to numpy arrays and reshape
X_val_transformed = np.array(X_val_transformed).T  # Transpose to match expected input shape
#Y_val_transformed = np.array(Y_val_transformed).reshape(1, -1)

# Final shape check
# print("Transformed valing data shape:", X_val_transformed.shape)
# print("Transformed labels shape:", Y_val_transformed.shape)

# Proceed with training
layer_sizes = [676, 256, 128, 8]
nn_model = MultiClassNN(layer_sizes)
nn_model.train(save_path,X_train_transformed, Y_train_transformed, epochs=525, mini_batch_size=448)

# # Evaluate on validation set
# accuracy = nn_model.evaluate(X_val_transformed, Y_val_transformed)
# print(f"Final Validation Accuracy: {accuracy:.4f}")

# # Print some predictions
# print("\nSample predictions:")
# sample_predictions = nn_model.predict(X_val_transformed[:, :10])
# print("Predicted:", sample_predictions)
# print("Actual:   ", Y_val_transformed[:10])

# After making predictions
predictions = nn_model.predict(X_val_transformed)

# Convert to a 1-D numpy array of integer class indices
pred = np.array(predictions).flatten()  # Ensure it's 1-D

# Save predictions as a pickle file
with open(args.save_predictions_path, 'wb') as f:
    pickle.dump(pred, f)

print(f"Predictions saved to {args.save_predictions_path}")