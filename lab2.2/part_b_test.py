import os
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle
from testloader import CustomImageDataset

# Define the CNN model (same as in train script)
class MultiClassCNN(nn.Module):
    def __init__(self):
        super(MultiClassCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 1)
        self.fc1 = nn.Linear(128 * 11 * 24, 512)
        self.fc2 = nn.Linear(512, 8)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 11 * 24)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Testing function
def test_model(test_dataset_root, load_weights_path, save_predictions_path):
    transform = transforms.Compose([
        transforms.Resize((50, 100)),
        transforms.ToTensor()
    ])

    # Load dataset
    csv_path = os.path.join(test_dataset_root, "public_test.csv")
    test_dataset = CustomImageDataset(root_dir=test_dataset_root, csv=csv_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize model and load weights
    model = MultiClassCNN().float()
    model.load_state_dict(torch.load(load_weights_path))
    model.eval()

    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.float()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # Save predictions to a pickle file
    predictions = np.array(predictions)
    with open(save_predictions_path, 'wb') as f:
        pickle.dump(predictions, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_root', required=True, help='Path to the test dataset root')
    parser.add_argument('--load_weights_path', required=True, help='Path to load saved weights')
    parser.add_argument('--save_predictions_path', required=True, help='Path to save predictions')
    args = parser.parse_args()

    test_model(args.test_dataset_root, args.load_weights_path, args.save_predictions_path)
