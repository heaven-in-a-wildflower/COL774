import os
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle
from testloader import CustomImageDataset

# Define the improved CNN model (same as in train script)
class ImprovedMultiClassCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(ImprovedMultiClassCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5, 11))  # Adaptive pooling to fixed output size
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 11, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedMultiClassCNN().to(device)
    model.load_state_dict(torch.load(load_weights_path, map_location=device))
    model.eval()

    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
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