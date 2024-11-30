import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from trainloader import CustomImageDataset  # Ensure this file is correct

# Define the transformations [Import this definition as well to your training script!]
transform = transforms.Compose([
    transforms.Resize((50, 100)),  # Resize to 50x100 (height x width)
    transforms.ToTensor(),         # Convert the image to a tensor and also rescales the pixels by dividing them by 255
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 12 * 25, 1)  # Adjust based on output size after convolutions

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 25)  # Flatten the output
        x = self.fc1(x)
        return x

def train_model(train_dataset_root, save_weights_path):
    # Load dataset
    csv_path = os.path.join(train_dataset_root, "public_train.csv")
    train_dataset = CustomImageDataset(root_dir=train_dataset_root,csv=csv_path,transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    l=[]
    for inputs,labels in train_loader:
        l.append((inputs,labels))

    torch.manual_seed(0)

    # Initialize model, loss function, and optimizer
    model = SimpleCNN().float()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    correct = 0
    total = 0
    
    for epoch in range(8):
        #model.train()
        running_loss = 0.0
        for inputs, labels in l:
            inputs, labels = inputs.float(), labels.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            # Calculate accuracy
            predicted = torch.sigmoid(outputs).round()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

            running_loss += loss.item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/8], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save weights after each epoch
        torch.save(model.state_dict(), save_weights_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_root', required=True, help='Path to the train dataset root')
    parser.add_argument('--save_weights_path', required=True, help='Path to save weights')
    args = parser.parse_args()

    train_model(args.train_dataset_root, args.save_weights_path)
