import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from trainloader import *

# Define the CNN model
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
        x = x.view(-1, 128 * 11 * 24)  # Flatten the tensor using the correct shape
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training function
def train_model(train_dataset_root, save_weights_path):

    # Load dataset
    csv_path = os.path.join(train_dataset_root, "public_train.csv")
    train_dataset = CustomImageDataset(root_dir=train_dataset_root, csv=csv_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    l = []
    for inputs, labels in train_loader:
        l.append((inputs, labels))

    torch.manual_seed(0)
    
    # Initialize model, loss function, and optimizer
    model = MultiClassCNN().float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(8):
        # model.train()
        running_loss = 0.0

        # Calculate accuracy on the training set (without setting model to eval mode)
        correct = 0
        total = 0
        
        # Training step
        for inputs, labels in l:
            inputs, labels = inputs.float(), labels.long()  # Convert labels to long for CrossEntropyLoss
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
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
