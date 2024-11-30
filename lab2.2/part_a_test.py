import os
import argparse
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from testloader import CustomImageDataset  
from part_a_train import SimpleCNN  


transform = transforms.Compose([
    transforms.Resize((50, 100)),  
    transforms.ToTensor(),         
])

def test_model(test_dataset_root, load_weights_path, save_predictions_path):
    # Load dataset
    csv_path = os.path.join(test_dataset_root, "public_test.csv")
    test_dataset = CustomImageDataset(root_dir=test_dataset_root, csv=csv_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize model and load weights
    model = SimpleCNN().float()
    model.load_state_dict(torch.load(load_weights_path))
    model.eval()

    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.float()
            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs)).numpy()  # Sigmoid for binary classification
            predictions.extend(preds.flatten())

    # Convert predictions to a 1D numpy array
    predictions = np.array(predictions, dtype=int)

    # Save predictions as a pickle file
    with open(save_predictions_path, 'wb') as f:
        pickle.dump(predictions, f)  # Use pickle to save the array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_root', required=True, help='Path to the test dataset root')
    parser.add_argument('--load_weights_path', required=True, help='Path to load weights')
    parser.add_argument('--save_predictions_path', required=True, help='Path to save predictions')
    args = parser.parse_args()

    test_model(args.test_dataset_root, args.load_weights_path, args.save_predictions_path)
