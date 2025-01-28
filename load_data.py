import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

# Dataset path
DATASET_PATH = "dataset"  # Replace with the path to your dataset folder

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (for compatibility with standard models)
    transforms.ToTensor()          # Convert images to PyTorch tensors
])

# Load dataset
dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

# Classes
classes = dataset.classes  # ['cats', 'tomatoes']
print(f"Classes: {classes}")

# Visualize some samples
def show_samples(data_loader):
    batch = next(iter(data_loader))
    images, labels = batch
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0))  # Convert tensor image to numpy array
        ax.set_title(f"Class: {classes[labels[i]]}")
        ax.axis('off')
    plt.show()

if __name__ == "__main__":
    print(f"Total samples in dataset: {len(dataset)}")
    show_samples(data_loader)
