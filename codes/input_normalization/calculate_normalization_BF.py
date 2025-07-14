# -*- coding: utf-8 -*-

import argparse
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision import datasets, transforms
import os

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_mode', default='whole', type=str, 
                    help="Specify patient IDs (folder names) as a comma-separated list (e.g., '01,03'). Use 'whole' to include all.")
parser.add_argument('--partition_name', type=str)

def main(args):
    print('---------')
    print(args.partition_name)
    dataset_mode = args.dataset_mode

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root='/data/BF', transform=transform)

    # Get all available class (folder) names
    all_classes = dataset.classes

    # If 'whole' is specified, use all folders
    if dataset_mode == 'whole':
        selected_classes = all_classes
    else:
        # Convert comma-separated input into a list
        selected_classes = dataset_mode.split(",")
        # Validate if the provided classes exist
        invalid_classes = [cls for cls in selected_classes if cls not in all_classes]
        if invalid_classes:
            raise ValueError(f"Invalid classes provided: {invalid_classes}. Available classes: {all_classes}")

    # Filter dataset based on selected classes
    filtered_indices = [i for i, (_, label) in enumerate(dataset.samples) if dataset.classes[label] in selected_classes]

    # Create a subset of the dataset
    filtered_dataset = Subset(dataset, filtered_indices)

    # Print dataset details
    print(f"Selected classes: {selected_classes}")
    print(f"Total images after filtering: {len(filtered_dataset)}")

    # Create DataLoader
    dataloader = DataLoader(filtered_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Compute mean and std
    mean = torch.zeros(3).to(device)
    std = torch.zeros(3).to(device)
    count = 0

    print("Calculating mean and standard deviation...")
    for images, _ in tqdm(dataloader):
        images = images.to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        count += batch_samples

    mean /= count
    std /= count

    print("Mean:", mean.tolist())
    print("Std:", std.tolist())

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
