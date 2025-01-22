import os
import torch
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

import numpy as np
from collections import Counter
import pandas as pd

## Data Wrangling ## 

def check_for_corrupt_images(folder_path):

    if not os.path.exists(folder_path):
        print(f"Error: Path '{folder_path}' does not exist.")
        return
    
    print(f"Checking images in: {folder_path}")
    changes_made = False
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify image integrity
            except (IOError, SyntaxError) as e:
                print(f"Removing corrupt image: {file_path}")
                os.remove(file_path)
                changes_made = True
    
    if not changes_made:
        print(f"All images in '{folder_path}' are intact.")
    else:
        print(f"Completed checking images. Some corrupt images were removed.")




# Scale thresholding function
def scale_thresholding(img_tensor, lower=0.0, upper=1.0):
    print(f"Before Thresholding: Min={img_tensor.min().item()}, Max={img_tensor.max().item()}")
    img_tensor = torch.clamp(img_tensor, min=lower, max=upper)
    print(f"After Thresholding: Min={img_tensor.min().item()}, Max={img_tensor.max().item()}")
    return img_tensor



def scale_thresholding_main(folder_path, output_path, batch_size=16, lower=0.0, upper=1.0, resize=False, new_size=(256, 256)):
    # saves all image type as what they were originally

    # Define transformations based on resize parameter
    if resize:
        transform = transforms.Compose([
            transforms.Resize(new_size),  # Resize to new dimensions
            transforms.ToTensor()         # Convert PIL images to PyTorch tensors
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()         # Keep original size and convert to tensor
        ])
    
    # Load dataset
    dataset = datasets.ImageFolder(root=folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # Keep shuffle=False to preserve order
    
    # Create output directory structure
    for class_name in dataset.classes:
        os.makedirs(os.path.join(output_path, class_name), exist_ok=True)

    # Process and save images
    for idx, (images, labels) in enumerate(dataloader):
        for i, (image, label) in enumerate(zip(images, labels)):
            # Apply thresholding
            scaled_image = torch.clamp(image, min=lower, max=upper)
            
            # Preserve original file name and add suffix
            original_file_path = dataset.samples[idx * batch_size + i][0]  # Get the original file path
            original_file_name = os.path.basename(original_file_path)      # Extract the file name
            file_name, ext = os.path.splitext(original_file_name)          # Split name and extension
            new_file_name = f"{file_name}_rescaled{ext}"                   # Add suffix
            
            # Save to the correct class folder
            class_name = dataset.classes[label]
            save_path = os.path.join(output_path, class_name, new_file_name)
            save_image(scaled_image, save_path)
            print(f"Saved: {save_path}")



## EDA ## 

def analyze_dimensions(dimensions):
    dimension_counts = Counter(dimensions)
    dimension_counts = dict(sorted(dimension_counts.items(), key=lambda item: item[1], reverse=True))
    widths = [dim[0] for dim in dimensions]
    heights = [dim[1] for dim in dimensions]

    dimension_stats = {
        "average_width": np.mean(widths),
        "average_height": np.mean(heights),
        "median_width": np.median(widths),
        "median_height": np.median(heights),
        "mode_width": Counter(widths).most_common(1)[0][0],
        "mode_height": Counter(heights).most_common(1)[0][0],
        "min_width": np.min(widths),
        "min_height": np.min(heights),
        "max_width": np.max(widths),
        "max_height": np.max(heights),
        "std_dev_width": np.std(widths),
        "std_dev_height": np.std(heights),
    }

    print("Dimension Counts:")
    for dim, count in dimension_counts.items():
        print(f"{dim[0]}x{dim[1]}: {count} counts")

    print("\nDimension Statistics:")
    for stat, value in dimension_stats.items():
        print(f"{stat.replace('_', ' ').capitalize()}: {value}")

    print(f"\nAverage Dimension: {dimension_stats['average_width']}x{dimension_stats['average_height']}")


def analyze_file_sizes(file_sizes):
    file_sizes_np = np.array(file_sizes)
    file_size_stats = {
        "average": np.mean(file_sizes_np),
        "median": np.median(file_sizes_np),
        "mode": Counter(file_sizes).most_common(1)[0][0],
        "min": np.min(file_sizes_np),
        "max": np.max(file_sizes_np),
        "std_dev": np.std(file_sizes_np),
    }

    print("\nFile Size Statistics:")
    for stat, value in file_size_stats.items():
        print(f"{stat.capitalize()}: {value} bytes")

def analyze_file_types(file_types):
    file_type_counts = Counter(file_types)
    file_type_counts = dict(sorted(file_type_counts.items(), key=lambda item: item[1], reverse=True))

    print("\nFile Type Counts:")
    for ftype, count in file_type_counts.items():
        print(f"{ftype}: {count} counts")


def analyze_images(folder_path):
    # use the other 3 analysis functions at once

    dimensions = []
    file_sizes = []
    file_types = []

    # Traverse through the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    dimensions.append(img.size)
                    file_sizes.append(os.path.getsize(file_path))
                    file_types.append(img.format)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    analyze_dimensions(dimensions)
    analyze_file_sizes(file_sizes)
    analyze_file_types(file_types)