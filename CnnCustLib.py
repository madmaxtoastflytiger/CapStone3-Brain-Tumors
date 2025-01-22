import os
import torch
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from statistics import mode, StatisticsError

import numpy as np
from collections import Counter
import pandas as pd

## Data Wrangling ## 

def check_for_corrupt_images(folder_path, remove_corrupted=True):
    """
    Check for corrupted images in the specified folder.

    Parameters:
        folder_path (str): Path to the folder containing images.
        remove_corrupted (bool): If True, remove corrupted images; otherwise, just report them.

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        print(f"Error: Path '{folder_path}' does not exist.")
        return

    print(f"Checking images in: {folder_path}")
    corrupted_count = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify image integrity
            except (IOError, SyntaxError):
                corrupted_count += 1
                if remove_corrupted:
                    print(f"Removing corrupt image: {file_path}")
                    os.remove(file_path)

    if corrupted_count == 0:
        print(f"All images in '{folder_path}' are intact.")
    else:
        if remove_corrupted:
            print(f"Completed checking images. {corrupted_count} corrupt images were removed.")
        else:
            print(f"Completed checking images. {corrupted_count} corrupt images were found.")




# Scale thresholding function
def scale_thresholding(img_tensor, lower=0.0, upper=1.0):
    print(f"Before Thresholding: Min={img_tensor.min().item()}, Max={img_tensor.max().item()}")
    img_tensor = torch.clamp(img_tensor, min=lower, max=upper)
    print(f"After Thresholding: Min={img_tensor.min().item()}, Max={img_tensor.max().item()}")
    return img_tensor



def scale_thresholding_main(folder_path, output_path, batch_size=16, lower=0.0, upper=1.0, resize=False, new_size=(256, 256)):
    # saves all image type as what they were originally, so a png will still be a png and not converted into a jpg

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

def analyze_file_type(folder_path):
    """
    Scans through all image files in a folder and returns a count of file types.
    
    Args:
        folder_path (str): Path to the folder containing image files.
        
    Returns:
        dict: A dictionary with file extensions as keys and their counts as values,
              including the total number of files scanned.
    """
    file_types = []
    total_files = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            total_files += 1
            _, ext = os.path.splitext(file)
            file_types.append(ext.lower())
    
    return {
        'file_type_counts': Counter(file_types),
        'total_files_scanned': total_files
    }


def analyze_file_size(folder_path, unit="KB"):
    """
    Scans through all image files in a folder and provides descriptive stats for file sizes.
    
    Args:
        folder_path (str): Path to the folder containing image files.
        unit (str): Unit for the file size stats ('Bytes', 'KB', 'MB', or 'GB').
                   Default is 'KB'.
        
    Returns:
        dict: A dictionary containing min, max, mean, median, mode, std, and total file sizes 
              in the specified unit, as well as the total number of files scanned.
    """
    file_sizes = []
    total_files = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                file_sizes.append(os.path.getsize(file_path))
    
    # Conversion factors
    unit_conversion = {
        "Bytes": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3
    }
    
    if unit not in unit_conversion:
        raise ValueError(f"Invalid unit '{unit}'. Choose from 'Bytes', 'KB', 'MB', or 'GB'.")
    
    conversion_factor = unit_conversion[unit]
    
    if file_sizes:
        try:
            mode_size = mode(file_sizes) / conversion_factor
        except StatisticsError:  # Handle no unique mode
            mode_size = None
        
        return {
            f'min_size_{unit}': min(file_sizes) / conversion_factor,
            f'max_size_{unit}': max(file_sizes) / conversion_factor,
            f'mean_size_{unit}': np.mean(file_sizes) / conversion_factor,
            f'median_size_{unit}': np.median(file_sizes) / conversion_factor,
            f'std_size_{unit}': np.std(file_sizes) / conversion_factor,
            f'mode_size_{unit}': mode_size,
            'total_files_scanned': total_files
        }
    else:
        return {'total_files_scanned': total_files}

    


def analyze_file_dimensions(folder_path):
    """
    Scans through all image files in a folder and provides a count of dimensions
    and descriptive stats for the dimensions. Dimension counts are sorted in descending order.
    
    Args:
        folder_path (str): Path to the folder containing image files.
        
    Returns:
        dict: A dictionary containing counts of dimensions, descriptive stats including mode and std,
              and the total number of files scanned.
    """
    dimensions = []
    total_files = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    dimensions.append(img.size)  # (width, height)
            except Exception:
                continue  # Skip non-image files or unsupported formats
    
    dimension_counts = Counter(dimensions)
    # Sort dimension counts in descending order
    sorted_dimension_counts = dict(sorted(dimension_counts.items(), key=lambda x: x[1], reverse=True))
    
    if dimensions:
        width_stats = [dim[0] for dim in dimensions]
        height_stats = [dim[1] for dim in dimensions]
        
        try:
            mode_width = mode(width_stats)
        except StatisticsError:  # Handle no unique mode
            mode_width = None
            
        try:
            mode_height = mode(height_stats)
        except StatisticsError:  # Handle no unique mode
            mode_height = None
        
        return {
            'dimension_counts': sorted_dimension_counts,
            'width_stats': {
                'min_width': min(width_stats),
                'max_width': max(width_stats),
                'mean_width': np.mean(width_stats),
                'median_width': np.median(width_stats),
                'std_width': np.std(width_stats),
                'mode_width': mode_width
            },
            'height_stats': {
                'min_height': min(height_stats),
                'max_height': max(height_stats),
                'mean_height': np.mean(height_stats),
                'median_height': np.median(height_stats),
                'std_height': np.std(height_stats),
                'mode_height': mode_height
            },
            'total_files_scanned': total_files
        }
    else:
        return {'total_files_scanned': total_files}