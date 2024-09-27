import tensorflow as tf
import tensorboard as tb
import torch
import timm
from torchvision import transforms
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import yaml

# Load the pretrained ViT model
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.head = torch.nn.Identity()

# Freeze the weights for all layers
for param in model.parameters():
    param.requires_grad = False

def extract_features(image_path):
    image = Image.open(image_path)
    image = transforms.Resize((224, 224))(image)
    image = transforms.ToTensor()(image)
    extracted_features = model(image.unsqueeze(0))
    return extracted_features

if __name__ == "__main__":
    # Read configurations from config.yaml
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Get folder paths from the config
    folder_path = config['crop_folder_path']
    output_folder = config['runs_path']

    # Create a SummaryWriter for TensorBoard visualization
    writer = SummaryWriter(log_dir=output_folder)

    # List to store the image paths
    image_paths = []

    # Iterate over the images in the folder
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        
        # Skip directories
        if os.path.isdir(image_path):
            continue
        
        image_paths.append(image_path)

    # Count the total number of images
    total_images = len(image_paths)

    # Extract features and add to TensorBoard
    for i, image_path in enumerate(image_paths):
        extracted_features = extract_features(image_path)
        writer.add_embedding(extracted_features, global_step=os.path.basename(image_path))
        
        # Print progress
        print(f"Processed image {i+1}/{total_images}")

    # Close the SummaryWriter
    writer.close()

