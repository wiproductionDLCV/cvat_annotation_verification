import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from natsort import natsorted
from datetime import datetime
import yaml
import re

if __name__ == "__main__":
    # Read configurations from config.yaml
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Get folder paths from the config
    base_path = config['runs_path']
    folder_path = config['images_path']
    classes_file_path = config['classes_file_path']

    # Function to load features from tensors.tsv files
    def load_features(base_path):
        features = []
        image_names = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file == 'tensors.tsv':
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        line = f.readline().strip()
                        parts = line.split('\t')
                        features.append([float(part) for part in parts])
                        image_name = root.split(	os.sep)[-2]
                        image_names.append(image_name)
        return np.array(features), image_names

    # Load features and image names
    features, image_names = load_features(base_path)

    # Load classes from classes.txt
    classes = []
    with open(classes_file_path, 'r') as f:
        for line in f:
            classes.append(line.strip())

    # Function to update image names based on class names
   
    def update_image_names(image_names, classes):
        updated_image_names = []
        for image_name in image_names:
            for class_name in classes:
                if class_name in image_name:
                    image_name = image_name.split(class_name)[0].rstrip('_') + ".png"
                    break
            updated_image_names.append(image_name)
        return updated_image_names
    

    # Update image names
    updated_image_names = update_image_names(image_names, classes)

    # Group features and image names by class
    class_features = {class_name: [] for class_name in classes}
    class_image_names = {class_name: [] for class_name in classes}

    for feature, image_name, updated_image_name in zip(features, image_names, updated_image_names):
        for class_name in classes:
            if class_name in image_name:
                class_features[class_name].append(feature)
                class_image_names[class_name].append(updated_image_name)
                break

    # Normalize and run Isolation Forest for each class
    outlier_images = []

    for class_name in classes:
        if len(class_features[class_name]) > 0:
            features_normalized = StandardScaler().fit_transform(class_features[class_name])

            # Dynamically adjusting the Isolation Forest model
            clf = IsolationForest(random_state=42, contamination=0.05, n_estimators=100)
            outlier_pred = clf.fit_predict(features_normalized)

            # Check if there are at least 2 samples for PCA
            if len(features_normalized) >= 2:
                # Visualize in 2D PCA space
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(features_normalized)
                plt.close()

            # Re-identifying outlier images based on adjusted model
            outlier_images.extend([class_image_names[class_name][i] for i in range(len(class_image_names[class_name])) if outlier_pred[i] == -1])

    print("Outlier Images:", outlier_images)

    def find_image_positions(image_names, folder):
        # Get all image names in the folder and sort them naturally
        all_image_names = natsorted(os.listdir(folder))

        positions = []
        for image_name in image_names:
            try:
                # Find the position of each image name in the sorted list
                position = all_image_names.index(image_name)
                positions.append(position)  # Adding 1 to make it human-readable (1-based index)
            except ValueError:
                positions.append(-1)  # Append -1 if image not found
        return positions

    # Find the positions of the outlier images in the specified folder
    positions = find_image_positions(outlier_images, folder_path)

    # Print positions
    print("Positions of Outlier Images:", positions)

    # Write positions to a file
    today_date = datetime.now().strftime("%Y%m%d")
    output_file = f"doubtful_annotations_{today_date}.txt"
    with open(output_file, 'w') as f:
        for image_name, position in zip(outlier_images, positions):
            f.write(f"{image_name}: {position}\n")
            
    print(f"Positions of outlier images have been written to {output_file}")

