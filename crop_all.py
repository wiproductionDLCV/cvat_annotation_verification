import os
import argparse
import yaml
from image_utils import load_image, create_directory, save_image

class RegionOfInterestExtractor:
    """
    A class to extract region of interest objects from YOLO training dataset images.
    """

    def __init__(self, config):
        """
        Initialize the RegionOfInterestExtractor class with configurations.

        Args:
            config (dict): Dictionary containing configuration parameters.
        """
        self.images_folder = config['images_path']
        self.labels_folder = config['labels_path']
        self.output_folder = config['crop_folder_path']
        self.classes_file = config['classes_file_path']

    def create_output_folder(self):
        """
        Create the output folder if it doesn't exist.
        """
        create_directory(self.output_folder)

    def load_image(self, image_file):
        """
        Load an image from the images folder.

        Args:
            image_file (str): The filename of the image.

        Returns:
            numpy.ndarray: The loaded image.
        """
        image_path = os.path.join(self.images_folder, image_file)
        return load_image(image_path)

    def read_yolo_detection_file(self, image_file):
        """
        Read the corresponding YOLO detection file for a given image.

        Args:
            image_file (str): The filename of the image.

        Returns:
            list: A list of dictionaries containing bounding box information.
        """
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(self.labels_folder, label_file)

        if not os.path.isfile(label_path):
            return []

        with open(label_path, "r") as file:
            lines = file.readlines()

        yolo_data = []
        for line in lines:
            data = line.strip().split()
            if len(data) >= 5:
                class_index = int(data[0])
                x_center, y_center, box_width, box_height = map(float, data[1:])
                yolo_data.append({"class_index": class_index, "x_center": x_center, "y_center": y_center, "box_width": box_width, "box_height": box_height})

        return yolo_data

    def extract_roi(self, image, yolo_data, class_name):
        """
        Extract region of interest from an image based on YOLO data.

        Args:
            image (numpy.ndarray): The image to process.
            yolo_data (list): The YOLO detection data corresponding to the image.
            class_name (str): The class name for which to extract region of interest.

        Returns:
            list: A list of region of interest images as numpy arrays.
        """
        class_index = self.get_class_index(class_name)
        if class_index is None or not yolo_data:
            return []

        roi_images = []

        for data in yolo_data:
            detected_class_index = data["class_index"]
            if detected_class_index == class_index:
                x_center = data["x_center"]
                y_center = data["y_center"]
                box_width = data["box_width"]
                box_height = data["box_height"]

                x = int((x_center - (box_width / 2)) * image.shape[1])
                y = int((y_center - (box_height / 2)) * image.shape[0])
                w = int(box_width * image.shape[1])
                h = int(box_height * image.shape[0])

                # Check if the ROI dimensions are valid
                if w > 0 and h > 0:
                    roi = image[y:y + h, x:x + w]
                    roi_images.append(roi)

        return roi_images

    def save_roi(self, image_file, rois, class_name):
        """
        Save the extracted region of interest as separate images.

        Args:
            image_file (str): The filename of the original image.
            rois (list): A list of region of interest images as numpy arrays.
            class_name (str): The class name for which the ROIs are extracted.
        """
        if rois:
            image_name = os.path.splitext(image_file)[0]
            for i, roi in enumerate(rois):
                output_file = f"{image_name}_{class_name}_{i}.png"
                output_path = os.path.join(self.output_folder, output_file)

                print(f"Saving image {i + 1} of {len(rois)} for class {class_name} to {output_path}")

                if roi is None:
                    print(f"Warning: Empty ROI for {output_path}")
                elif roi.size == 0:
                    print(f"Warning: Empty ROI (size is zero) for {output_path}")
                else:
                    save_image(roi, output_path)
        else:
            print(f"No regions of interest found in {image_file} for class {class_name}.")

    def get_class_index(self, class_name):
        """
        Get the index of the class from the classes.txt file.

        Args:
            class_name (str): The class name for which to extract region of interest.

        Returns:
            int: The index of the class or None if not found.
        """
        if not os.path.isfile(self.classes_file):
            return None

        with open(self.classes_file, "r") as file:
            classes = file.read().strip().split('\n')

        if class_name in classes:
            return classes.index(class_name)
        return None

    def process_images(self):
        """
        Process each image in the images folder, extract region of interest, and save them as separate images.
        """
        self.create_output_folder()

        # Read class names from classes.txt
        with open(self.classes_file, "r") as file:
            classes = file.read().strip().split('\n')

        # Process each image in the images folder for each class
        for class_name in classes:
            for image_file in os.listdir(self.images_folder):
                image = self.load_image(image_file)
                yolo_data = self.read_yolo_detection_file(image_file)
                rois = self.extract_roi(image, yolo_data, class_name)
                self.save_roi(image_file, rois, class_name)
if __name__ == "__main__":
    # Read configurations from config.yaml
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Create an instance of the RegionOfInterestExtractor class and process the images
    extractor = RegionOfInterestExtractor(config)
    extractor.process_images()

