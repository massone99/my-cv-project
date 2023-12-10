import os
from PIL import Image
import numpy as np
import glob
from YoloReader import YOLOReader
from ImageVisualizer import YoloAnnotationVisualizer


class ImageDirectory:
    def __init__(self, directory):
        self.directory = directory

    def load_and_preprocess_data(self):
        """
        Loads and preprocesses all images and their corresponding annotations in the specified directory.

        Returns:
            A tuple of two lists: the preprocessed images and their corresponding annotations.
        """
        image_files = glob.glob(f"{self.directory}/*.png")
        images = []
        annotations = []
        for image_file in image_files:
            # Load and preprocess the image
            image = Image.open(image_file)
            image = image.resize(
                (224, 224)
            )  # Resize to the input size expected by MobileNet
            image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
            images.append(image)

            # Load and preprocess the annotations
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            annotation_file = f"{self.directory}/{base_name}.txt"
            annotation = YOLOReader.read_yolo_format(annotation_file)
            annotations.append(annotation)

        return images, annotations

    def visualize_all_images_and_annotations(self):
        """
        Visualizes all images and their corresponding annotations in the specified directory.

        Returns:
            None
        """
        image_files = glob.glob(f"{self.directory}/*.png")
        for image_file in image_files:
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            annotation_file = f"{self.directory}/{base_name}.txt"
            annotations = YOLOReader.read_yolo_format(annotation_file)
            yoloVisualizer = YoloAnnotationVisualizer()
            yoloVisualizer.visualize_annotations_from_raw_data(image_file, annotations)

    def visualize_predictions(self, images, predictions):
        """
        Visualizes the given predictions on the given images.

        Args:
            images: A list of preprocessed images.
            predictions: A list of corresponding predictions.

        Returns:
            None
        """
        for image, prediction in zip(images, predictions):
            # Convert the prediction to the format expected by YoloAnnotationVisualizer
            # prediction = YOLOReader.convert_prediction_to_yolo_format(prediction)

            # Visualize the prediction on the image
            yoloVisualizer = YoloAnnotationVisualizer()
            yoloVisualizer.visualize_annotations_from_raw_data(image, prediction)
