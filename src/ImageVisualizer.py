from abc import ABC, abstractmethod
import cv2
from matplotlib import pyplot as plt
import numpy as np

class AnnotationVisualizer(ABC):
    @abstractmethod
    def visualize_annotations_from_raw_data(self, image_path: str, annotations: str) -> None:
        pass

class YoloAnnotationVisualizer(AnnotationVisualizer):
    def visualize_annotations_from_raw_data(self, image_path: str, annotations: str) -> None:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        for annotation in annotations:
            class_index, x_center, y_center, width, height = annotation
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            x_top_left = int(x_center - width / 2)
            y_top_left = int(y_center - height / 2)
            cv2.rectangle(
                image,
                (x_top_left, y_top_left),
                (x_top_left + int(width), y_top_left + int(height)),
                (255, 0, 0),
                2,
            )
        plt.imshow(image)
        plt.show()
        
    def visualize_annotations_from_raw_data(self, image, annotations):
        """
        Visualizes the given annotations on the given image.

        Args:
            image: An image object.
            annotations: A list of corresponding annotations.

        Returns:
            None
        """
            # Convert the image to a supported depth
        image = image.astype(np.float32)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        for annotation in annotations:
            class_index, x_center, y_center, width, height = annotation
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            x_top_left = int(x_center - width / 2)
            y_top_left = int(y_center - height / 2)
            cv2.rectangle(
                image,
                (x_top_left, y_top_left),
                (x_top_left + int(width), y_top_left + int(height)),
                (255, 0, 0),
                2,
            )
        plt.imshow(image)
        plt.show()    