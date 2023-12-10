import glob
import os

import numpy as np
from ImageDirectory import ImageDirectory

from GPUConfig import GPUConfig
from keras.models import Model
from MobileNetModel import MobileNetModel


if __name__ == "__main__":
    GPUConfig.setup()
    model = MobileNetModel()
    train_data_path = (
        r"data\yolo-format\task_training-2023_12_04_21_10_08-yolo 1.1\obj_train_data"
    )
    test_data_path = (
        r"data/yolo-format/task_test-set-2023_12_04_21_10_14-yolo 1.1/obj_train_data"
    )
    train_dir = ImageDirectory(train_data_path)
    test_dir = ImageDirectory(test_data_path)
    # train_dir.visualize_all_images_and_annotations()
    images, annotations = train_dir.load_and_preprocess_data()
    test_images, test_annotations = test_dir.load_and_preprocess_data()

    # Initialize the model
    model: Model = MobileNetModel()

    # Train the model
    model.train(images, annotations)
    predictions = model.predict(test_images)
    test_dir.visualize_predictions(test_images, predictions)
