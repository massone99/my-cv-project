import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

class MobileNetModel:
    def __init__(
        self, input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    ) -> Model:
        self.base_model = MobileNetV2(
            input_shape=input_shape, include_top=include_top, weights=weights
        )
        self.base_model.trainable = False
        x = Flatten()(self.base_model.output)
        x = Dense(1024, activation="relu")(x)
        outputs = Dense(5, activation="softmax")(x)  # Change number of units to 5
        outputs = Reshape((1, 5))(outputs)  # Reshape the output to match the annotations
        self.model = Model(self.base_model.input, outputs)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy")  # Change loss function to categorical_crossentropy
        self.model

    def train(self, images, annotations) -> None:
        """
        Trains the model on the given images and annotations.

        Args:
            images: A list of preprocessed images.
            annotations: A list of corresponding annotations.
        """
        # Convert lists to numpy arrays
        images = np.array(images)
        annotations = np.array(annotations)

        # Reshape annotations to match the model's output shape
        annotations = annotations.reshape(-1, 1, 5)

        # Train the model
        self.model.fit(images, annotations)
    
    def predict(self, images) -> np.ndarray:
        """
        Makes predictions on the given images.

        Args:
            images: A list of preprocessed images.

        Returns:
            A numpy array of predictions.
        """
        # Convert list to numpy array
        images = np.array(images)

        # Make predictions
        predictions = self.model.predict(images)

        return predictions    