"""Simple neural network example using Keras, TensorFlow, and the MNIST dataset.

This script demonstrates an end-to-end classification workflow that includes:
- Loading and preprocessing data with NumPy and pandas
- Splitting the dataset with scikit-learn utilities
- Building, training, and evaluating a dense neural network
- Visualizing learning curves with Matplotlib

Run it directly with ``python src/nn_example_keras_tf_mnist.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
NUM_CLASSES = 10


@dataclass
class DatasetSplit:
    """In-memory representation of the flattened image data."""

    features: np.ndarray
    targets: np.ndarray


def load_and_prepare_data(
    sample_size: int = 20000,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> Tuple[DatasetSplit, DatasetSplit]:
    """Load MNIST digits and prepare train / test splits.

    Args:
        sample_size: Optional number of samples to keep for faster runs.
        test_size: Fraction reserved for final evaluation.
        random_state: Seed passed to pandas and scikit-learn.

    Returns:
        Tuple containing training and testing DatasetSplit objects.
    """

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # Combine the predefined splits so we can resample with pandas / scikit-learn utilities.
    images = np.concatenate([train_images, test_images], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

    # Flatten (28x28 -> 784) and scale the pixel intensities to [0, 1].
    flattened = images.reshape(images.shape[0], -1).astype("float32") / 255.0

    # Use pandas for beginner-friendly inspection or further preprocessing.
    df = pd.DataFrame(flattened)
    df["label"] = labels

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    features = df.drop(columns="label").to_numpy(dtype="float32")
    targets = df["label"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        stratify=targets,
        random_state=random_state,
    )

    return DatasetSplit(x_train, y_train), DatasetSplit(x_test, y_test)


def build_model(input_dimension: int, num_classes: int = NUM_CLASSES) -> keras.Model:
    """Create a small feedforward neural network."""

    model = keras.Sequential(
        [
            layers.Input(shape=(input_dimension,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history: keras.callbacks.History) -> None:
    """Visualize the training and validation curves."""

    metrics = history.history
    epochs = range(1, len(metrics["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, metrics["loss"], label="Training Loss")
    axes[0].plot(epochs, metrics["val_loss"], label="Validation Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].legend()

    axes[1].plot(epochs, metrics["accuracy"], label="Training Accuracy")
    axes[1].plot(epochs, metrics["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle("Training History", fontsize=14)
    plt.tight_layout()
    plt.show()


def evaluate_model(model: keras.Model, test_data: DatasetSplit) -> None:
    """Print accuracy, loss, and a detailed classification report."""

    test_loss, test_accuracy = model.evaluate(test_data.features, test_data.targets, verbose=0)
    print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}")

    predictions = model.predict(test_data.features, verbose=0)
    predicted_labels = predictions.argmax(axis=1)
    print("\nClassification report:\n")
    print(classification_report(test_data.targets, predicted_labels, digits=4))


def main() -> None:
    """Train, evaluate, and visualize a dense neural network on MNIST."""

    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)

    print("Loading and preprocessing MNIST data...")
    train_split, test_split = load_and_prepare_data()

    print("Building the neural network...")
    model = build_model(train_split.features.shape[1])
    model.summary()

    print("\nTraining the model...")
    history = model.fit(
        train_split.features,
        train_split.targets,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        verbose=2,
    )

    print("\nEvaluating the model...")
    evaluate_model(model, test_split)

    print("\nPlotting the learning curves...")
    plot_history(history)


if __name__ == "__main__":
    main()
