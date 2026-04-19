"""
Colorectal histology image classification using CNNs.

Trains a baseline CNN, applies data augmentation, and fine-tunes
a pre-trained VGG16 model on the Colorectal Histology dataset
(5,000 H&E-stained tissue images, 8 classes).

Reference: Kather JN et al., Scientific Reports, 2016.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
from tensorflow.image import resize_with_pad
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import VGG16


OUTPUT_DIR = "outputs"
RANDOM_SEED = 42
TEST_SIZE = 0.2


def plot_metric(history, metric="accuracy", save_path=None, title_prefix=""):
    training = history.history[metric]
    validation = history.history[f"val_{metric}"]
    best_epoch = validation.index(max(validation))

    plt.figure(figsize=(8, 5))
    plt.title(f"{title_prefix}{metric.capitalize()} Over Training Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.plot(training, label="Train")
    plt.plot(validation, label="Validation")
    plt.axvline(x=best_epoch, linestyle="--", color="green", label="Best Epoch")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def resize_images(images, height, width):
    return np.array(
        [resize_with_pad(img, height, width, antialias=True) for img in images]
    ).astype(np.uint8)


def load_dataset():
    print("Loading Colorectal Histology dataset...")
    data, info = tfds.load(
        "colorectal_histology", split="train",
        as_supervised=True, with_info=True,
    )
    feature_dict = info.features["label"].names
    images = np.array([image.numpy() for image, _ in data]).astype(np.uint8)
    labels = np.array([feature_dict[int(label)] for _, label in data])

    print(f"  Loaded {len(labels)} images with shape {images.shape[1:]}")
    print(f"  Classes: {sorted(set(labels))}")
    return images, labels


def build_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))

    for filters in [8, 16, 32, 64]:
        model.add(Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"],
    )
    return model


def augment_image(image):
    """Flip image vertically to create augmented sample."""
    return np.flipud(image)


def build_transfer_model(num_classes):
    """Swap VGG16's last layer for our 8-class output and freeze everything else."""
    base = VGG16(include_top=True, weights="imagenet")

    new_output = Dense(num_classes, activation="softmax")(base.layers[-2].output)
    model = Model(base.input, new_output)

    for layer in model.layers:
        layer.trainable = False
    model.layers[-1].trainable = True

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-transfer", action="store_true",
                        help="Skip VGG16 transfer learning (faster on CPU)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and split data
    images, labels = load_dataset()
    labels_ohe = np.array(pd.get_dummies(labels))
    ohe_to_label = {np.argmax(ohe): lbl for ohe, lbl in zip(labels_ohe, labels)}
    num_classes = len(set(labels))

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_ohe, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Save some sample images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(X_train[idx])
        ax.set_title(ohe_to_label[np.argmax(y_train[idx])], fontsize=10)
        ax.axis("off")
    plt.suptitle("Sample Training Images")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"), dpi=150)
    plt.close()

    # --- Step 1: Baseline CNN ---
    print("\n--- Training baseline CNN ---")
    cnn = build_cnn(X_train.shape[1:], num_classes)
    cnn.summary()

    history = cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
    plot_metric(history, save_path=os.path.join(OUTPUT_DIR, "baseline_accuracy.png"),
                title_prefix="Baseline — ")

    # --- Step 2: Data augmentation ---
    print("\n--- Training with augmented data ---")
    n_aug = 100
    X_aug = np.array([augment_image(X_train[i]) for i in range(n_aug)])
    y_aug = y_train[:n_aug].copy()

    # show original vs augmented
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for i in range(3):
        axes[0, i].imshow(X_train[i])
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        axes[1, i].imshow(X_aug[i])
        axes[1, i].set_title("Flipped")
        axes[1, i].axis("off")
    plt.suptitle("Data Augmentation Example")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "augmentation_comparison.png"), dpi=150)
    plt.close()

    aug_history = cnn.fit(X_aug, y_aug, validation_data=(X_test, y_test), epochs=5)
    plot_metric(aug_history, save_path=os.path.join(OUTPUT_DIR, "augmented_accuracy.png"),
                title_prefix="After Augmentation — ")

    # --- Step 3: Transfer learning with VGG16 ---
    if not args.skip_transfer:
        print("\n--- Transfer learning (VGG16) ---")
        transfer = build_transfer_model(num_classes)
        transfer.summary()

        print("Resizing images to 224x224 for VGG16...")
        X_train_r = resize_images(X_train, 224, 224)
        X_test_r = resize_images(X_test, 224, 224)

        t_history = transfer.fit(X_train_r, y_train,
                                 validation_data=(X_test_r, y_test), epochs=10)
        plot_metric(t_history,
                    save_path=os.path.join(OUTPUT_DIR, "transfer_learning_accuracy.png"),
                    title_prefix="Transfer Learning — ")
    else:
        print("\nSkipping transfer learning (--skip-transfer)")

    print(f"\nDone! Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
