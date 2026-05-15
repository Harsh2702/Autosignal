import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


CLASSES = 43
IMAGE_SIZE = (30, 30)


def load_train_data(data_dir: str):
    data, labels = [], []
    for class_id in range(CLASSES):
        path = os.path.join(data_dir, "train", str(class_id))
        for filename in os.listdir(path):
            try:
                img = Image.open(os.path.join(path, filename)).resize(IMAGE_SIZE)
                data.append(np.array(img))
                labels.append(class_id)
            except Exception:
                print(f"Could not load {filename}")
    return np.array(data), np.array(labels)


def build_model(input_shape):
    model = Sequential([
        Conv2D(64, (5, 5), activation="relu", input_shape=input_shape),
        Conv2D(64, (5, 5), activation="relu"),
        MaxPool2D((2, 2)),
        Dropout(0.20),
        Conv2D(32, (3, 3), activation="relu"),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(CLASSES, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def plot_history(history):
    plt.figure(0)
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("accuracy.png")
    plt.show()

    plt.figure(1)
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()


def evaluate(model, data_dir: str):
    test_csv = pd.read_csv(os.path.join(data_dir, "Test.csv"))
    labels = test_csv["ClassId"].values
    imgs = test_csv["Path"].values

    data = []
    for img_path in imgs:
        full_path = os.path.join(data_dir, img_path)
        img = Image.open(full_path).resize(IMAGE_SIZE)
        data.append(np.array(img))

    X_test = np.array(data)
    preds = np.argmax(model.predict(X_test), axis=1)
    print(f"Test accuracy: {accuracy_score(labels, preds):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train traffic sign classifier")
    parser.add_argument("--data-dir", default=".", help="Path to GTSRB dataset root")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    print("Loading training data...")
    data, labels = load_train_data(args.data_dir)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    y_train_cat = to_categorical(y_train, CLASSES)
    y_test_cat = to_categorical(y_test, CLASSES)

    model = build_model(X_train.shape[1:])
    history = model.fit(X_train, y_train_cat, batch_size=32, epochs=args.epochs,
                        validation_data=(X_test, y_test_cat))

    model.save("traffic_classifier.h5")
    print("Model saved to traffic_classifier.h5")

    plot_history(history)
    evaluate(model, args.data_dir)


if __name__ == "__main__":
    main()
