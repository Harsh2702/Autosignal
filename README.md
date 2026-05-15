# Autosignal

CNN-based traffic sign classifier trained on the GTSRB dataset. Classifies road signs into 43 categories and ships a tkinter GUI for real-time image prediction.

## What it does

- Loads the GTSRB training images, resizes them to 30×30, and trains a five-layer CNN
- Plots training/validation accuracy and loss curves
- Evaluates final accuracy on the GTSRB test set
- `gui.py` opens a desktop app where you upload any traffic sign image and get the predicted label

## Dataset

[GTSRB — German Traffic Sign Recognition Benchmark](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

Download and extract so the root contains `train/` (subdirectories 0–42) and `Test.csv`.

## Tech Stack

- Python 3
- TensorFlow / Keras
- OpenCV
- Pillow
- scikit-learn
- tkinter (GUI, included in standard Python on Windows)
- NumPy, pandas, matplotlib

## How to Run

```bash
pip install -r requirements.txt
```

**Train the model:**
```bash
python main.py --data-dir /path/to/gtsrb --epochs 10
```
Saves `traffic_classifier.h5` in the current directory.

**Launch the GUI:**
```bash
python gui.py
```
Click **Upload an image**, select a traffic sign image, then click **Classify Image** to see the predicted sign name.

## Model Architecture

```
Conv2D(64) → Conv2D(64) → MaxPool → Dropout(0.20)
Conv2D(32) → Conv2D(32) → MaxPool → Dropout(0.25)
Flatten → Dense(128) → Dropout(0.5) → Dense(43, softmax)
```
Trained with Adam optimizer and categorical cross-entropy loss.
