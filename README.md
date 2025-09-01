Traffic Sign Classification (TensorFlow/Keras)
Overview

This project trains a Convolutional Neural Network (CNN) to classify traffic sign images into 43 categories.
The script loads images from a directory structure (one folder per class), preprocesses them to a fixed size, splits into train/test sets, builds a CNN, trains it, evaluates accuracy, and optionally saves the trained model.

Works out-of-the-box with datasets organized like GTSRB (German Traffic Sign Recognition Benchmark), i.e., class folders named 0 to 42.

Features

Simple directory-based data loading with OpenCV

Automatic one-hot encoding of labels

Train/test split with scikit-learn

Compact CNN (2× Conv+Pooling → Flatten → Dense)

Configurable image size, epochs, and split ratio via constants

Optional .h5 model saving

Requirements

Python 3.8+

TensorFlow 2.x

OpenCV (cv2)

NumPy

scikit-learn

Install with:

pip install tensorflow opencv-python numpy scikit-learn

Dataset Structure

Place your dataset under a directory where each subfolder is a numeric class label:

data_directory/
├─ 0/
│  ├─ img_0001.png
│  ├─ img_0002.jpg
│  └─ ...
├─ 1/
│  ├─ ...
├─ 2/
│  ├─ ...
└─ ...
   └─ 42/
      ├─ ...


Folder names must be integers in [0, NUM_CATEGORIES-1].

Images can be any common format readable by OpenCV (.png, .jpg, …).

Usage

Run the training script:

python traffic.py data_directory [model.h5]


Arguments

data_directory (required): path to the dataset root (see structure above).

model.h5 (optional): if provided, the trained model is saved to this file.

Examples

# Train and evaluate (no save)
python traffic.py ./traffic_data

# Train, evaluate, and save the model
python traffic.py ./traffic_data traffic_model.h5

How It Works
Key Hyperparameters (edit in the script)
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

Data Loading (load_data)

Recursively reads each class folder.

Loads images with OpenCV (cv2.imread), resizes to (IMG_WIDTH, IMG_HEIGHT).

Builds two lists:

images: list of H×W×3 numpy.ndarray (BGR order as read by OpenCV).

labels: integer class IDs.

Preprocessing

Labels are converted to one-hot vectors via:

tf.keras.utils.to_categorical(labels)


Train/test split via:

train_test_split(images, labels, test_size=TEST_SIZE)

Model (get_model)

Architecture:

Input: (IMG_WIDTH, IMG_HEIGHT, 3)
Conv2D(32, (3,3), relu)
MaxPooling2D(2,2)
Conv2D(64, (3,3), relu)
MaxPooling2D(2,2)
Flatten
Dense(128, relu)
Dense(NUM_CATEGORIES, softmax)


Compilation:

optimizer='adam'
loss='categorical_crossentropy'
metrics=['accuracy']

Training & Evaluation
model.fit(x_train, y_train, epochs=EPOCHS)
model.evaluate(x_test, y_test, verbose=2)

Saving

If model.h5 is passed, the trained model is saved:

model.save("model.h5")

Notes & Tips

Color channels: OpenCV loads images in BGR. TensorFlow typically uses RGB. For many tasks it won’t matter if consistent across the dataset, but for best practice convert to RGB:

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


Normalization: Consider scaling pixel values to [0,1]:

image = image.astype("float32") / 255.0


Augmentation: Improve generalization by augmenting training data (e.g., flips, rotations). You can add a tf.keras.preprocessing.image.ImageDataGenerator or tf.keras.layers.Random* layers.

Class imbalance: If some classes have fewer images, consider using class_weight in model.fit.

GPU: Training will be faster with a GPU-enabled TensorFlow installation.

Reproducibility: Set seeds in NumPy, TensorFlow, and Python’s random if you need deterministic behavior.

Inference (Using a Saved Model)
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("traffic_model.h5")

IMG_WIDTH, IMG_HEIGHT = 30, 30

img = cv2.imread("test_image.jpg")
img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
# Optional: convert to RGB and normalize as done in training
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = img.astype("float32") / 255.0

x = np.expand_dims(img, axis=0)  # shape: (1, H, W, 3)
probs = model.predict(x)[0]      # softmax probabilities
pred_class = np.argmax(probs)

print("Predicted class:", pred_class)
print("Class probabilities:", probs)




requirements.txt

tensorflow
opencv-python
numpy
scikit-learn



Acknowledgments

Inspired by the GTSRB dataset setup and standard CNN baselines for traffic sign recognition.
