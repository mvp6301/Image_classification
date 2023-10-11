

import os
import random
import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, Xception, EfficientNetB0
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

# Define the path to your image dataset
dataset_path = "/content/drive/MyDrive/glasses-and-coverings/glasses-and-coverings"

# Get a list of image file paths
image_paths = list(paths.list_images(dataset_path))

# Shuffle the image paths
random.seed(42)
random.shuffle(image_paths)

# Initialize empty lists to store data and labels
data = []
labels = []

image_dims = (224, 224, 3)

for image_path in image_paths:
    # Load and resize the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_dims[1], image_dims[0]))

    # Preprocess the image
    image = image.astype("float") / 255.0

    # Append the image data to the 'data' list
    data.append(image)

    # Extract labels from the image path (modify this based on your dataset structure)
    label = image_path.split(os.path.sep)[-2].split("_")
    labels.append(label)

# Convert 'data' and 'labels' to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# Split the data into training and testing sets
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20)

# Define a function to create and compile models
def build_model(hp, base_model, model_name):
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=hp.Int('units_fc1', min_value=32, max_value=512, step=32), activation="relu"))
    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)))
    model.add(Dense(units=4, activation='softmax'))
    optimizer = Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    return model

# Create a tuner for hyperparameter tuning
tuners = {
    'MobileNetV2': RandomSearch(lambda hp: build_model(hp, MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=image_dims)), 'MobileNetV2'), objective='val_accuracy', max_trials=1, directory='keras_tuner_results_MobileNetV2', project_name='glasses_and_coverings_MobileNetV2'),
    'ResNet50V2': RandomSearch(lambda hp: build_model(hp, ResNet50V2(weights='imagenet', include_top=False, input_tensor=Input(shape=image_dims)), 'ResNet50V2'), objective='val_accuracy', max_trials=1, directory='keras_tuner_results_ResNet50V2', project_name='glasses_and_coverings_ResNet50V2'),
    'Xception': RandomSearch(lambda hp: build_model(hp, Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=image_dims)), 'Xception'), objective='val_accuracy', max_trials=1, directory='keras_tuner_results_Xception', project_name='glasses_and_coverings_Xception'),
    'EfficientNetB0': RandomSearch(lambda hp: build_model(hp, EfficientNetB0(weights='imagenet', include_top=False, input_tensor=Input(shape=image_dims)), 'EfficientNetB0'), objective='val_accuracy', max_trials=1, directory='keras_tuner_results_EfficientNetB0', project_name='glasses_and_coverings_EfficientNetB0')
}

best_models = {}

for model_name, tuner in tuners.items():
    tuner.search(
        trainX,
        trainY,
        epochs=10,
        validation_data=(testX, testY),
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_models[model_name] = best_model

    loss, accuracy = best_model.evaluate(testX, testY)
    print(f"Test accuracy for {model_name}: {accuracy}")

    best_model.save(f"best_model_{model_name}.h5")

