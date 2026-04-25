# Test script for the trained ResNet model on test data

import os
import tensorflow as tf
import keras
import numpy as np
import pandas as pd

# Load the trained model (change to the epoch you want)
model = keras.models.load_model("Digits_ResNet_save_at_5.keras")

# Load test IDs from CSV
test_df = pd.read_csv("../Data/test.csv")
image_paths = [os.path.join("../Data/test", f"{img_id}.png") for img_id in test_df["Id"].astype(str)]

# Create dataset from file paths only
test_data = tf.data.Dataset.from_tensor_slices(image_paths)

def load_and_preprocess(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (32, 32))
    image = tf.cast(image, tf.float32) / 255.0
    return image

test_data = test_data.map(load_and_preprocess)
test_data = test_data.batch(32)

# Predict on test data
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

test_df["Category"] = predicted_labels
test_df.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")
