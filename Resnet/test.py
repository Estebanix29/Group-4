# Test script for the trained ResNet model on test data

import os
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from Resnet import model as base_model

def evaluate_model(epoch):
    """
    Load model weights from checkpoint and evaluate on test data.
    
    Args:
        epoch: The epoch number of the weights checkpoint to load
    
    Returns:
        DataFrame with test IDs and predicted categories
    """
    # Clone base model and load weights for specified epoch
    eval_model = keras.models.clone_model(base_model)
    weights_path = f"Digits_ResNet_save_at_{epoch}.keras"
    print(f"Loading weights from {weights_path}")
    eval_model.load_weights(weights_path)
    
    # Load test IDs from CSV
    test_df = pd.read_csv("../Data/test.csv")
    image_paths = [os.path.join("../Data/test", f"{img_id}.png") for img_id in test_df["Id"].astype(str)]
    
    # Create dataset from file paths
    test_data = tf.data.Dataset.from_tensor_slices(image_paths)
    
    def load_and_preprocess(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, (32, 32))
        image = tf.cast(image, tf.float32) / 255.0
        return image
    
    test_data = test_data.map(load_and_preprocess).batch(32)
    
    # Predict on test data
    predictions = eval_model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)
    
    test_df["Category"] = predicted_labels
    test_df.to_csv("submission.csv", index=False)
    print(f"Predictions from epoch {epoch} saved to submission.csv")
    
    return test_df
    return test_df
