# Data read in
import tensorflow as tf
import keras

image_size = (32, 32)
batch_size = 32

train_data, val_data = keras.utils.image_dataset_from_directory(
    "../Data/train",
    labels="inferred",
    label_mode="int",
    validation_split=0.2,
    subset="both",
    seed=1354,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)


def normalize_data(data, label):
    data = tf.cast(data, tf.float32) / 255.0
    return data, label

print("Training Classes:")
class_names = train_data.class_names
print(class_names)

train_data = train_data.map(normalize_data)
val_data = val_data.map(normalize_data)