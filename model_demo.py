import os
from pathlib import Path

import keras
import tensorflow as tf
from PIL import Image

classes = [
    "Banded_Chlorosis",
    "Brown_Rust",
    "Brown_Spot",
    "Viral",
    "Yellow_Leaf",
    "Healthy",
]
classes.sort()
# print(classes)

MODEL_PATH = "hf://ktrin-u/KEY-cnn"
TEST_PATH = Path("./test/")
OUTPUT_FILENAME = Path("./results.csv")

model = keras.models.load_model(MODEL_PATH)
output_file = open(OUTPUT_FILENAME, "w+")
image_names = os.listdir(TEST_PATH)
image_names.sort(key=lambda name: int(name.split(".")[0]))

for img_name in image_names:
    img = Image.open(TEST_PATH.joinpath(img_name))
    img_array = keras.utils.img_to_array(img)
    img_resized = keras.layers.Resizing(128, 128, pad_to_aspect_ratio=True)(img_array)
    img_resized = tf.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_resized, batch_size=1, verbose="3")  # type: ignore
    prediction_class_index = prediction.argmax()
    # print(prediction)
    result = f"{img_name},{classes[prediction_class_index]}"
    # print(result)
    output_file.write(result + "\n")

output_file.close()
