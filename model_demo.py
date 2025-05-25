import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to suppresss the tensorflow messages

import keras
import tensorflow as tf
from PIL import Image
from pathlib import Path


MODEL_PATH = "hf://ktrin-u/clearflow-drain-classifier"
TEST_PATH = Path("./test/")
OUTPUT_FILENAME = Path("./results.csv")

model = keras.models.load_model(MODEL_PATH)
output_file = open(OUTPUT_FILENAME, "xt")

for img_name in os.listdir(TEST_PATH):
    img = Image.open(TEST_PATH.joinpath(img_name))
    img_array = keras.utils.img_to_array(img)
    img_resized = keras.layers.Resizing(128, 128, pad_to_aspect_ratio=True)(img_array)
    img_resized = tf.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_resized, batch_size=1, verbose="0")  # type: ignore
    result = f"{img_name},{prediction}"
    print(result)
    output_file.write(result + "\n")

output_file.close()