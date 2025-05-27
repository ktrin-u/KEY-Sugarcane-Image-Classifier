import os
from pathlib import Path

import keras
import tensorflow as tf
from PIL import Image

#Upon inspecting the given data, there are 6 classes given
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

# Path to the folder that contains the test imagegs
TEST_PATH = Path("./test/")


# n is the number of images that will be used for predictions.
def method1_predict(n: int):
    # Model path and output filename to be used for the first method
    MODEL_PATH = "hf://ktrin-u/KEY-cnn"
    OUTPUT_FILENAME = Path("./method1-results.csv")

    # This loads the CNN model and prepares the output file to be written to
    model = keras.models.load_model(MODEL_PATH)
    output_file = open(OUTPUT_FILENAME, "w+")
    output_file.write("image_filename,predicted_label\n") # Add header row
    image_names = os.listdir(TEST_PATH)
    image_names.sort(key=lambda name: int(name.split(".")[0]))
    image_names = image_names[:n]

    # We then go through each image, open it, process it to be suitable for the model, then get the predicted class
    # The result is a combination of the image name and its predicted class.
    # Each result is written into the output file followed by a new line
    for img_name in image_names[:n if n > 0 else None:]:
        img = Image.open(TEST_PATH.joinpath(img_name))
        img_array = keras.utils.img_to_array(img)
        img_resized = keras.layers.Resizing(96, 96, pad_to_aspect_ratio=True)(img_array)
        img_resized = tf.expand_dims(img_resized, axis=0)
        prediction = model.predict(img_resized, batch_size=1, verbose="2")  # type: ignore
        prediction_class_index = prediction.argmax()
        # print(prediction)
        result = f"{img_name},{classes[prediction_class_index]}"
        print(result)
        output_file.write(result + "\n")

    #Once all the images have been given predictions, the output file is closed
    output_file.close()


# n is the number of images that will be used for predictions.
def method2_predict(n: int):
    # Model path and output filename to be used for the first method
    MODEL_PATH = "hf://ktrin-u/KEY-ViT"
    OUTPUT_FILENAME = Path("./method2-results.csv")

    # This loads the ViT model and prepares the output file to be written to
    model = keras.models.load_model(MODEL_PATH)
    output_file = open(OUTPUT_FILENAME, "w+")
    output_file.write("image_filename,predicted_label\n") # Add header row
    image_names = os.listdir(TEST_PATH)
    image_names.sort(key=lambda name: int(name.split(".")[0]))

    # We then go through each image, open it, process it to be suitable for the model, then get the predicted class
    # The result is a combination of the image name and its predicted class.
    # Each result is written into the output file followed by a new line
    for img_name in image_names[:n if n > 0 else None:]:
        img = Image.open(TEST_PATH.joinpath(img_name))
        img_array = keras.utils.img_to_array(img)
        img_resized = keras.layers.Resizing(96, 96, pad_to_aspect_ratio=True)(img_array)
        img_resized = tf.expand_dims(img_resized, axis=0)
        prediction = model.predict(img_resized, batch_size=1, verbose="2")  # type: ignore
        prediction_class_index = prediction.argmax()
        # print(prediction)
        result = f"{img_name},{classes[prediction_class_index]}"
        print(result)
        output_file.write(result + "\n")

    #Once all the images have been given predictions, the output file is closed
    output_file.close()

# method1_predict()
#method2_predict()