import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to suppresss the tensorflow messages

from pathlib import Path

import keras
import tensorflow as tf
from django.http import JsonResponse
from ninja import Form, NinjaAPI
from ninja.files import UploadedFile
from PIL import Image

MODEL_PATH = "hf://ktrin-u/clearflow-drain-classifier"
TEST_PATH = Path("./test/")

api = NinjaAPI()
model = keras.models.load_model(MODEL_PATH)


@api.post("/predict")
def predict(request, images: list[UploadedFile]):
    results: dict[str, str] = {}
    for image in images:
        img = Image.open(image.file)
        img_array = keras.utils.img_to_array(img)
        img_resized = keras.layers.Resizing(128, 128, pad_to_aspect_ratio=True)(img_array)
        img_resized = tf.expand_dims(img_resized, axis=0)
        prediction = model.predict(img_resized, batch_size=1, verbose="0")  # type: ignore
        results[str(image.name)] = str(prediction[0][0])
    return JsonResponse(results)
