import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to suppresss the tensorflow messages

from pathlib import Path

import keras
import tensorflow as tf
from django.http import JsonResponse
from ninja import NinjaAPI, Swagger
from ninja.files import UploadedFile
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
METHOD1_PATH = "hf://ktrin-u/KEY-cnn"
METHOD2_PATH = "hf://ktrin-u/KEY-ViT"
TEST_PATH = Path("./test/")

api = NinjaAPI(title="Sugarcane Image Classifier", description="This allows you to upload multiple images then use either method to classify them.", docs=Swagger())
method1 = keras.models.load_model(METHOD1_PATH)
method2 = keras.models.load_model(METHOD2_PATH)

@api.post("/predict/method1", tags=["predict"])
def predict_using_CNN(request, images: list[UploadedFile]) -> JsonResponse:
    results: dict[str, str] = {}
    for image in images:
        img = Image.open(image.file)
        img_array = keras.utils.img_to_array(img)
        img_resized = keras.layers.Resizing(96, 96, pad_to_aspect_ratio=True)(img_array)
        img_resized = tf.expand_dims(img_resized, axis=0)
        prediction = method1.predict(img_resized, batch_size=1, verbose="0")  # type: ignore
        prediction_class_index = prediction.argmax()
        results[str(image.name)] = classes[prediction_class_index]
    return JsonResponse(results)

@api.post("/predict/method2", tags=["predict"])
def predict_using_ViT(request, images: list[UploadedFile]) -> JsonResponse:
    results: dict[str, str] = {}
    for image in images:
        img = Image.open(image.file)
        img_array = keras.utils.img_to_array(img)
        img_resized = keras.layers.Resizing(96, 96, pad_to_aspect_ratio=True)(img_array)
        img_resized = tf.expand_dims(img_resized, axis=0)
        prediction = method2.predict(img_resized, batch_size=1, verbose="0")  # type: ignore
        prediction_class_index = prediction.argmax()
        results[str(image.name)] = classes[prediction_class_index]
    return JsonResponse(results)