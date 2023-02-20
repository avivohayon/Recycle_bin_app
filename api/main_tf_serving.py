import numpy as np
import  tensorflow as tf
from keras import layers, Sequential
from io import BytesIO
from PIL import Image
import uvicorn
from fastapi import FastAPI, UploadFile, File
import uvicorn
import  requests

app = FastAPI()
MODEL = tf.keras.models.load_model("../models/2")
CLASS_NAMES = ["blue_bin", "orange_bin", "purple_bun"]
IMAGE_SIZE = 256

end_point = "http://localhost:8501/v1/models/recycle_model/versions/2:predict"

@app.get("/ping")
async  def ping():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    img = read_file_as_img(await file.read())
    resize_img = resize_img_input(img).numpy()
    # the models get a batch of images this way we
    # make sure a single image will be sent as batch (list in a list) as well
    # predictions = MODEL.predict(resize_img)
    print("1")
    json_data = {"instances": resize_img.tolist()}
    print("2")
    response = requests.post(end_point, json=json_data)
    print("3")
    predict = np.array((response.json()["predictions"][0]))
    predict_class = CLASS_NAMES[np.argmax(predict)]
    confidence = np.max(predict)
    # predict_class = CLASS_NAMES[np.argmax(predict)]
    # confidence = np.max(predict)
    #
    return {"file_name": file.filename,
            "predicted_class": predict_class,
            "confidence": float(confidence)
            }


def resize_img_input(image: np.array):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return tf.expand_dims(image, 0)
def read_file_as_img(bytes_data) -> np.ndarray:
    """

    :param bytes_data: the file as bytes
    :return: the images itself as np array
    """
    return np.array(Image.open(BytesIO(bytes_data)))

if __name__== "__main__":
    uvicorn.run(app, host="localhost", port=8000)



#todo
# 1 - build the model code from conda into here, then save the build model
# 2 - check again how good is it (the oe from anaconda predict "blue" all the time"
# 3 - find a way to create more pics data
#
# IMAGE_SIZE = 256
# BATCH_SIZE = 16
# RGB_CHANNELS = 3
# EPOCHS = 50
#
# dataset = tf_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     "pic_dataset",
#     shuffle=True,
#     image_size=(IMAGE_SIZE,IMAGE_SIZE),
#     batch_size=BATCH_SIZE
# )
#
#
# resize_rescale_layer = tf.keras.Sequential([
#     layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
#     layers.experimental.preprocessing.Rescaling(1.0/255)
# ])
#
#
#
# data_augmentation_layer = tf.keras.Sequential([
#     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
#     layers.experimental.preprocessing.RandomRotation(0.2)
# ])
#
# input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
# num_of_classes = 3
# model = models.Sequential([
#     resize_rescale_layer,
#     data_augmentation_layer,
#     layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(num_of_classes, activation='softmax')
#
# ])