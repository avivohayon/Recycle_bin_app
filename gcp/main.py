# -------- This file have a code which we deploy into the Google Could ---------

from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = "recycle-bin-models"
CLASS_NAMES = ["blue_bin", "orange_bin", "purple_bun"]


model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """
    blob is a collection of binary data stored as a single entity in a database
    his function will run on a different server in Google cloud,
    that server will download the model from the bucket. so when it downloads it locally on that  server on cloud
    :param bucket_name: the name of the blob
    :params source_blob_name: the blob on the bucket
    :param destination_file_name: the destination file path to store the model
    """
    storage_client = storage.Client()  # download our saved object (the saved trained .h5 file model)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def predict(request):
    """
    deploy this predict function into the cloud
    :param request:
    :return:
    """
    global  model
    if model is None:
    # this cloud func will be executed few multiple times so we need to execute "download_blob" only on the first call
        download_blob(BUCKET_NAME, "models/recycle1.h5", "/tmp/recycle1.h5")

    model = tf.keras.models.load_model("/tmp/recycle1.h5")

    img = request.files["file"]
    img = np.array(Image.open(img).convert("RGB").resize((256, 256)))
    img = img/255

    # the model prediction expects a batch of images so we pad the image
    img_arr = tf.expand_dims(img, 0)
    predictions = model.predict(img_arr)
    print(predictions)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return {"predicted class": predicted_class, "confidence": confidence}

