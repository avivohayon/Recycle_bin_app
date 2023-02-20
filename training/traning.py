import tensorflow as tf
from tensorflow import keras
from keras import layers, models, Sequential
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

IMAGE_SIZE = 256
BATCH_SIZE = 16
RGB_CHANNELS = 3
EPOCHS = 50

data_gen = ImageDataGenerator(rescale=1./255)
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "pic_dataset",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_names = dataset.class_names



def get_partition_tf(ds, tarin_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=100):
    train_size = tarin_split
    val_size = val_split
    test_size = test_split
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=8)

    train_dataset = ds.take(int(len(ds) * train_size))
    test_dataset = ds.skip(len(train_dataset))
    val_dataset = test_dataset.take(int(len(ds) * val_size))
    test_dataset = test_dataset.skip(len(val_dataset))
    return train_dataset, val_dataset, test_dataset


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

if __name__== "__main__":
    # for image_batch, label_batch in dataset.take(1):
    #     # print(image_batch[0].numpy())
    #     plt.imshow(image_batch[0].numpy().astype("uint8"))
    #     plt.show()
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #
    #     break

    # # print("1")
    train_dataset, val_dataset, test_dataset = get_partition_tf(dataset)
    # caching data to improve performance
    train_dataset = train_dataset.cache().shuffle(100).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().shuffle(100).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().shuffle(100).prefetch(buffer_size=tf.data.AUTOTUNE)
    # preprocessing the data. resize, rescale and use data augmentation
    # first, creating the layers for preprocessing
    resize_rescale_layer = tf.keras.Sequential([
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.Rescaling(1.0 / 255)
    ])

    data_augmentation_layer = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2)
    ])

    # second, we build the CNN (convolution neural network) model by connecting:
    # preprocessing layers -> data_augmentation_layer -> Cov layer -> polling layer... -> flatten data
    input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    num_of_classes = 3
    model = models.Sequential([
        resize_rescale_layer,
        data_augmentation_layer,
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_of_classes, activation='softmax')

    ])

    model.build(input_shape=input_shape)
    model.summary()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data=val_dataset)

    print("2")
    # model = models.load_model("../models/1")

    scores = model.evaluate(test_dataset)
    print("first plot")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), acc, label='Training Accuracy')
    plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCHS), loss, label='Training Loss')
    plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    print("4 \n second plot")
    plt.figure(figsize=(15, 15))
    # model = models.load_model("../models/1")
    for images, labels in test_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            predicted_class, confidence = predict(model, images[i].numpy())
            actual_class = class_names[labels[i]]

            plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")

            plt.axis("off")
        plt.show()
    # saving
    print("5")
    model_version = max([int(i) for i in os.listdir("../models")]) + 1
    model.save(f"../models/{model_version}")
    model.save(f"../models/{model_version}/recycle.h5")
#
# IMAGE_SIZE = 256
# BATCH_SIZE = 32
# RGB_CHANNELS = 3
# dataset = tf_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     "pic_dataset",
#     shuffle=True,
#     image_size=(IMAGE_SIZE,IMAGE_SIZE),
#     batch_size=BATCH_SIZE
# )