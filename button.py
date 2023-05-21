import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import argparse
import json
import os
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_UP)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

img_height = 224
img_width = 224
tf.keras.utils.disable_interactive_logging()

train_ds = tf.keras.utils.image_dataset_from_directory(
    'Dataset',
    validation_split=0.2,
    subset='training',
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True)

def capture():
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)

    result, image = cam.read()
    if result:
        cv2.imshow("Detection", image)
        cv2.imwrite("Detection.png", image)
        cv2.waitKey(0)
        cv2.destroyWindow("Detection")
    else:
        print("No image detected. Please! try again")

def identify():
    capture()
    model = tf.keras.models.load_model('poisonous-plants.h5')

    detection_path = 'Tests/IMG20230507164521.jpg'
    img = tf.keras.utils.load_img(
        detection_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    plt.figure(figsize=(2, 2))
    plt.imshow((img_array[0].numpy()).astype('uint8'))
    plt.title("{}:{:.2f}".format(train_ds.class_names[np.argmax(score)], 100 * np.max(score)))

    plt.axis('off')
    print(json.dumps(train_ds.class_names[np.argmax(score)]))
    plt.show()

while True:
    input_state = GPIO.input(21)
    if input_state == False:
        print('Button Pressed')
        identify()
        