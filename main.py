import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import argparse
import json
import os
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
    
    # reading the input using the camera
    result, image = cam.read()
    
    # If image will detected without any error, 
    # show result
    if result:
    
        # showing result, it take frame name and image 
        # output
        cv2.imshow("Detection", image)
    
        # saving image in local storage
        cv2.imwrite("Detection.png", image)
    
        # If keyboard interrupt occurs, destroy image 
        # window
        cv2.waitKey(0)
        cv2.destroyWindow("Detection")
    
    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")
def identify():
    capture()
    model = tf.keras.models.load_model('poisonous-plants.h5')

    detection_path = 'Detection.png'
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

identify()
