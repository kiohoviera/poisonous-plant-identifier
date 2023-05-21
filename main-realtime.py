import json

import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf

# checkpoint_path = "training_gender/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

model = models.load_model('poisonous-plants.h5')
# train_ds = {'Negative', 'Positive'}

video = cv2.VideoCapture(0)
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
    'Dataset',
    validation_split=0.2,
    subset='training',
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True)

while True:
    _, frame = video.read()

    im = Image.fromarray(frame, 'RGB')

    im = im.resize((img_height, img_width))

    img_array = np.array(im)

    # Expand dimensions to match the 4D Tensor shape.
    img_array = np.expand_dims(img_array, axis=0)

    # Calling the predict function using keras
    prediction = model.predict(img_array)  # [0][0]
    score = tf.nn.softmax(prediction[0])
    if train_ds.class_names[np.argmax(score)] != 'Not':
        print(json.dumps(train_ds.class_names[np.argmax(score)]))

    cv2.imshow("Prediction", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
