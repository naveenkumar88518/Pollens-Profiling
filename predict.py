import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore

IMG_SIZE = 128

model = tf.keras.models.load_model('model/pollen_cnn_model.h5')

img_path = 'sample_pollen.jpg'  # change image path
img = cv2.imread(img_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

prediction = model.predict(img)
class_index = np.argmax(prediction)

print("Predicted Class Index:", class_index)
