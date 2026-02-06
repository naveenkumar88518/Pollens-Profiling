import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

IMG_SIZE = 128

model = tf.keras.models.load_model('model/pollen_cnn_model.h5')

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical'
)

loss, accuracy = model.evaluate(test_data)
print(f"🎯 Test Accuracy: {accuracy * 100:.2f}%")
