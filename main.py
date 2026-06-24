import tensorflow as tf
from actors import Client, Server, Channel
from utils import split_data
import tests as test

model_path = "cnn_model_cifar10.keras"

# Load model and test data
model = tf.keras.models.load_model(model_path)
(_, _), data = tf.keras.cifar10.load_data()
#num_of_calib_data = len(images) / 2
#num_of_test_data = len(images) - num_of_calib_data
#calib_images = images[:num_of_calib_data]
#calib_labels = labels[:num_of_calib_data]
#test_images = images[num_of_test_data:]
#test_labels = images[num_of_test_data:]

test.marginal_coverage(model, data, 1, split_data, 0.1, 100, 1000)