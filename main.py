# %%
import tensorflow as tf
from utils import split_data_homo
import tests as test

model_path = "cnn_model_cifar10.keras"

# Load model and test data
model = tf.keras.models.load_model(model_path)
(_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
test_images = test_images.astype("float32") / 255.0
data = list(zip(test_images, test_labels))

#test.marginal_coverage(model, data, 10, split_data_homo, 0.1, 10, 1000)
test.histogram_test(model, data, split_data_homo)
# %%
