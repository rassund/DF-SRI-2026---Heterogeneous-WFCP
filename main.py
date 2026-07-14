# %%
import tensorflow as tf
from utils import split_data_homo
import experiments as ex

model_path = "cnn_model_cifar10.keras"

# Load model and test data
model = tf.keras.models.load_model(model_path)
(_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
test_images = test_images.astype("float32") / 255.0
data = list(zip(test_images, test_labels))

# The SNR ranges from -20dB to 20dB
n_0 = [
    100, 31.62, 10, 3.162, 1,
    0.316, 0.1, 0.0316, 0.01
]

num_bins = 60
num_clients = 20
alpha = 0.2

ex.marginal_coverage(model, data, num_clients, split_data_homo, alpha, 10, 1000, n_0[8], num_bins)
ex.histogram_test(model, data, split_data_homo, num_bins, n_0[8], num_clients, alpha)
# %%
