import tensorflow as tf
from actors import Client, Server, Channel

model_path = "cnn_model_cifar10.keras"

# Load model and test data
model = tf.keras.models.load_model(model_path)
(_, _), (images, labels) = tf.keras.cifar10.load_data()
num_of_calib_data = len(images) / 2
num_of_test_data = len(images) - num_of_calib_data
calib_images = images[:num_of_calib_data]
calib_labels = labels[:num_of_calib_data]
test_images = images[num_of_test_data:]
test_labels = images[num_of_test_data:]

# Create clients
clients = []
num_of_clients = 5

for k in range(num_of_clients):
    data_k = calib_data_split[k]
    clients.append(Client(data_k, model, M))

server = Server()
channel = Channel()