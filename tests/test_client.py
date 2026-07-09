import unittest
import numpy as np
import tensorflow as tf
from actors import Client
from utils import split_data_homo

class TestClient(unittest.TestCase):
    num_clients = 10
    min_gain = 1.0
    num_bins = 5
    num_calib_data = 1000
    codebook = np.eye(num_bins)
    model_path = "cnn_model_cifar10.keras"

    @classmethod
    def setUpClass(cls):
        cls.model = tf.keras.models.load_model(cls.model_path)
        (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        test_images = test_images.astype("float32") / 255.0
        cls.data = list(zip(test_images, test_labels))

    def setUp(self):
        gains = np.random.rayleigh(scale=np.sqrt(0.5), size=self.num_clients)

        np.random.shuffle(self.data)
        calib_data = self.data[:self.num_calib_data]
        calib_data_split = split_data_homo(calib_data, self.num_clients)

        self.clients = []
        for i in range(self.num_clients):
            self.clients.append(Client(calib_data_split[i], self.model, self.codebook, gains[i], self.min_gain))
    
    def test_quantization_boundaries(self):
        """
        Test that all quantization boundaries quantize to the correct bin.
        """
        client = self.clients[0]
        edges = np.linspace(0, 1, self.num_bins + 1)
        eps = 1e-10

        # Special case for the first edge
        self.assertEqual(client.quantize(0.0), edges[1], "Score 0.0 is not quantized to the correct bin.")

        for i in range(1, self.num_bins):
            # On boundary
            self.assertEqual(client.quantize(edges[i]), edges[i], f"Score {edges[i]} is not quantized to bin {edges[i]}.")
            # Just above boundary
            self.assertEqual(client.quantize(edges[i] + eps), edges[i + 1], f"Score {edges[i] + eps} is not quantized to bin {edges[i]}.")