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
        Note that bin indices range from 0 to num_bins - 1.
        """
        client = self.clients[0]
        edges = np.linspace(0, 1, self.num_bins + 1)
        eps = 1e-10

        # Special case for the first edge
        self.assertEqual(client.quantize(0.0, edges), 0, "Score 0.0 is not quantized to the correct bin.")

        for i in range(1, self.num_bins):
            # On boundary
            self.assertEqual(client.quantize(edges[i], edges), i - 1, f"Score {edges[i]} is not quantized to bin {i - 1}.")
            # Just above boundary
            self.assertEqual(client.quantize(edges[i] + eps, edges), i, f"Score {edges[i] + eps} is not quantized to bin {i}.")
        
        self.assertRaises(ValueError, client.quantize, -0.1, edges)
        self.assertRaises(ValueError, client.quantize, 1.1, edges)
    
    def test_quantization_interior(self):
        client = self.clients[0]
        edges = np.linspace(0, 1, self.num_bins + 1)
        
        for m in range(self.num_bins):
            left = edges[m]
            right = edges[m + 1]

            interior_point = (left + right) / 2
            self.assertEqual(client.quantize(interior_point, edges), m, f"Interior point {interior_point} is not in bin {m}.")
    
    def test_histogram_sum(self):
        client = self.clients[0]
        self.assertEqual(client.histogram.sum(), 1.0, "The sum of the client's histogram does not equal 1.")