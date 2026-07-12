import unittest
import numpy as np
import tensorflow as tf
from actors import Server

class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.min_gain = 1.0
        cls.num_bins = 5
        cls.num_calib_data = 1000
        cls.codebook = np.eye(cls.num_bins)
        cls.noise_ratio = 0.5
        cls.model_path = "cnn_model_cifar10.keras"

        cls.model = tf.keras.models.load_model(cls.model_path)
        (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        test_images = test_images.astype("float32") / 255.0
        cls.data = list(zip(test_images, test_labels))

    def setUp(self):
        self.server = Server(self.model, self.codebook, self.num_calib_data, self.min_gain, self.noise_ratio)
    
    def test_tbma_decode_matches_equation(self):
        self.server.num_active_clients = 5
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = self.server.tbma_decode(data)

        expected = (self.num_calib_data / (np.sqrt(self.num_bins * 1.0) * self.min_gain * 5 * (self.num_calib_data + 1))) * data
        expected[-1] += 1 / (self.num_calib_data + 1)

        np.testing.assert_allclose(result, expected, err_msg="The TBMA decoded data does not match the equation.")
    
    def test_tbma_decode_matched_filtering(self):
        self.server.num_active_clients = 2
        data = np.array([3., 5., 7., 9., 11.])
        result = self.server.tbma_decode(data)

        expected_w = self.codebook.T @ data
        expected = (self.num_calib_data / (np.sqrt(self.num_bins * 1.0) * self.min_gain * 2 * (self.num_calib_data + 1))) * expected_w
        expected[-1] += 1 / (self.num_calib_data + 1)

        np.testing.assert_allclose(result, expected, err_msg="The TBMA decode process does not use the codebook.")
    
    def test_aggregate_data_updates_server_state(self):
        channel = DummyChannel()
        self.server.aggregate_data(channel)
        self.assertEqual(self.server.num_active_clients, 7, "The server does not receive the correct number of clients from the channel.")
        self.assertIsNotNone(self.server.histogram, "The server does not compute a histogram after receiving data from the channel.")


class DummyChannel():
    def receive(self):
        return np.array([1., 2., 3., 4., 5.]), 7