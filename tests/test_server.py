import unittest
import numpy as np
import tensorflow as tf
from actors import Server
from unittest.mock import Mock
from utils import cifar10_labels

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
    
    def test_threshold_returns_valid_quantization_level(self):
        self.server.num_active_clients = 4
        self.server.histogram = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        threshold = self.server.threshold(0.1)
        valid_levels = np.linspace(0, 1, 5)
        self.assertIn(threshold, valid_levels, "The server does not return a valid threshold.")
    
    def test_threshold_uniform_histogram(self):
        self.server.num_calib_data = 20
        self.server.min_gain = 1000
        self.server.snr = 1.0
        self.server.num_active_clients = 4
        self.server.histogram = np.array([0.15, 0.25, 0.25, 0.20, 0.15])
        threshold = self.server.threshold(0.2)
        self.assertEqual(threshold, 0.75, "The server chooses the wrong bin from the histogram as threshold.")
    
    def test_threshold_monotonicity(self):
        self.server.num_active_clients = 4
        self.server.histogram = np.array([0.15, 0.25, 0.25, 0.20, 0.15])
        t90 = self.server.threshold(0.10)
        t95 = self.server.threshold(0.05)
        self.assertGreaterEqual(t95, t90, "The server returns a larger threshold for a larger alpha.")
    
    def test_threshold_never_outside_histogram(self):
        self.server.num_calib_data = 20
        self.server.min_gain = 0.01
        self.server.snr = 1 / 100
        self.server.num_active_clients = 1
        self.server.histogram = np.array([0.15, 0.25, 0.25, 0.15, 0.20])
        threshold = self.server.threshold(0.01)
        self.assertEqual(threshold, 1.0, "The server returns a threshold outside the final bin of the histogram.")
    
    def test_empty_prediction_set(self):
        self.server.threshold = Mock(return_value = -1)
        pred = self.server.pred_sets(0.1, np.zeros((1, 32, 32, 3)))
        self.assertEqual(pred, [[]], "The server returns a non-empty prediction set even though threshold < 0.")
    
    def test_all_labels_in_prediction_set(self):
        self.server.threshold = Mock(return_value = 1)
        pred = self.server.pred_sets(0.1, np.zeros((1, 32, 32, 3)))
        self.assertEqual(pred, [cifar10_labels], "The prediciton set does not contain all labels even though threshold == 1.")
    
    def test_one_label_in_prediction_set(self):
        self.server.model = Mock()
        self.server.model.predict.return_value = np.array([[
            0.82, 0.02, 0.02, 0.02,
            0.02, 0.02, 0.02, 0.02,
            0.02, 0.02]])
        self.server.threshold = Mock(return_value = 0.2)
        pred = self.server.pred_sets(0.1, np.zeros((1, 32, 32, 3)))
        self.assertEqual(pred, [[cifar10_labels[0]]], "The prediction set does not contain the correct label.")


class DummyChannel():
    def receive(self):
        return np.array([1., 2., 3., 4., 5.]), 7