import unittest
import numpy as np
import tensorflow as tf
from actors import Client

class TestClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.min_gain = 1.0
        cls.num_bins = 5
        cls.num_calib_data = 1000
        cls.codebook = np.eye(cls.num_bins)
        cls.model_path = "cnn_model_cifar10.keras"

        cls.model = tf.keras.models.load_model(cls.model_path)
        (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        test_images = test_images.astype("float32") / 255.0
        cls.data = list(zip(test_images, test_labels))

    def setUp(self):
        gain = np.random.rayleigh(scale=np.sqrt(0.5))

        np.random.shuffle(self.data)
        calib_data = self.data[:self.num_calib_data]

        self.client = Client(calib_data, self.model, self.codebook, gain, self.min_gain)
    
    def test_quantization_boundaries(self):
        """
        Test that all quantization boundaries quantize to the correct bin.
        Note that bin indices range from 0 to num_bins - 1.
        """
        edges = np.linspace(0, 1, self.num_bins + 1)
        eps = 1e-10

        # Special case for the first edge
        self.assertEqual(self.client.quantize(0.0, edges), 0, "Score 0.0 is not quantized to the correct bin.")

        for i in range(1, self.num_bins):
            # On boundary
            self.assertEqual(self.client.quantize(edges[i], edges), i - 1, f"Score {edges[i]} is not quantized to bin {i - 1}.")
            # Just above boundary
            self.assertEqual(self.client.quantize(edges[i] + eps, edges), i, f"Score {edges[i] + eps} is not quantized to bin {i}.")
        
        self.assertRaises(ValueError, self.client.quantize, -0.1, edges)
        self.assertRaises(ValueError, self.client.quantize, 1.1, edges)
    
    def test_quantization_interior(self):
        edges = np.linspace(0, 1, self.num_bins + 1)
        
        for m in range(self.num_bins):
            left = edges[m]
            right = edges[m + 1]

            interior_point = (left + right) / 2
            self.assertEqual(self.client.quantize(interior_point, edges), m, f"Interior point {interior_point} is not in bin {m}.")
    
    def test_histogram_sum(self):
        self.assertEqual(self.client.histogram.sum(), 1.0, "The sum of the client's histogram does not equal 1.")
    
    def test_client_histogram_against_manual(self):
        quantized_scores = self.client.quantized_scores

        hist = np.zeros(self.num_bins)
        for m in range(self.num_bins):
            count = 0
            for idx in quantized_scores:
                if idx == m:
                    count += 1
            hist[m] = count / len(quantized_scores)
        
        np.testing.assert_allclose(hist, self.client.histogram, err_msg="The client's histogram does not match the manually computed histogram.")
    
    def test_tbma_encode_below_h_min(self):
        self.client.h = self.min_gain / 2
        hist = np.array([0.2, 0.3, 0.1, 0.25, 0.15])
        result = self.client.tbma_encode(hist)
        expected = np.zeros(5)
        np.testing.assert_allclose(result, expected, err_msg="The client transmits their histogram despite their gain being below the threshold.")
    
    def test_tbma_encode_above_h_min(self):
        self.client.h_min = 1.0
        self.client.h = 4.0
        hist = np.array([0.2, 0.3, 0.1, 0.25, 0.15])

        gamma = np.sqrt(self.client.M * self.client.P) * self.client.h_min
        gamma_k = gamma / self.client.h
        expected = gamma_k * (self.client.codebook @ hist)

        result = self.client.tbma_encode(hist)

        np.testing.assert_allclose(result, expected, err_msg="The client transmits the wrong TBMA encoded signal.")