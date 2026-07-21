import unittest
import numpy as np
from actors import Channel

class TestChannel(unittest.TestCase):
    def setUp(self):
        self.channel = Channel(0.5, "Gaussian")
    
    def test_apply_noise_preserves_shape(self):
        data = np.ones((5, 10))
        noise = self.channel.apply_noise(data.copy())
        self.assertEqual(noise.shape, data.shape, "The channel does not preserve the shape of the signal when applying noise.")
    
    def test_apply_noise_changes_data(self):
        data = np.zeros(1000)
        noise = self.channel.apply_noise(data.copy())
        assert not np.array_equal(noise, data), "The signal does not change after applying noise."
    
    def test_apply_noise_zero_mean(self):
        data = np.zeros(500000)
        noisy = self.channel.apply_noise(data.copy())
        noise = noisy - data
        self.assertLess(np.mean(noise), 0.01, "The mean of the noise is not approximately zero.")
    
    def test_apply_noise_correct_variance(self):
        data = np.zeros(500000)
        noisy = self.channel.apply_noise(data.copy())
        noise = noisy - data
        np.testing.assert_allclose(np.var(noise), self.channel.n_0, rtol=0.05, err_msg="The noise does not have the correct variance N_0.")
    
    def test_apply_noise_zero_noise_power(self):
        self.channel.n_0 = 0.0
        data = np.random.rand(100)
        noisy = self.channel.apply_noise(data.copy())
        np.testing.assert_array_equal(noisy, data, err_msg="The channel applies noise even though noise power is zero.")
    
    def test_apply_noise_unknown_type(self):
        self.channel.noise_type = "None"
        data = np.random.rand(100)
        out = self.channel.apply_noise(data.copy())
        np.testing.assert_array_equal(out, data, err_msg="Noise is applied even though noise type is set to none.")
    
    def test_receive_aggregates_signal(self):
        self.channel.noise_type = "None"
        self.channel.data = [
            {"signal": np.array([1., 2., 3.]), "fading": 2.0, "num_calib": 15},
            {"signal": np.array([4., 5., 6.]), "fading": 0.5, "num_calib": 10},
        ]

        received, num_clients, num_calib_data = self.channel.receive()

        expected = (
            2.0 * np.array([1., 2., 3.]) +
            0.5 * np.array([4., 5., 6.])
        )

        np.testing.assert_array_equal(received, expected, "The channel does not aggregate the signals correctly.")
        self.assertEqual(num_clients, 2, "The channel does not return the correct number of clients.")
        self.assertEqual(num_calib_data, 25, "The channel does not return the correct number of calibration data.")
    
    def test_receive_clears_channel(self):
        self.channel.data = [
            {"signal": np.ones(3), "fading": 1, "num_calib": 1}
        ]
        self.channel.receive()
        self.assertEqual(self.channel.data, [], "The channel does not clear its buffer after sending data through.")
    
    def test_receive_clients_with_no_data(self):
        data, h, n = np.array([0, 0, 0]), 1.0, 1
        self.channel.transmit(data, h, n)
        self.assertRaises(ValueError, self.channel.receive)