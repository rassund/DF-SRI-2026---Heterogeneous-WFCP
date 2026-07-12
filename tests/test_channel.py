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