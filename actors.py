import numpy as np
from utils import score_func, cifar10_labels

class Client:
    def __init__(self, data, model, codebook, gain, min_gain):
        images = np.array([d[0] for d in data])
        labels = np.array([d[1] for d in data])

        self.softmax_dists = model.predict(images, verbose=False)

        self.noncon_scores = np.array([
            self.noncon_score(self.softmax_dists[i], labels[i])
            for i in range(len(labels))
        ])

        self.M = codebook.shape[0]
        self.codebook = codebook
        self.P = 1.0 # Fixed power constraint. Different SNR can be tested by varying noise ratio N_0.
        self.h = gain
        self.h_min = min_gain

        bins = np.linspace(0, 1, self.M + 1)
        quantized_scores = np.array([
            self.quantize(s[0], bins)
            for s in self.noncon_scores
        ])

        self.histogram = self.compute_histogram(quantized_scores)
    
    def noncon_score(self, softmax_dist, label):
        """ Compute nonconformity score: s(x,y) = 1 - p(y|x) """
        return score_func(softmax_dist, label)
    
    def quantize(self, score, bins):
        """ Uniform quantization into M bins """
        if score < 0 or score > 1:
            raise ValueError("Nonconformity scores must lie in [0, 1].")
        return np.digitize(score, bins[1:], right=True)
    
    def compute_histogram(self, quantized_scores):
        """ Compute local histogram p_k """
        counts = np.bincount(quantized_scores, minlength=self.M)
        return counts / len(quantized_scores)
    
    def tbma_encode(self, histogram):
        """
        Encodes the local histogram into a TBMA signal
        Based on equations 28, 31, and 35 in Federated Inference With Reliable Uncertainty Quantificiation Over
        Wireless Channels via Conformal Prediction (2024) by Zhu et al.  
        """
        c, p = self.codebook, histogram
        gamma = np.sqrt(self.M * self.P) * self.h_min

        if self.h**2 < self.h_min**2:
            gamma_k = 0
        else:
            gamma_k = gamma / self.h

        return gamma_k * (c @ p)
    
    def transmit(self, channel):
        """ Convert this clients histogram into a TBMA signal and transmit it into the channel """
        channel.transmit(self.tbma_encode(self.histogram), self.h)
    
    def get_histogram(self):
        return self.histogram


class Server:
    def __init__(self, model, codebook, num_calib_data, min_gain, noise_ratio):
        self.model = model
        self.M = codebook.shape[0]
        self.codebook = codebook
        self.histogram = None
        self.num_active_clients = None
        self.num_calib_data = num_calib_data
        self.min_gain = min_gain
        self.P = 1.0
        self.snr = self.P / noise_ratio

    
    def aggregate_data(self, channel):
        """ Aggregate all data currently in the channel """
        data, self.num_active_clients = channel.receive()
        self.histogram = self.tbma_decode(data)
    
    def tbma_decode(self, data):
        """
        Estimate a histogram from the TBMA signal.
        Based on equations 29, 30, and 37 in Federated Inference With Reliable Uncertainty Quantificiation Over
        Wireless Channels via Conformal Prediction (2024) by Zhu et al. 
        """
        k, n, p, m, h_min = self.num_active_clients, self.num_calib_data, self.P, self.M, self.min_gain
        w = self.codebook.T @ data # eq. 30
        r = (n / (np.sqrt(m * p) * h_min * k * (n + 1))) * w # eq. 37
        r[-1] += 1 / (n + 1)
        return r
    
    def threshold(self, alpha):
        """
        Compute and return the threshold.
        Based on equations 38-42 in Federated Inference With Reliable Uncertainty Quantificiation Over
        Wireless Channels via Conformal Prediction (2024) by Zhu et al.
        """
        #n = len(self.noncon_scores) CENTRALIZED THRESHOLD COMPUTATION
        #q_level = int(np.ceil((n + 1) * (1 - alpha)))
        #return np.quantile(self.noncon_scores, q_level / n, method = 'higher')
        m, h_min, snr, k, n_a = self.M, self.min_gain, self.snr, self.num_active_clients, self.num_calib_data
        n_d = n_a / k
        sigma = n_d**2 / (m * h_min**2 * snr * (n_a + 1)) # eq. 38
        alpha_c = alpha - sigma * self.M / (4 * alpha) # eq. 42

        cdf = np.cumsum(self.histogram) # eq. 39
        idx = np.searchsorted(cdf, 1 - alpha_c) # eq. 40
        idx = min(idx, m - 1)
        
        s = np.linspace(0, 1, m) # eq. 41
        return s[idx]
    
    def pred_sets(self, alpha, images):
        """ Compute and return the prediction sets for all images """
        pred_sets = []

        # quantile correction to determine alpha_c
        threshold = self.threshold(alpha)

        softmax_dist = self.model.predict(images, verbose=False)

        for probs in softmax_dist:
            pred_set = []
            for i in range(len(cifar10_labels)):
                score = score_func(probs, i)
                if (score <= threshold):
                    pred_set.append(cifar10_labels[i])
            pred_sets.append(pred_set)

        return pred_sets


class Channel:
    def __init__(self, noise_ratio):
        self.data = []
        self.n_0 = noise_ratio
    
    def apply_noise(self, data, noise_type):
        """ Apply noise to all data currently in the channel """
        if noise_type == "Gaussian":
            noise = np.random.normal(0, np.sqrt(self.n_0), size = data.shape)
            data += noise
        
        return data
    
    def transmit(self, data, h):
        if h > 0:
            self.data.append({
                "signal": data,
                "fading": h
            })
    
    def receive(self):
        aggregate_data = np.zeros_like(self.data[0]["signal"])

        for packet in self.data:
            h = packet["fading"]
            x = packet["signal"]
            aggregate_data += h * x
        
        aggregate_data = self.apply_noise(aggregate_data, "Gaussian")

        num_clients = len(self.data)
        self.data = []

        return aggregate_data, num_clients