import numpy as np
from utils import score_func, cifar10_labels, Modes

class Client:
    def __init__(self, data, model, codebook, gain, min_gain, N_max=None):
        images = np.array([d[0] for d in data])
        labels = np.array([d[1] for d in data])

        softmax_dists = model.predict(images, verbose=False)

        self.noncon_scores = np.array([
            self.noncon_score(softmax_dists[i], labels[i])
            for i in range(len(labels))
        ])

        self.M = codebook.shape[0]
        self.codebook = codebook
        self.P = 1.0 # Fixed power constraint. Different SNR can be tested by varying noise ratio N_0.
        self.h = gain
        self.h_min = min_gain
        self.N_d = len(self.noncon_scores)
        self.N_max = N_max # If N_max == none then homogeneous distribution is assumed.

        bins = np.linspace(0, 1, self.M + 1)
        self.quantized_scores = np.array([
            self.quantize(s[0], bins)
            for s in self.noncon_scores
        ])

        self.histogram = self.compute_histogram(self.quantized_scores)
    
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
        gamma = np.sqrt(self.M * self.P) * self.h_min # eq. 35
        gamma /= self.N_d if self.N_max == None else self.N_max

        if self.h**2 < self.h_min**2: # eq. 31
            gamma_k = 0
        else:
            gamma_k = gamma / self.h

        return gamma_k * self.N_d * (c @ p) # eq. 28
    
    def transmit(self, channel):
        """ Convert this clients histogram into a TBMA signal and transmit it into the channel """
        channel.transmit(self.tbma_encode(self.histogram), self.h, self.N_d)


class Server:
    def __init__(self, model, codebook, min_gain, noise_ratio, mode=Modes.HOMO, N_max = None):
        self.model = model
        self.M = codebook.shape[0]
        self.codebook = codebook
        self.histogram = None
        self.num_active_clients = None
        self.num_calib_data = None
        self.min_gain = min_gain
        self.P = 1.0
        self.snr = self.P / noise_ratio
        self.mode = mode
        self.N_max = N_max
    
    def aggregate_data(self, channel):
        """ Aggregate all data currently in the channel """
        data, self.num_active_clients, self.num_calib_data = channel.receive()
        self.histogram = self.tbma_decode(data)
    
    def tbma_decode(self, data):
        """
        Estimate a histogram from the TBMA signal.
        Based on equations 29, 30, and 37 in Federated Inference With Reliable Uncertainty Quantificiation Over
        Wireless Channels via Conformal Prediction (2024) by Zhu et al. 
        """
        k, n_a, p, m, h_min = self.num_active_clients, self.num_calib_data, self.P, self.M, self.min_gain
        w = self.codebook.T @ data # eq. 30
        r = (n_a / (np.sqrt(m * p) * h_min * k * (n_a + 1))) * w # eq. 37
        r[-1] += 1 / (n_a + 1)
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
        m, h_min, snr, k_a, n_a = self.M, self.min_gain, self.snr, self.num_active_clients, self.num_calib_data
        
        if self.mode == Modes.HOMO:
            n_d = n_a / k_a
            n = n_d
        elif self.mode == Modes.HETERO:
            n = self.N_max
        else:
            raise ValueError(f"Invalid mode: {self.mode.name}.")
        
        sigma2 = n**2 / (m * h_min**2 * snr * (n_a + 1)**2) # eq. 38
        alpha_c = alpha - sigma2 * m / (4 * alpha) # eq. 42

        if self.mode == Modes.HOMO:
            target = 1 - alpha_c
        elif self.mode == Modes.HETERO:
            target = np.ceil((1 - alpha_c) * (n_a + k_a)) / n_a
        
        cdf = np.cumsum(self.histogram) # eq. 39
        idx = np.searchsorted(cdf, target) # eq. 40
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
    def __init__(self, noise_ratio, noise_type="Gaussian"):
        self.data = []
        self.n_0 = noise_ratio
        self.noise_type = noise_type
    
    def apply_noise(self, data):
        """ Apply noise to all data currently in the channel """
        if self.noise_type == "Gaussian":
            noise = np.random.normal(0, np.sqrt(self.n_0), size = data.shape)
            return data + noise
        
        return data
    
    def transmit(self, data, h, n_d):
        if np.count_nonzero(data) != 0:
            self.data.append({
                "signal": data,
                "fading": h,
                "num_calib": n_d
            })
    
    def receive(self):
        if not self.data:
            raise ValueError("Cannot receive from an empty channel.")

        aggregate_data = np.zeros_like(self.data[0]["signal"])
        n_a = 0

        for packet in self.data:
            h = packet["fading"]
            x = packet["signal"]
            aggregate_data += h * x
            n_a += packet["num_calib"]
        
        aggregate_data = self.apply_noise(aggregate_data)

        num_clients = len(self.data)
        self.data = []

        return aggregate_data, num_clients, n_a