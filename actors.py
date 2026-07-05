import numpy as np
from utils import score_func, cifar10_labels

class Client:
    def __init__(self, data, model, num_bins, codebook, gain, min_gain):
        """
        model: the predictive model p(y|x) \\
        num_bins: M
        """
        images = np.array([d[0] for d in data])
        labels = np.array([d[1] for d in data])

        self.softmax_dists = model.predict(images, verbose=False)

        self.noncon_scores = np.array([
            self.noncon_score(self.softmax_dists[i], labels[i])
            for i in range(len(labels))
        ])

        self.M = num_bins
        self.codebook = codebook
        self.P = 1.0 # Fixed power constraint. Different SNR can be tested by varying noise ratio N_0.
        self.h = gain
        self.h_min = min_gain

        self.quantized_scores = np.array([
            self.quantize(s[0])
            for s in self.noncon_scores
        ])

        self.compute_histogram()
    
    def noncon_score(self, softmax_dist, label):
        """ Compute nonconformity score: s(x,y) = 1 - p(y|x) """
        return score_func(softmax_dist, label)
    
    def quantize(self, score):
        """ Uniform quantization into M bins """
        bin_width = 1.0 / self.M
        bin_idx = int(np.floor(score / bin_width))
        return min(bin_idx, self.M - 1)
    
    def compute_histogram(self):
        """ Compute local histogram p_k """
        counts = np.bincount(self.quantized_scores, minlength=self.M)
        self.histogram = counts / len(self.quantized_scores)
    
    def tbma_encode(self, histogram):
        """
        Encodes the local histogram into a TBMA signal
        Based on equations 28, 31, and 35 in Federated Inference With Reliable Uncertainty Quantificiation Over Wireless Channels
        via Conformal Prediction (2024) by Zhu et al.  
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
    def __init__(self, model, num_bins, codebook):
        self.model = model
        self.M = num_bins
        self.codebook = codebook
        self.histogram = None
    
    def aggregate_data(self, channel):
        """ Aggregate all data currently in the channel """
        data = channel.receive()
        self.histogram = self.tbma_decode(data)
    
    def tbma_decode(self, data):
        """
        Estimate a histogram from the TBMA signal.
        Based on equations 29 and 30 in Federated Inference With Reliable Uncertainty Quantificiation Over Wireless Channels
        via Conformal Prediction (2024) by Zhu et al. 
        """
        w = self.codebook.T @ data # eq. 30
        # r preprocessing (eq. 37)
        return bin_energy / np.sum(bin_energy)
    
    def threshold(self, alpha):
        """ Calculate and return the threshold based on the nonconformity scores and the alpha """
        #n = len(self.noncon_scores) CENTRALIZED THRESHOLD COMPUTATION
        #q_level = int(np.ceil((n + 1) * (1 - alpha)))
        #return np.quantile(self.noncon_scores, q_level / n, method = 'higher')

        cumulative = np.cumsum(self.histogram)

        bin_idx = np.searchsorted(cumulative, 1-alpha)

        lower_mass = (cumulative[bin_idx-1] if bin_idx > 0 else 0)
        bin_fraction = ((1-alpha - lower_mass) / self.histogram[bin_idx])

        bin_width = 1.0 / self.M

        threshold = (bin_idx + bin_fraction) * bin_width
        return min(threshold, 1.0)
    
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
    
    def get_histogram(self):
        return self.histogram


class Channel:
    def __init__(self, snr=20):
        """
        snr: signal-to-noise ratio
        """
        self.data = []
        self.snr = snr
    
    def apply_noise(self, data, noise_type): # REVISIT THIS
        """ Apply noise to all data currently in the channel """
        if noise_type == "Gaussian":
            signal_power = np.mean(data ** 2)
            noise_power = signal_power / self.snr
            noise = np.random.normal(0, np.sqrt(noise_power), size = data.shape)
            data += noise
        
        return data
    
    def transmit(self, data, h):
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

        self.data = []

        return aggregate_data