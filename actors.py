import numpy as np
from utils import score_func, cifar10_labels

class Client:
    def __init__(self, data, model, num_bins, codebook):
        """
        model: callable returning p(y|x) \\
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
        """ Encodes the local histogram into a TBMA signal """
        return self.codebook @ np.sqrt(histogram)
    
    def transmit(self, channel):
        """ Convert this clients histogram into a TBMA signal and transmit it into the channel """
        channel.transmit(self.tbma_encode(self.histogram))
    
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
        """ Estimate a histogram from the TBMA signal """
        bin_energy = (self.codebook.T @ data) ** 2
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
    def __init__(self):
        self.data = []
    
    def apply_noise(data):
        """ Apply noise to all data currently in the channel """
        return data
    
    def transmit(self, data):
        self.data.append(data)
    
    def receive(self):
        aggregate_data = np.sum(self.data, axis=0)
        #h = 1.0 # Fading coefficient
        #for x in self.data:
        #    received += h * x
        
        #received = self.apply_noise(received)

        self.data = []

        return aggregate_data