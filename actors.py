import numpy as np
from utils import score_func

class Client:
    def __init__(self, data, model, num_bins):
        """
        model: callable returning p(y|x) \\
        num_bins: M
        """
        (images, labels) = data
        self.softmax_dists = model.predict(images)
        self.noncon_scores = []
        for i in range(len(labels)):
            self.noncon_scores.append(self.noncon_score(self.softmax_dists[i], labels[i]))

        self.M = num_bins
    
    def noncon_score(self, softmax_dist, label):
        """ Compute nonconformity score: s(x,y) = 1 - p(y|x) """
        return score_func(softmax_dist, label)
    
    def quantize(self, s):
        """ Uniform quantization into M bins """
    
    def compute_histogram(self):
        """ Compute local histogram p_k """
    
    def transmit(self, channel):
        """ Transmit the histogram of this client into the channel """
        channel.transmit(self.noncon_scores)


class Server:
    def __init__(self, model):
        self.model = model
    
    def aggregate_data(self, channel):
        """ Aggregate all data currently in the channel """
        self.noncon_scores = channel.flush()
    
    def threshold(self, alpha):
        """ Calculate and return the threshold based on the nonconformity scores and the alpha """
        n = len(self.noncon_scores)
        q_level = int(np.ceil((n + 1) * (1 - alpha)))
        return np.quantile(self.noncon_scores, q_level / n, method = 'higher')
    
    def pred_sets(self, alpha, data):
        """ Compute and return the prediction sets for all images in data """
        pred_sets = []
        (images, labels) = data

        threshold = self.threshold(alpha)
        softmax_dist = self.model.predict(images)
        for image in softmax_dist:
            pred_set = []
            for i in range(len(labels)):
                noncon_score = score_func(image, i)
                if (noncon_score <= threshold):
                    pred_set.append(labels[i])
            pred_sets.append(pred_set)

        return pred_sets


class Channel:
    def apply_noise():
        """ Apply noise to all data currently in the channel """
    
    def transmit(self, data):
        self.data.append(data)
    
    def flush(self):
        data = self.data
        self.data = []
        return data