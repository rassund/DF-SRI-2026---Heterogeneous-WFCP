import numpy as np

class Client:
    def __init__(self, data, model, num_bins):
        """
        data: list of (x, y) pairs \\
        model: callable returning p(y|x) \\
        num_bins: M
        """
        self.data = data
        self.model = model
        self.M = num_bins
    
    def noncon_score(self, x, y):
        """ Compute nonconformity score: s(x,y) = 1 - p(y|x) """
        softmax = self.model.predict(x)
        return 1.0 - softmax[y]
    
    def quantize(self, s):
        """ Uniform quantization into M bins """
    
    def compute_histogram(self):
        """ Compute local histogram p_k """
    
    def transmit(self, channel):
        """ Transmit the histogram of this client into the channel """


class Server:
    def aggregate_data(self, channel):
        """ Aggregate all data currently in the channel """
    
    def threshold(self, alpha):
        """ Calculate and return the threshold based on the nonconformity scores and the alpha """
        n = len(self.noncon_scores)
        q_level = int(np.ceil((n + 1) * (1 - alpha)))
        return np.quantile(self.noncon_scores, q_level / n, method = 'higher')
    
    def predict_set(alpha):
        """ Compute and return the prediction set """
        s = threshold(alpha)


class Channel:
    def apply_noise():
        """ Apply noise to all data currently in the channel """