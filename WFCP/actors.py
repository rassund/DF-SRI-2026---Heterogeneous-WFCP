class Client:
    def __init__(self, data, model, num_bins):
        """
        data: list of (x, y) pairs
        model: callable returning p(y|x)
        num_bins: M
        """
        self.data = data
        self.model = model
        self.M = num_bins
    
    def nonconformity_score(self, x, y):
        """ Compute nonconformity score """
    
    def quantize(self, s):
        """ Uniform quantization into M bins """
    
    def compute_histogram(self):
        """ Compute local histogram p_k """


class Server:


class Channel:
    