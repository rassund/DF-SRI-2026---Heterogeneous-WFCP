import numpy as np
from abc import ABC, abstractmethod
from actors import Client, Channel, Server
from utils import Modes

class MethodInterface(ABC):
    @abstractmethod
    def calibrate(self, calib_data_split) -> None:
        """
        Calibrates the method based on the calibration data.

        Parameters
        ----------
        calib_data_split : list of list of (image, label)
            The calibration dataset split over the number of clients.
            Client i finds their calibration data at index i.
        """

    @abstractmethod
    def predict(self, alpha, images) -> list:
        """
        Returns the prediction sets of all images.
        
        Parameters
        ----------
        alpha : float
            1-alpha is the target coverage.
        images : list
            The images to create prediction sets for.
        
        Returns
        -------
        prediction_sets : list of lists
            List containing the prediction set for each image.
            The prediction set for image i is found at index i.
        """


class CentralCP(MethodInterface):
    def __init__(self, model):
        self.server = Server(
            model, codebook=np.eye(1), min_gain=1.0,
            noise_ratio=1.0, mode=Modes.CENTRAL
        )

    def calibrate(self, calib_data_split):
        self.server.calibrate(calib_data_split)

    def predict(self, alpha, images):
        return self.server.pred_sets(alpha, images)


class WFCP(MethodInterface):
    def __init__(self, model, num_bins, num_clients, noise_ratio, gains, min_gain=1.0):
        self.clients = []
        for i in range(num_clients):
             self.clients.append(Client(
                  model, codebook=np.eye(num_bins), gain=gains[i],
                  min_gain=min_gain
             ))
        self.channel = Channel(noise_ratio)
        
        self.server = Server(
            model, codebook=np.eye(num_bins), min_gain=min_gain,
            noise_ratio=noise_ratio, mode=Modes.HOMO
        )
    
    def calibrate(self, calib_data_split):
        for i in range(self.clients):
            self.clients[i].calibrate(calib_data_split[i])

    def predict(self, alpha, images):
        for client in self.clients:
            client.transmit(self.channel)

        self.server.aggregate_data(self.channel)
        return self.server.pred_sets(alpha, images)


class HETERO_WFCP(MethodInterface):
    def __init__(self, model, num_bins, num_clients, noise_ratio, gains, min_gain=1.0):
        self.clients = []
        for i in range(num_clients):
             self.clients.append(Client(
                  model, codebook=np.eye(num_bins), gain=gains[i],
                  min_gain=min_gain
             ))
        self.channel = Channel(noise_ratio)
    
        self.server = Server(
            model, codebook=np.eye(num_bins), min_gain=min_gain,
            noise_ratio=noise_ratio, mode=Modes.HETERO
        )

        self.N_max = None
    
    def calibrate(self, calib_data_split):
        self.N_max = max(calib_data_split, key=len)
        for i in range(self.clients):
            self.clients[i].calibrate(calib_data_split[i], self.N_max)
    
    def predict(self, alpha, images):
        for client in self.clients:
            client.transmit(self.channel)
    
        self.server.aggregate_data(self.channel)
        return self.server.pred_sets(alpha, images, self.N_max)