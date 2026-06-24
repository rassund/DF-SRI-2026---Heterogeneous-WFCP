import numpy as np
import matplotlib.pyplot as plt
from actors import Client, Server, Channel

def marginal_coverage(model, data, num_clients, split_data, alpha, num_trials, num_calib_data):
    """
    Based on the procedure presented in A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty
    Quantification (2022) by Anastasios et al.

    Marginal coverage is evaluated by constructing prediction sets for each image in the validation data and then
    computing how many of those sets include their true label. This is done over a certain number of trials,
    where each trial randomizes which data is calibration data and which is validation data. Marginal coverage
    is achieved if the mean coverage over all trials is roughly equal to 1 - alpha.
    """
    coverages = np.zeros((num_trials,))
    for r in range(num_trials):
        channel = Channel()
        server = Server()
        np.random.shuffle(data)
        calib_data, val_data = (data[:num_calib_data], data[num_calib_data:])
        calib_data_split = split_data(calib_data)
        for k in range(num_clients):
            client = Client(calib_data_split[k], model, 1)
            client.transmit(channel)
        
        server.aggregate_data(channel)
        pred_sets = server.pred_sets(alpha)
        (_, val_labels) = val_data
        n = len(val_labels)
        k = 0
        for i in n:
            if val_labels[i] in pred_sets[i]:
                k += 1
        coverages[r] = k / n
    
    print("Coverage: " + coverages.mean())
    plt.hist(coverages)