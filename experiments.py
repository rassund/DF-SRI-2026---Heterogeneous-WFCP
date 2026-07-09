import numpy as np
import matplotlib.pyplot as plt
from actors import Client, Server, Channel
from utils import cifar10_labels

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
    all_pred_set_sizes = []
    codebook = np.eye(5) # We use the identity matrix as codebook

    for r in range(num_trials):
        # Perfect CSI is assumed, thus the gains are known to the clients
        gains = np.random.rayleigh(scale=np.sqrt(0.5), size=num_clients)
        min_gain = 1.0

        channel = Channel(0.05)
        server = Server(model, codebook, num_calib_data, min_gain, 0.05)
        
        np.random.shuffle(data)
        calib_data, val_data = (data[:num_calib_data], data[num_calib_data:])
        calib_data_split = split_data(calib_data, num_clients)
        for k in range(num_clients):
            client = Client(calib_data_split[k], model, codebook, gains[k], min_gain)
            client.transmit(channel)
        
        server.aggregate_data(channel)

        val_images = np.array([d[0] for d in val_data])
        val_labels = np.array([d[1] for d in val_data])

        pred_sets = server.pred_sets(alpha, val_images)

        # Measure sizes
        pred_set_sizes = [len(s) for s in pred_sets]
        all_pred_set_sizes.extend(pred_set_sizes)

        # Calculate coverage
        n = len(val_labels)
        k = 0
        for i in range(n):
            if cifar10_labels[val_labels[i][0]] in pred_sets[i]:
                k += 1
        coverages[r] = k / n
    
    # Coverage result
    print(f"Coverage: {coverages.mean()}")
    plt.figure()
    plt.hist(coverages, bins=10)
    plt.xlabel("Coverage")
    plt.ylabel("Frequency")
    plt.title("Marginal Coverage over trials")
    plt.show()

    # Efficiency / adaptivity result
    print(f"Average prediction set size: {np.mean(all_pred_set_sizes)}")
    plt.figure()
    plt.hist(all_pred_set_sizes, bins=range(1, len(cifar10_labels)+1))
    plt.xlabel("Prediction set size")
    plt.ylabel("Frequency")
    plt.title("Prediction set size distribution")
    plt.show()

def histogram_test(model, data, split_data):
    client_histograms = []

    codebook = np.eye(5)

    gains = np.random.rayleigh(scale=np.sqrt(0.5), size=10)
    min_gain = 1.0

    channel = Channel(0.05)
    server = Server(model, codebook, 1000, min_gain, 0.05)

    np.random.shuffle(data)
    calib_data = data[:1000]
    calib_data_split = split_data(calib_data, 10)

    for i in range(10):
        client = Client(calib_data_split[i], model, codebook, gains[i], min_gain)
        client_histograms.append(client.get_histogram())
        client.transmit(channel)
    
    server.aggregate_data(channel)

    true_histogram = np.mean(client_histograms, axis=0)
    estimated_histogram = server.histogram

    print("True histogram:")
    print(true_histogram)
    print("Estimated histogram:")
    print(estimated_histogram)
    print("Threshold:")
    print(server.threshold(0.05))