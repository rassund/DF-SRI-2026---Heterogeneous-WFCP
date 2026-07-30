import numpy as np
import matplotlib.pyplot as plt
from actors import Client, Server, Channel
from utils import cifar10_labels, split_data_homo, split_data_hetero, Modes
from methods import CentralCP, WFCP, HETERO_WFCP

def marginal_coverage(model, data, num_clients, mode, alpha, num_trials, num_calib_data, noise_ratio, num_bins):
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
    codebook = np.eye(num_bins) # We use the identity matrix as codebook

    for r in range(num_trials):
        # Perfect CSI is assumed, thus the gains are known to the clients
        gains = np.random.rayleigh(scale=np.sqrt(0.5), size=num_clients)
        min_gain = 1.0

        channel = Channel(noise_ratio)
        server = Server(model, codebook, min_gain, noise_ratio)
        
        np.random.shuffle(data)
        calib_data, val_data = (data[:num_calib_data], data[num_calib_data:])

        if mode == Modes.HOMO:
            calib_data_split = split_data_homo(calib_data, num_clients)
        else:
            calib_data_split = split_data_hetero(calib_data, num_clients)
        
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

def histogram_test(model, data, split_data, num_bins, noise_ratio, num_clients, alpha):
    client_histograms = []

    codebook = np.eye(num_bins)

    gains = np.random.rayleigh(scale=np.sqrt(0.5), size=20)
    min_gain = 1.0

    channel = Channel(noise_ratio)
    server = Server(model, codebook, min_gain, noise_ratio)

    np.random.shuffle(data)
    calib_data = data[:1000]
    calib_data_split = split_data(calib_data, num_clients)

    for i in range(num_clients):
        client = Client(calib_data_split[i], model, codebook, gains[i], min_gain)
        client_histograms.append(client.histogram)
        client.transmit(channel)
    
    server.aggregate_data(channel)

    true_histogram = np.mean(client_histograms, axis=0)
    estimated_histogram = server.histogram

    print("True histogram:")
    print(true_histogram)
    print("Estimated histogram:")
    print(estimated_histogram)
    print("Threshold:")
    print(server.threshold(alpha))

def coverage_experiment(config, alphas):
    results = {
        "central": [],
        "wfcp": [],
        "hetero": []
    }

    central = CentralCP(config.model),
    wfcp = WFCP(config.model, config.num_bins, config.num_clients, config.noise_ratio, config.gains, config.min_gain),
    hetero = HETERO_WFCP(config.model, config.num_bins, config.num_clients, config.noise_ratio, config.gains, config.min_gain)

    methods = [central, wfcp, hetero]

    for i in range(3):
        means, stds = evaluate_coverage(methods[i], config.data, alphas, config.num_trials, config.num_calib_data)
        results[i].append({
            "means": means,
            "stds": stds
        })

    return results

def evaluate_coverage(method, data, alphas, num_trials, num_calib_data):
    """
    Evaluates the given method based on marginal coverage.
    The evaluation is done over many trials with the result being the average of all trials.

    Parameters
    ----------
    method : instance of MethodInterface
        The CP method to evaluate.
    data : list of (image, label)
        The test data to base the evaluation on.
        This is split into a calibration set and a validation set.
    alphas : list of floats
        The error rates to evaluate.
    num_trials : int
        The number of trials to run and average over.
    num_calib_data : int
        The number of data to use as calibration data. The rest will be used as validation data.
    """

    calib_data, val_data = get_calib_and_val_data(data, num_calib_data)

    means = []
    stds = []

    for alpha in alphas:
        coverages = []

        for trial in range(num_trials):
            method.calibrate(calib_data_split)
            prediction_sets = method.predict(alpha, val_images)

            coverage = marginal_coverage(prediction_sets, val_labels)
            coverages.append(coverage)

        means.append(np.mean(coverages))
        stds.append(np.std(coverages))

    return means, stds

def get_calib_and_val_data(data, num_calib_data):
    """
    Picks a random sample of data as calibration data and the remaining data as validation data.

    Parameters
    ----------
    data : list of (image, label)
        The dataset to sample from.
    num_calib_data : int
        The number of data to pick as calibration data.
    
    Returns
    -------
    calib_data : list of (image, label)
        The sampled calibration data.
    val_data : list of (image, label)
        The remaining validation data.
    """
    if num_calib_data > len(data):
        raise ValueError("The number of calibration data cannot be greater than the total number of data.")

    d = np.random.sample(data, len(data))
    calib_data, val_data = (data[:num_calib_data], data[num_calib_data:])
    return calib_data, val_data