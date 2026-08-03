import numpy as np
from utils import cifar10_labels, split_data_homo, split_data_hetero, get_calib_and_val_data, ExperimentConfig, ExperimentResult
from methods import CentralCP, WFCP, HETERO_WFCP

# def marginal_coverage1(model, data, num_clients, mode, alpha, num_trials, num_calib_data, noise_ratio, num_bins):
#     """
#     Based on the procedure presented in A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty
#     Quantification (2022) by Anastasios et al.

#     Marginal coverage is evaluated by constructing prediction sets for each image in the validation data and then
#     computing how many of those sets include their true label. This is done over a certain number of trials,
#     where each trial randomizes which data is calibration data and which is validation data. Marginal coverage
#     is achieved if the mean coverage over all trials is roughly equal to 1 - alpha.
#     """
#     coverages = np.zeros((num_trials,))
#     all_pred_set_sizes = []
#     codebook = np.eye(num_bins) # We use the identity matrix as codebook

#     for r in range(num_trials):
#         # Perfect CSI is assumed, thus the gains are known to the clients
#         gains = np.random.rayleigh(scale=np.sqrt(0.5), size=num_clients)
#         min_gain = 1.0

#         channel = Channel(noise_ratio)
#         server = Server(model, codebook, min_gain, noise_ratio)
        
#         np.random.shuffle(data)
#         calib_data, val_data = (data[:num_calib_data], data[num_calib_data:])

#         if mode == Modes.HOMO:
#             calib_data_split = split_data_homo(calib_data, num_clients)
#         else:
#             calib_data_split = split_data_hetero(calib_data, num_clients)
        
#         for k in range(num_clients):
#             client = Client( model, codebook, gains[k], min_gain)
#             client.calibrate(calib_data_split[k])
#             client.transmit(channel)
        
#         server.aggregate_data(channel)

#         val_images = np.array([d[0] for d in val_data])
#         val_labels = np.array([d[1] for d in val_data])

#         pred_sets = server.pred_sets(alpha, val_images)

#         # Measure sizes
#         pred_set_sizes = [len(s) for s in pred_sets]
#         all_pred_set_sizes.extend(pred_set_sizes)

#         # Calculate coverage
#         n = len(val_labels)
#         k = 0
#         for i in range(n):
#             if cifar10_labels[val_labels[i][0]] in pred_sets[i]:
#                 k += 1
#         coverages[r] = k / n
    
#     # Coverage result
#     print(f"Coverage: {coverages.mean()}")
#     # plt.figure()
#     # plt.hist(coverages, bins=10)
#     # plt.xlabel("Coverage")
#     # plt.ylabel("Frequency")
#     # plt.title("Marginal Coverage over trials")
#     # plt.show()

#     # Efficiency / adaptivity result
#     print(f"Average prediction set size: {np.mean(all_pred_set_sizes)}")
#     # plt.figure()
#     # plt.hist(all_pred_set_sizes, bins=range(1, len(cifar10_labels)+1))
#     # plt.xlabel("Prediction set size")
#     # plt.ylabel("Frequency")
#     # plt.title("Prediction set size distribution")
#     # plt.show()

# def histogram_test(model, data, split_data, num_bins, noise_ratio, num_clients, alpha):
#     client_histograms = []

#     codebook = np.eye(num_bins)

#     gains = np.random.rayleigh(scale=np.sqrt(0.5), size=20)
#     min_gain = 1.0

#     channel = Channel(noise_ratio)
#     server = Server(model, codebook, min_gain, noise_ratio)

#     np.random.shuffle(data)
#     calib_data = data[:1000]
#     calib_data_split = split_data(calib_data, num_clients)

#     for i in range(num_clients):
#         client = Client(model, codebook, gains[i], min_gain)
#         client.calibrate(calib_data_split[i])
#         client_histograms.append(client.histogram)
#         client.transmit(channel)
    
#     server.aggregate_data(channel)

#     true_histogram = np.mean(client_histograms, axis=0)
#     estimated_histogram = server.histogram

#     print("True histogram:")
#     print(true_histogram)
#     print("Estimated histogram:")
#     print(estimated_histogram)
#     print("Threshold:")
#     print(server.threshold(alpha))

#     central_server = Server(model, codebook, min_gain, noise_ratio, Modes.CENTRAL)
#     central_server.calibrate(calib_data)
#     print("Central threshold:", central_server.threshold(alpha))
#     t = data[1000:]
#     i = np.array([d[0] for d in t])
#     sets = central_server.pred_sets(alpha, i)
#     sizes = [len(S) for S in sets]
#     print("Central server average set size:", np.mean(sizes))

def coverage_and_set_size_experiment(config: ExperimentConfig, alphas=None, dirichlet_alphas=None, noise_ratios=None):
    skip_central = False

    if alphas is None:
        alphas = [config.alpha]
        skip_central = True
    if dirichlet_alphas is None:
        dirichlet_alphas = [config.dirichlet_alpha]
    if noise_ratios is None:
        noise_ratios = [config.noise_ratio]

    results = []

    for noise_ratio in noise_ratios:
        methods = {
            "central": lambda: CentralCP(config.model),
            "wfcp": lambda: WFCP(config.model, config.num_bins, config.num_clients, noise_ratio, config.gains, config.min_gain),
            "hetero": lambda: HETERO_WFCP(config.model, config.num_bins, config.num_clients, noise_ratio, config.gains, config.min_gain)
        }

        for name, method in methods.items():
            if name == "central" and skip_central:
                continue
            
            for dir_alpha in dirichlet_alphas:
                for alpha in alphas:
                    mean, std, avg_size, size_std = evaluate_method(
                        method,
                        config.data,
                        alpha,
                        config.num_trials,
                        config.num_calib_data,
                        config.num_valid_data,
                        dir_alpha,
                        1 if name == "central" else config.num_clients
                    )
                    
                    results.append(ExperimentResult(
                        method=name,
                        target_coverage=alpha,
                        dirichlet_alpha=dir_alpha,
                        noise_ratio=noise_ratio,
                        coverage=mean,
                        coverage_std=std,
                        set_size=avg_size,
                        size_std=size_std
                    ))

    return results

def evaluate_method(method_func, data, alpha, num_trials, num_calib_data, num_valid_data, dir_alpha, num_clients):
    """
    Evaluates the given method based on marginal coverage and average prediction set size.
    The evaluation is done over many trials with the result being the average of all trials.

    Parameters
    ----------
    method : instance of MethodInterface
        The CP method to evaluate.
    data : list of (image, label)
        The test data to base the evaluation on.
        This is split into a calibration set and a validation set.
    alpha : float
        The target error rate.
    num_trials : int
        The number of trials to run and average over.
    num_calib_data : int
        The number of data to use as calibration data.
    num_valid_data : int
        The number of data to use as validation data.
    dir_alpha : float
        The degree of heterogeneity.
    num_clients : int
        The number of clients to split the calibration data across.
    
    Returns
    -------
    mean : float
        The mean coverage for the target alpha over all trials.
    std : float
        The standard deviation of the coverages over all trials.
    avg_size : float
        The average prediction set sizes for the target alpha value over all trials.
    """

    coverages = []
    set_sizes = []

    for _ in range(num_trials):
        method = method_func()
        calib_data, val_data = get_calib_and_val_data(data, num_calib_data, num_valid_data)
        val_images = np.array([d[0] for d in val_data])
        val_labels = np.array([d[1] for d in val_data])
        
        if dir_alpha > 10:
            calib_data_split = split_data_homo(data=calib_data, num_groups=num_clients)
        else:
            calib_data_split = split_data_hetero(data=calib_data, num_groups=num_clients, alpha=dir_alpha)

        method.calibrate(calib_data_split)

        prediction_sets = method.predict(alpha, val_images)

        coverage = marginal_coverage(prediction_sets, val_labels)
        coverages.append(coverage)

        set_sizes.append([len(s) for s in prediction_sets])

    mean = np.mean(coverages)
    std = np.std(coverages)
    avg_size = np.mean(set_sizes)
    size_std = np.std(set_sizes)

    return mean, std, avg_size, size_std

def marginal_coverage(pred_sets, val_labels):
    return np.mean([cifar10_labels[y[0]] in pred_set for y, pred_set in zip(val_labels, pred_sets)])