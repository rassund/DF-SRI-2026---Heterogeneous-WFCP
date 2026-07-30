import random
import numpy as np
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass
from typing import Any

cifar10_labels = [ "airplane", "automobile", "bird",
                      "cat", "deer", "dog",
                      "frog", "horse", "ship", "truck" ]

def split_data_homo(data, num_groups):
    """
    Splits the data homogeneously, i.e. each split should contain roughly the same proportion of each class.

    Parameters
    ----------
    data : list of (image, label)
        The dataset to split.
    num_groups : int
        Number of groups to split the data into.
    
    Returns
    -------
    data_split : list
        A list of the splits. Client i can get their dataset from data_split[i].
    """

    if num_groups == 1:
        return data

    label_groups = defaultdict(list)
    
    for image, label in data:
        label_groups[label[0]].append((image, label))
    
    for label in label_groups:
        random.shuffle(label_groups[label])

    data_split = [[] for _ in range(num_groups)]

    for label, samples in label_groups.items():
        for i, sample in enumerate(samples):
            data_split[i % num_groups].append(sample)

    for d in data_split:
        random.shuffle(d)

    return data_split

def split_data_hetero(data, num_groups, alpha=0.5, min_samples=5):
    """
    Splits the data heterogeneously, i.e. each split has a different label distribution and size.

    A different label distribution is generated for each split using a Dirichlet distribution.
    
    Parameters
    ----------
    data : list of (image, label)
        The dataset to split.
    num_groups : int
        Number of groups to split the data into.
    alpha : float
        Controls the level of heterogeneity. A smaller value is more heterogeneous. A larger value is closer to IID.
    min_samples : int
        The minimum number of samples per client.

    Returns
    -------
    data_split : list
        A list of the splits. Client i can get their dataset from data_split[i].
    """

    if num_groups == 1:
        return data

    rng = np.random.default_rng()

    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(data):
        class_indices[label[0] if isinstance(label, np.ndarray) else label].append(idx)
    
    for indices in class_indices.values():
        rng.shuffle(indices)
    
    group_indices = [[] for _ in range(num_groups)]

    for indices in class_indices.values():
        proportions = rng.dirichlet(alpha * np.ones(num_groups))
        counts = (proportions * len(indices)).astype(int)
    
        while counts.sum() < len(indices):
            counts[rng.integers(num_groups)] += 1
        
        start = 0
        for group, count in enumerate(counts):
            group_indices[group].extend(indices[start:start+count])
            start += count

    counts = np.floor(proportions * len(indices)).astype(int)
    remainder = len(indices) - counts.sum()

    for i in rng.permutation(num_groups)[:remainder]:
        counts[i] += 1

    data_split = [[data[i] for i in indices] for indices in group_indices]
    return data_split

def score_func(softmax, label):
    return 1.0 - softmax[label]

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

    d = random.sample(data, len(data))
    calib_data, val_data = (d[:num_calib_data], d[num_calib_data:])
    return calib_data, val_data


class Modes(Enum):
    CENTRAL = 1
    HOMO = 2
    HETERO = 3


@dataclass
class ExperimentConfig:
    data: list
    model: Any
    num_trials: int
    num_clients: int
    num_bins: int
    num_calib_data: int
    min_gain: float
    noise_ratio: float
    dirichlet_alpha: float
    alpha: float
    gains: list = None

    def generate_gains(self):
        self.gains = np.random.rayleigh(scale=np.sqrt(0.5), size=self.num_clients)