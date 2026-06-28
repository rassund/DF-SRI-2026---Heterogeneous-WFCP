import random
from collections import defaultdict

cifar10_labels = [ "airplane", "automobile", "bird",
                      "cat", "deer", "dog",
                      "frog", "horse", "ship", "truck" ]

def split_data_homo(data, num_groups):
    """
    Splits the data homogeneously, i.e. each split should contain roughly the same proportion of each class.
    
    The data is assumed to be a list of tuples, where each tuple is of the form (image, label).
    """
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

def score_func(softmax, label):
    return 1.0 - softmax[label]