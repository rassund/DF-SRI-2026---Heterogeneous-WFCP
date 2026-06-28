cifar10_labels = [ "airplane", "automobile", "bird",
                      "cat", "deer", "dog",
                      "frog", "horse", "ship", "truck" ]

def split_data(data):
    data_split = [data]
    return data_split

def score_func(softmax, label):
    return 1.0 - softmax[label]