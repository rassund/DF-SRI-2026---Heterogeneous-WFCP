def split_data(images, labels):
    data = [images, labels]
    data_split = [data]
    return data_split

def score_func(softmax_dist, label):
    return 1.0 - softmax_dist[label]