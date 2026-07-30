# %%
import tensorflow as tf
from experiments import coverage_experiment, set_size_experiment
from plotting import plot_coverage, plot_set_size
from utils import ExperimentConfig

model_path = "cnn_model_cifar10.keras"

model = tf.keras.models.load_model(model_path)
(_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
test_images = test_images.astype("float32") / 255.0
data = list(zip(test_images, test_labels))

# The SNR ranges from -20dB to 20dB
n_0 = [
    100, 31.62, 10, 3.162, 1,
    0.316, 0.1, 0.0316, 0.01
]

dirichlet_alphas = [10, 1, 0.5, 0.1, 0.05] # Defines the level of heterogeneity

alphas = [0.01, 0.05, 0.10, 0.20]

config = ExperimentConfig(
    data=data,
    model=model,
    num_trials=20,
    num_clients=20,
    num_bins=60,
    num_calib_data=1000,
    min_gain=1.0,
    noise_ratio=10,
    dirichlet_alpha=0.5,
    alpha=0.1
)

coverage = coverage_experiment(config, alphas)
plot_coverage(coverage)

set_size = set_size_experiment(config, alphas)
plot_set_size(set_size)
# %%
