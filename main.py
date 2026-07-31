# %%
import tensorflow as tf
from experiments import coverage_experiment
from plotting import plot_coverage
from utils import ExperimentConfig

model_path = "cnn_model_cifar10.keras"

model = tf.keras.models.load_model(model_path)
(_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
test_images = test_images.astype("float32") / 255.0
data = list(zip(test_images, test_labels))

# The SNR ranges from -20dB to 20dB. SNR = 10 * log_10(1.0 / n_0).
n_0 = [
    100, 31.62, 10, 3.162, 1,
    0.316, 0.1, 0.0316, 0.01
]

# Defines the level of heterogeneity. Lower = more hetero, higher = less hetero.
dirichlet_alphas = [10, 1, 0.5, 0.1, 0.05]

alphas = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20]

config = ExperimentConfig(
    data=data,
    model=model,
    num_trials=5,
    num_clients=20,
    num_bins=60,
    num_calib_data=1000,
    num_valid_data=9000,
    min_gain=1.0,
    noise_ratio=1.0,
    dirichlet_alpha=0.5,
    alpha=0.1
)
config.generate_gains()

coverage = coverage_experiment(config, alphas)
plot_coverage(coverage, alphas)

#set_size = set_size_experiment(config, alphas)
#plot_set_size(set_size, alphas)
# %%
