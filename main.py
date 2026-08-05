# %%
import tensorflow as tf
from experiments import coverage_and_set_size_experiment
from plotting import plot_metric
from utils import ExperimentConfig, load_model_and_data

model, data = load_model_and_data(model_path="cnn_model_cifar10.keras")

# The SNR ranges from -20dB to 20dB. SNR = 10 * log_10(1.0 / n_0).
n_0 = [
    100, 31.62, 10, 3.162, 1,
    0.316, 0.1, 0.0316, 0.01
]

# Defines the level of heterogeneity. Lower = more hetero, higher = less hetero.
dirichlet_alphas = [100, 10, 1, 0.5, 0.1, 0.05]

alphas = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14]

config = ExperimentConfig(
    data=data,
    model=model,
    num_trials=40,
    num_clients=20,
    num_bins=20,
    num_calib_data=4000,
    num_valid_data=400,
    min_gain=1.0,
    noise_ratio=1.0,
    dirichlet_alpha=5.0,
    alpha=0.2
)
config.generate_gains()

results_alphas = coverage_and_set_size_experiment(config, alphas=alphas)
plot_metric("coverage_vs_alpha", results_alphas, alphas, config, error_bars=True)
plot_metric("size_vs_alpha", results_alphas, alphas, config, error_bars=True)

results_heterogeneity = coverage_and_set_size_experiment(config, dirichlet_alphas=dirichlet_alphas)
plot_metric("coverage_vs_dirichlet", results_heterogeneity, dirichlet_alphas, config, error_bars=True, target_alpha=config.alpha)
plot_metric("size_vs_dirichlet", results_heterogeneity, dirichlet_alphas, config, error_bars=True)

results_noise = coverage_and_set_size_experiment(config, noise_ratios=n_0)
plot_metric("coverage_vs_noise", results_noise, n_0, config, error_bars=True, target_alpha=config.alpha)
plot_metric("size_vs_noise", results_noise, n_0, config, error_bars=True)
# %%
