import matplotlib.pyplot as plt
from utils import ExperimentResult

methods = [
    ("central", "o-", "Centralized CP"),
    ("wfcp", "s-", "WFCP"),
    ("hetero", "^-", "Heterogeneous WFCP")
]

def plot_coverage(results, alphas, error_bars=False):
    x = [1-a for a in alphas]

    plt.figure(figsize=(6,6))

    for method, marker, label in methods:
            y = []
            yerr = []
    
            for alpha in alphas:
                result = get_result(results, method, alpha=alpha)

                y.append(result.coverage)
                yerr.append(result.coverage_std)
    
            plt.errorbar(x, y, yerr=yerr if error_bars else None, fmt=marker, capsize=4, label=label)

    plt.plot([min(x), 1.0], [min(x), 1.0], "--", color="black", label="Ideal")

    plt.xlabel(r"Target coverage $(1-\alpha)$")
    plt.ylabel("Empirical coverage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout
    plt.savefig("figures/coverage_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()

def plot_set_size_vs_coverage(results, alphas):
    x = [1-a for a in alphas]

    plt.figure(figsize=(6,6))

    for method, marker, label in methods:
        y = []

        for alpha in alphas:
            result = get_result(results, method, alpha=alpha)

            y.append(result.set_size)

        plt.errorbar(x, y, fmt=marker, capsize=4, label=label)

    plt.xlabel(r"Target coverage $(1-\alpha)$")
    plt.ylabel("Average prediction set size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout
    plt.savefig("figures/size_coverage_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()

def plot_coverage_vs_dirichlet(results, dirichlet_alphas, target_alpha, error_bars=False):
    x = dirichlet_alphas

    plt.figure(figsize=(6,6))

    for method, marker, label in methods:
        y = []
        yerr = []

        for dir_alpha in dirichlet_alphas:
            result = get_result(results, method, dir_alpha=dir_alpha)
            if result.target_coverage != target_alpha:
                raise ValueError("The target coverage does not match the target coverage of the experiments.")

            y.append(result.coverage)
            yerr.append(result.coverage_std)

        plt.errorbar(x, y, yerr=yerr if error_bars else None, fmt=marker, capsize=4, label=label)

    plt.axhline(y=1-target_alpha, linestyle="--", color="black", label="Ideal")

    plt.xscale("log")
    plt.xticks(
        [0.05, 0.1, 0.5, 1, 10],
        ["0.05", "0.1", "0.5", "1", "10"]
    )
    plt.gca().invert_xaxis()
    plt.xlabel(r"Dirichlet $\alpha$")
    plt.ylabel("Empirical coverage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/coverage_dirichlet_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()

def plot_set_size_vs_dirichlet(results, dirichlet_alphas):
    x = dirichlet_alphas
    
    plt.figure(figsize=(6,6))

    for method, marker, label in methods:
        y = []

        for dir_alpha in dirichlet_alphas:
            result = get_result(results, method, dir_alpha=dir_alpha)

            y.append(result.set_size)

        plt.errorbar(x, y, fmt=marker, capsize=4, label=label)

    plt.xscale("log")
    plt.xticks(
        [0.05, 0.1, 0.5, 1, 10],
        ["0.05", "0.1", "0.5", "1", "10"]
    )
    plt.gca().invert_xaxis()
    plt.xlabel(r"Dirichlet $\alpha$")
    plt.ylabel("Average prediction set size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout
    plt.savefig("figures/size_dirichlet_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()

def plot_coverage_vs_noise(results, noise_ratios, target_alpha, error_bars=False):
    x = [1 / n for n in noise_ratios]

    plt.figure(figsize=(6,6))

    for method, marker, label in methods:
        y = []
        yerr = []

        for noise_ratio in noise_ratios:
            result = get_result(results, method, noise_ratio=noise_ratio)
            if result.target_coverage != target_alpha:
                raise ValueError("The target coverage does not match the target coverage of the experiments.")

            y.append(result.coverage)
            yerr.append(result.coverage_std)

        plt.errorbar(x, y, yerr=yerr if error_bars else None, fmt=marker, capsize=4, label=label)

    plt.axhline(y=1-target_alpha, linestyle="--", color="black", label="Ideal")

    plt.xscale("log")
    plt.xticks(
        [0.01, 0.1, 1, 10, 100],
        [r"$10^{-2}$", r"$10^{-1}$", r"$10^0$", r"$10^1$", r"$10^2$"]
    )
    plt.xlabel("SNR")
    plt.ylabel("Empirical coverage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/coverage_noise_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()

def plot_set_size_vs_noise(results, noise_ratios):
    x = [1 / n for n in noise_ratios]
    
    plt.figure(figsize=(6,6))

    for method, marker, label in methods:
        y = []

        for noise_ratio in noise_ratios:
            result = get_result(results, method, noise_ratio=noise_ratio)

            y.append(result.set_size)

        plt.errorbar(x, y, fmt=marker, capsize=4, label=label)

    plt.xscale("log")
    plt.xticks(
        [0.01, 0.1, 1, 10, 100],
        [r"$10^{-2}$", r"$10^{-1}$", r"$10^0$", r"$10^1$", r"$10^2$"]
    )
    plt.xlabel("SNR")
    plt.ylabel("Average prediction set size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout
    plt.savefig("figures/size_noise_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()

def get_result(results, method, alpha=None, dir_alpha=None, noise_ratio=None):
    for r in results:
        if (
            r.method == method
            and (alpha is None or r.target_coverage == alpha)
            and (dir_alpha is None or r.dirichlet_alpha == dir_alpha)
            and (noise_ratio is None or r.noise_ratio == noise_ratio)
        ):
            return r

    raise ValueError("No matching experiment could be found.")