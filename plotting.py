import matplotlib.pyplot as plt

def plot_coverage(coverage, alphas):
    target = [1-a for a in alphas]

    (central_means, central_stds) = coverage["central"]
    (homo_means, homo_stds) = coverage["wfcp"]
    (hetero_means, hetero_stds) = coverage["hetero"]

    plt.figure(figsize=(6,6))
    plt.errorbar(target, central_means, yerr=central_stds, fmt="o-", capsize=4, label="Centralized CP")
    plt.errorbar(target, homo_means, yerr=homo_stds, fmt="s-", capsize=4, label="WFCP")
    plt.errorbar(target, hetero_means, yerr=hetero_stds, fmt="^-", capsize=4, label="Heterogeneous WFCP")
    plt.plot([0.8, 1.0], [0.8, 1.0], "--", color="black", label="Ideal")

    plt.xlabel("Target coverage")
    plt.ylabel("Empirical coverage")
    plt.xlim(0.79, 1.0)
    plt.ylim(0.79, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout
    plt.show()