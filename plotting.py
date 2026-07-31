import matplotlib.pyplot as plt

def plot_coverage(coverage, alphas, error_bars=False):
    target = [1-a for a in alphas]

    (central_means, central_stds, _) = coverage["central"]
    (homo_means, homo_stds, _) = coverage["wfcp"]
    (hetero_means, hetero_stds, _) = coverage["hetero"]

    plt.figure(figsize=(6,6))

    plt.errorbar(target, central_means, yerr=central_stds if error_bars else None,
                 fmt="o-", capsize=4, label="Centralized CP")
    plt.errorbar(target, homo_means, yerr=homo_stds if error_bars else None,
                 fmt="s-", capsize=4, label="WFCP")
    plt.errorbar(target, hetero_means, yerr=hetero_stds if error_bars else None,
                 fmt="^-", capsize=4, label="Heterogeneous WFCP")

    plt.plot([min(target), 1.0], [min(target), 1.0], "--", color="black", label="Ideal")

    plt.xlabel(r"Target coverage $(1-\alpha)$")
    plt.ylabel("Empirical coverage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout
    plt.savefig("figures/coverage_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()

def plot_set_size(results, alphas):
    target = [1-a for a in alphas]

    (_, _, central_sizes) = results["central"]
    (_, _, homo_sizes) = results["wfcp"]
    (_, _, hetero_sizes) = results["hetero"]

    plt.figure(figsize=(6,6))

    plt.plot(target, central_sizes, marker="o", label="Centralized CP")
    plt.plot(target, homo_sizes, marker="s", label="WFCP")
    plt.plot(target, hetero_sizes, marker="^", label="Heterogeneous WFCP")

    plt.xlabel(r"Target coverage $(1-\alpha)$")
    plt.ylabel("Average prediction set size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout
    plt.savefig("figures/size_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()