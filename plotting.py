import matplotlib.pyplot as plt

METHODS = [
    ("central", "o-", "Centralized CP"),
    ("wfcp", "s-", "WFCP"),
    ("hetero", "^-", "Heterogeneous WFCP")
]

PLOT_CONFIGS = {
    "coverage_vs_alpha": dict(
        x_key="alpha",
        config_x_attr="alpha",
        metric="coverage",
        skip_central=False,
        xscale="linear",
        xticks=None,
        invert_x=False,
        xlabel=r"Target coverage $(1-\alpha)$",
        ylabel="Empirical coverage",
        ideal="diagonal",
        filename="figures/coverage_plot.pdf",
    ),
    "size_vs_alpha": dict(
        x_key="alpha",
        config_x_attr="alpha",
        metric="set_size",
        skip_central=False,
        xscale="linear",
        xticks=None,
        invert_x=False,
        xlabel=r"Target coverage $(1-\alpha)$",
        ylabel="Average prediction set size",
        ideal=None,
        filename="figures/size_coverage_plot.pdf",
    ),
    "coverage_vs_dirichlet": dict(
        x_key="dir_alpha",
        config_x_attr="dirichlet_alpha",
        metric="coverage",
        skip_central=True,
        xscale="log",
        xticks=([0.05, 0.1, 0.5, 1, 10, 100], ["0.05", "0.1", "0.5", "1", "10", "IID"]),
        invert_x=True,
        xlabel=r"Dirichlet $\alpha$",
        ylabel="Empirical coverage",
        ideal=("h", None),
        filename="figures/coverage_dirichlet_plot.pdf",
    ),
    "size_vs_dirichlet": dict(
        x_key="dir_alpha",
        config_x_attr="dirichlet_alpha",
        metric="set_size",
        skip_central=True,
        xscale="log",
        xticks=([0.05, 0.1, 0.5, 1, 10, 100], ["0.05", "0.1", "0.5", "1", "10", "IID"]),
        invert_x=True,
        xlabel=r"Dirichlet $\alpha$",
        ylabel="Average prediction set size",
        ideal=None,
        filename="figures/size_dirichlet_plot.pdf",
    ),
    "coverage_vs_noise": dict(
        x_key="noise_ratio",
        config_x_attr="noise_ratio",
        metric="coverage",
        skip_central=True,
        xscale="log",
        xticks=([0.01, 0.1, 1, 10, 100], [r"$10^{-2}$", r"$10^{-1}$", r"$10^0$", r"$10^1$", r"$10^2$"]),
        invert_x=False,
        xlabel="SNR",
        ylabel="Empirical coverage",
        ideal=("h", None),
        filename="figures/coverage_noise_plot.pdf",
    ),
    "size_vs_noise": dict(
        x_key="noise_ratio",
        config_x_attr="noise_ratio",
        metric="set_size",
        skip_central=True,
        xscale="log",
        xticks=([0.01, 0.1, 1, 10, 100], [r"$10^{-2}$", r"$10^{-1}$", r"$10^0$", r"$10^1$", r"$10^2$"]),
        invert_x=False,
        xlabel="SNR",
        ylabel="Average prediction set size",
        ideal=None,
        filename="figures/size_noise_plot.pdf",
    ),
}

CONFIG_EXCLUDE_FIELDS = {"data", "model", "gains"}

METRIC_FIELDS = {
    "coverage": ("coverage", "coverage_std"),
    "set_size": ("set_size", "size_std"),
}

def _get_result(results, method, alpha=None, dir_alpha=None, noise_ratio=None):
    for r in results:
        if (
            r.method == method
            and (alpha is None or r.target_coverage == alpha)
            and (dir_alpha is None or r.dirichlet_alpha == dir_alpha)
            and (noise_ratio is None or r.noise_ratio == noise_ratio)
        ):
            return r

    raise ValueError("No matching experiment could be found.")

def _style_axes(cfg):
    if cfg["xscale"] == "log":
        plt.xscale("log")
    if cfg["xticks"] is not None:
        values, labels = cfg["xticks"]
        plt.xticks(values, labels)
    if cfg["invert_x"]:
        plt.gca().invert_xaxis()
    plt.xlabel(cfg["xlabel"])
    plt.ylabel(cfg["ylabel"])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def _add_config_caption(config, exclude=CONFIG_EXCLUDE_FIELDS):
    items = vars(config).items()
    items = [(k, v) for k, v in items if k not in exclude]

    caption = ", ".join(f"{k} = {v}" for k, v in items)

    plt.gcf().text(0.5, -0.02, caption, ha="center", va="top", fontsize=9)

def plot_metric(plot_type, results, x_values, config, error_bars=False, target_alpha=None, show_config=True):
    cfg = PLOT_CONFIGS[plot_type]
    val_field, std_field = METRIC_FIELDS[cfg["metric"]]

    if cfg["x_key"] == "alpha":
        x = [1 - a for a in x_values]
    elif cfg["x_key"] == "noise_ratio":
        x = [1 / n for n in x_values]
    else:
        x = list(x_values)

    plt.figure(figsize=(6, 6))

    for method, marker, label in METHODS:
        if cfg["skip_central"] and method == "central":
            continue

        y, yerr = [], []
        for _, raw_xv in zip(x, x_values):
            kwargs = {cfg["x_key"]: raw_xv}
            result = _get_result(results, method, **kwargs)

            if target_alpha is not None and result.target_coverage != target_alpha:
                raise ValueError(
                    "The target coverage does not match the target coverage of the experiments."
                )

            y.append(getattr(result, val_field))
            yerr.append(getattr(result, std_field, None))

        use_yerr = yerr if (error_bars and all(v is not None for v in yerr)) else None
        plt.errorbar(x, y, yerr=use_yerr, fmt=marker, capsize=4, label=label)

    if cfg["ideal"] == "diagonal":
        plt.plot([min(x), max(x)], [min(x), max(x)], "--", color="black", label="Ideal")
    elif isinstance(cfg["ideal"], tuple) and cfg["ideal"][0] == "h":
        plt.axhline(y=1 - target_alpha, linestyle="--", color="black", label="Ideal")

    _style_axes(cfg)

    if show_config:
        _add_config_caption(config, CONFIG_EXCLUDE_FIELDS | {cfg["config_x_attr"]})

    plt.savefig(cfg["filename"], bbox_inches="tight", transparent=True)
    plt.show()