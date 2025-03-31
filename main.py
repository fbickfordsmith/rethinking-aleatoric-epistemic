from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy.stats import bernoulli, beta, norm, rv_continuous, rv_discrete
from scipy.stats import t as student_t
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap
from src.continuous import (
    estimate_info_quantities_for_gaussian_gaussian_gamma_model,
    update_params_for_gaussian_gamma,
)
from src.discrete import estimate_info_quantities_for_bernoulli_beta_model, update_params_for_beta
from src.matplotlib import set_font_size, use_latex_fonts


def run_updating_and_uncertainty_estimation(
    data_dist: rv_continuous | rv_discrete,
    prior_params: Tuple,
    seeds: Iterable,
    n_examples: Iterable,
) -> Dict[str, Any]:
    model_dists, pred_ig_errors, param_eig_errors = [], [], []

    for seed in tqdm(seeds):
        data = data_dist.rvs(size=max(n_examples), random_state=seed)

        model_dists_seed, pred_ig_errors_seed, param_eig_errors_seed = [], [], []

        for n in n_examples:
            if hasattr(data_dist, "pmf"):
                a_0, b_0 = prior_params

                if n == 0:
                    a_n, b_n = a_0, b_0
                else:
                    a_n, b_n = update_params_for_beta(data[:n], a_0, b_0)

                model_dist = beta(a=a_n, b=b_n)

                data_irr_pred_entropy, model_irr_pred_entropy, data_param_eig, model_param_eig = (
                    estimate_info_quantities_for_bernoulli_beta_model(data_dist, a_n, b_n)
                )

            else:
                a_0, b_0, k_0, m_0 = prior_params

                if n == 0:
                    a_n, b_n, k_n, m_n = a_0, b_0, k_0, m_0
                else:
                    a_n, b_n, k_n, m_n = update_params_for_gaussian_gamma(
                        data[:n], a_0, b_0, k_0, m_0
                    )

                model_dist = student_t(
                    loc=m_n, scale=np.sqrt(b_n * (k_n + 1) / (a_n * k_n)), df=(2 * a_n)
                )

                data_irr_pred_entropy, model_irr_pred_entropy, data_param_eig, model_param_eig = (
                    estimate_info_quantities_for_gaussian_gaussian_gamma_model(
                        data_dist, a_n, b_n, k_n, m_n
                    )
                )

            pred_ig_error = np.square(data_irr_pred_entropy - model_irr_pred_entropy)
            param_eig_error = np.square(data_param_eig - model_param_eig)

            model_dists_seed += [model_dist]
            pred_ig_errors_seed += [pred_ig_error]
            param_eig_errors_seed += [param_eig_error]

        model_dists += [model_dists_seed]
        pred_ig_errors += [pred_ig_errors_seed]
        param_eig_errors += [param_eig_errors_seed]

    results = {
        "data_dist": data_dist,
        "model_dists": model_dists,  # [len(seeds), len(n_examples)]
        "pred_ig_errors": np.array(pred_ig_errors),  # [len(seeds), len(n_examples)]
        "param_eig_errors": np.array(param_eig_errors),  # [len(seeds), len(n_examples)]
    }

    return results


def plot_results(
    results: Dict[str, Any],
    seeds: Iterable | Sequence,
    log_ns: Sequence,
    colors: Sequence,
    n_gauss: int,
    gauss_data: np.ndarray,
    gauss_scale: float,
) -> Tuple[Figure, Axes]:
    figure, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1, 3.5]}
    )

    for i, problem in enumerate(results):
        pred_ig_errors = np.mean(results[problem]["pred_ig_errors"], axis=0)
        param_eig_errors = np.mean(results[problem]["param_eig_errors"], axis=0)

        axes[i, 0].plot(log_ns, pred_ig_errors, label="Predictive IG")
        axes[i, 0].plot(log_ns, param_eig_errors, label="Parameter EIG")
        axes[i, 0].hlines(0, log_ns[0], log_ns[-1], linestyles=":", colors="gray")
        axes[i, 0].set(
            xlabel=r"Number of examples, $n$",
            xticks=log_ns,
            xticklabels=(rf"$10^{int(log_n)}$" for log_n in log_ns),
            ylabel=r"Squared error (nats$^2$)",
            title=(
                "Estimation error\nfor discrete data"
                if problem == "discrete"
                else "Estimation error\nfor continuous data"
            ),
        )
        axes[i, 0].legend(loc="upper right")

    for i in range(len(seeds)):
        for j in range(len(log_ns)):
            axes[0, 1].plot(
                seeds[i],
                results["discrete"]["model_dists"][i][j].expect(),
                ".",
                color=colors[j],
                label=(rf"$n = 10^{log_ns[j]}$" if i == 0 else None),
                zorder=2,
            )

        axes[0, 1].plot(
            seeds[i],
            results["discrete"]["data_dist"].pmf(1),
            ".",
            color="gray",
            label=(r"$n = \infty$" if i == 0 else None),
            zorder=0,
        )

    axes[0, 1].set(
        xlabel="Random seed",
        xticks=(10, 20, 30, 40, 50),
        xlim=(min(seeds) - 1, max(seeds) + 10),
        ylabel=r"$p_n(z=1)$",
        title="Predictive distributions for discrete data",
    )
    axes[0, 1].legend(loc="center right", handlelength=0.5)

    for i in range(len(seeds[:n_gauss])):
        for j in range(len(log_ns)):
            if i == 0:
                if j == 0:
                    label = r"$p_{10^0}(z)$"
                elif j == 1:
                    label = r"$p_{10^1}(z)$"
                elif j == 2:
                    label = r"$p_{10^2}(z)$"
                elif j == 3:
                    label = r"$p_{10^3}(z)$"
            else:
                label = None

            axes[1, 1].plot(
                seeds[:n_gauss][i]
                + gauss_scale * results["continuous"]["model_dists"][i][j].pdf(gauss_data),
                gauss_data,
                color=colors[j],
                linestyle="--",
                label=label,
                zorder=2,
            )

        axes[1, 1].plot(
            seeds[:n_gauss][i] + gauss_scale * results["continuous"]["data_dist"].pdf(gauss_data),
            gauss_data,
            color="gray",
            label=(r"$p_\infty(z)$" if i == 0 else None),
            zorder=0,
        )

    axes[1, 1].set(
        xlabel="Random seed",
        xticks=seeds[:n_gauss],
        xlim=(min(seeds[:n_gauss]) - 0.2, max(seeds[:n_gauss]) + 2.7),
        title="Predictive distributions for continuous data",
        ylabel=r"$z$",
    )
    axes[1, 1].legend(loc="center right", handlelength=1)

    return figure, axes


def main() -> None:
    seeds = range(1, 51)
    n_examples = (1, 10, 100, 1_000)

    results = {
        "discrete": run_updating_and_uncertainty_estimation(
            data_dist=bernoulli(p=0.5),
            prior_params=(5, 5),  # (a_0, b_0)
            seeds=seeds,
            n_examples=n_examples,
        ),
        "continuous": run_updating_and_uncertainty_estimation(
            data_dist=norm(loc=1, scale=1),
            prior_params=(1, 5, 1, 0),  # (a_0, b_0, k_0, m_0)
            seeds=seeds,
            n_examples=n_examples,
        ),
    }

    gauss_data_dist = results["continuous"]["data_dist"]
    gauss_data = np.linspace(gauss_data_dist.ppf(1e-7), gauss_data_dist.ppf(1 - 1e-7), 1_000)

    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    set_font_size(13)
    use_latex_fonts()

    figure, _ = plot_results(
        results=results,
        seeds=seeds,
        log_ns=np.log10(n_examples).astype(int),
        colors=[get_cmap("viridis_r")(x) for x in np.linspace(0, 1, len(n_examples) + 1)],
        n_gauss=10,
        gauss_data=gauss_data,
        gauss_scale=(0.8 / max(gauss_data_dist.pdf(gauss_data))),
    )

    figure.tight_layout(w_pad=1, h_pad=1)
    figure.savefig(save_dir / "estimation_errors.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
