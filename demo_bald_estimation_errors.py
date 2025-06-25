# %%
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import get_cmap
from scipy.special import digamma
from scipy.stats import bernoulli, beta, gamma, norm, rv_continuous, rv_discrete
from scipy.stats import t as student_t
from tqdm import tqdm

from matplotlib_utils import set_font_size, use_latex_fonts


def compute_params_for_beta_distribution(
    data: np.ndarray | Sequence[int], a_0: float, b_0: float
) -> Tuple[float, float]:
    """
    References:
        https://en.wikipedia.org/wiki/Conjugate_prior#When_the_likelihood_function_is_a_discrete_distribution
    """
    if len(data) == 0:
        return a_0, b_0

    else:
        a_n = a_0 + np.sum(data)
        b_n = b_0 + len(data) - np.sum(data)

        return a_n, b_n


def compute_params_for_gaussian_gamma_distribution(
    data: np.ndarray | Sequence[float], a_0: float, b_0: float, k_0: float, m_0: float
) -> Tuple[float, float, float, float]:
    """
    We follow Murphy [1] with a = alpha, b = beta, k = kappa, m = mu.

    Note:
        b here = beta in Murphy [1]
               = lambda on Wikipedia [2]
               = 1 / theta on Wikipedia [2]
               = 1 / theta in NumPy [3]
               = beta in SciPy

    References:
    [1] https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    [2] https://en.wikipedia.org/wiki/Gamma_distribution
    [3] https://numpy.org/doc/2.0/reference/random/generated/numpy.random.Generator.gamma.html
    [4] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
    """
    if len(data) == 0:
        return a_0, b_0, k_0, m_0

    else:
        n = len(data)
        data_mean = np.mean(data)

        a_n = a_0 + (n / 2)
        b_n = (
            b_0
            + (0.5 * np.sum((data - data_mean) ** 2))
            + ((k_0 * n * (data_mean - m_0) ** 2) / (2 * (k_0 + n)))
        )
        k_n = k_0 + n
        m_n = ((k_0 * m_0) + (n * data_mean)) / (k_0 + n)

        return a_n, b_n, k_n, m_n


def estimate_info_quantities_for_bernoulli_beta_model(
    data_dist: rv_discrete, a_n: float, b_n: float
) -> Tuple[float, float, float, float]:
    def param_conditional_predictive_entropy(prob: float) -> float:
        return bernoulli(p=prob).entropy()

    def param_posterior_entropy(data: int) -> float:
        a_n_plus_1, b_n_plus_1 = compute_params_for_beta_distribution([data], a_n, b_n)
        return beta(a=a_n_plus_1, b=b_n_plus_1).entropy()

    param_dist = beta(a=a_n, b=b_n)
    pred_dist = bernoulli(p=param_dist.mean())

    data_entropy = data_dist.entropy()
    expected_cond_pred_entropy = param_dist.expect(param_conditional_predictive_entropy)

    data_param_eig = param_dist.entropy() - data_dist.expect(param_posterior_entropy)
    model_param_eig = pred_dist.entropy() - expected_cond_pred_entropy

    return data_entropy, expected_cond_pred_entropy, data_param_eig, model_param_eig


def estimate_info_quantities_for_gaussian_gaussian_gamma_model(
    data_dist: rv_continuous, a_n: float, b_n: float, k_n: float, m_n: float
) -> Tuple[float, float, float, float]:
    def expected_conditional_predictive_entropy(a_n: float, b_n: float) -> float:
        """
        E_{μ,λ}[H(x|μ,λ)] = E_{μ,λ}[(1/2) * log(2πe/λ)]
                          = (1/2) * (log(2π) + 1 - E_{λ}[log(λ)])
                          = (1/2) * (log(2π) + 1 - ψ(a) + log(b))

        where E_{λ}[log(λ)] = ψ(a) - log(b) is a standard result [1].

        References:
        [1] https://en.wikipedia.org/wiki/Gamma_distribution#Logarithmic_expectation_and_variance
        """
        return 0.5 * (np.log(2 * np.pi) + 1 - digamma(a_n) + np.log(b_n))

    def param_entropy(a_n: float, b_n: float, k_n: float) -> float:
        """
        H(μ,λ) = H(λ) + E_{λ}[H(μ|λ)]
               = H(λ) + E_{λ}[(1/2) * log(2πe/kλ)]
               = H(λ) + (1/2) * (log(2π) + 1 - log(k) - E_{λ}[log(λ)]
               = H(λ) + (1/2) * (log(2π) + 1 - log(k) - ψ(a) + log(b))

        Check:
        >>> l_dist = gamma(a=a_0, scale=b_0)
        >>> ls = l_dist.rvs(size=10_000)
        >>> m_dist = norm(loc=m_0, scale=np.sqrt(1 / (k_0 * ls)))
        >>> ms = m_dist.rvs(size=10_000)
        >>> logprobs = l_dist.logpdf(ls) + m_dist.logpdf(ms)
        >>> entropy = -np.mean(logprobs)
        """
        entropy = gamma(a=a_n, scale=b_n).entropy()
        entropy += 0.5 * (np.log(2 * np.pi) + 1 - np.log(k_n) - digamma(a_n) + np.log(b_n))
        return entropy

    def param_posterior_entropy(data: float) -> float:
        a_n_plus_1, b_n_plus_1, k_n_plus_1, _ = compute_params_for_gaussian_gamma_distribution(
            [data], a_n, b_n, k_n, m_n
        )
        return param_entropy(a_n_plus_1, b_n_plus_1, k_n_plus_1)

    pred_dist = student_t(loc=m_n, scale=np.sqrt(b_n * (k_n + 1) / (a_n * k_n)), df=(2 * a_n))

    data_entropy = data_dist.entropy()
    expected_cond_pred_entropy = expected_conditional_predictive_entropy(a_n, b_n)

    data_param_eig = param_entropy(a_n, b_n, k_n) - data_dist.expect(param_posterior_entropy)
    model_param_eig = pred_dist.entropy() - expected_cond_pred_entropy

    return data_entropy, expected_cond_pred_entropy, data_param_eig, model_param_eig


def run_updating_and_uncertainty_estimation(
    data_dist: rv_continuous | rv_discrete,
    prior_params: Dict[str, float | int],
    seeds: Iterable,
    n_examples: Iterable,
) -> Dict[str, Any]:
    pred_dists, pred_ig_errors, param_eig_errors = [], [], []

    for seed in tqdm(seeds, desc=data_dist.__class__.__name__):
        data = data_dist.rvs(size=max(n_examples), random_state=seed)

        pred_dists_seed, pred_ig_errors_seed, param_eig_errors_seed = [], [], []

        for n in n_examples:
            if hasattr(data_dist, "pmf"):
                a_n, b_n = compute_params_for_beta_distribution(data[:n], **prior_params)

                pred_dist = beta(a=a_n, b=b_n)

                data_entropy, expected_cond_pred_entropy, data_param_eig, model_param_eig = (
                    estimate_info_quantities_for_bernoulli_beta_model(data_dist, a_n, b_n)
                )

            else:
                a_n, b_n, k_n, m_n = compute_params_for_gaussian_gamma_distribution(
                    data[:n], **prior_params
                )

                pred_dist = student_t(
                    loc=m_n, scale=np.sqrt(b_n * (k_n + 1) / (a_n * k_n)), df=(2 * a_n)
                )

                data_entropy, expected_cond_pred_entropy, data_param_eig, model_param_eig = (
                    estimate_info_quantities_for_gaussian_gaussian_gamma_model(
                        data_dist, a_n, b_n, k_n, m_n
                    )
                )

            pred_ig_error = np.square(data_entropy - expected_cond_pred_entropy)
            param_eig_error = np.square(data_param_eig - model_param_eig)

            pred_dists_seed += [pred_dist]
            pred_ig_errors_seed += [pred_ig_error]
            param_eig_errors_seed += [param_eig_error]

        pred_dists += [pred_dists_seed]
        pred_ig_errors += [pred_ig_errors_seed]
        param_eig_errors += [param_eig_errors_seed]

    results = {
        "data_dist": data_dist,
        "pred_dists": pred_dists,
        "pred_ig_errors": np.array(pred_ig_errors),
        "param_eig_errors": np.array(param_eig_errors),
    }

    return results


# %%
seeds = range(1, 51)
n_examples = (1, 10, 100, 1_000)

results = {
    "discrete": run_updating_and_uncertainty_estimation(
        data_dist=bernoulli(p=0.5),
        prior_params=dict(a_0=5, b_0=5),
        seeds=seeds,
        n_examples=n_examples,
    ),
    "continuous": run_updating_and_uncertainty_estimation(
        data_dist=norm(loc=1, scale=1),
        prior_params=dict(a_0=1, b_0=5, k_0=1, m_0=0),
        seeds=seeds,
        n_examples=n_examples,
    ),
}


# %%
save_dir = Path(__file__).parent / "results"
save_dir.mkdir(parents=True, exist_ok=True)

set_font_size(13)
use_latex_fonts()

log_ns = np.log10(n_examples).astype(int)
colors = [get_cmap("viridis_r")(x) for x in np.linspace(0, 1, len(n_examples) + 1)]

n_gauss = 10
gauss_data_dist = results["continuous"]["data_dist"]
gauss_data = np.linspace(gauss_data_dist.ppf(1e-7), gauss_data_dist.ppf(1 - 1e-7), 1_000)
gauss_scale = 0.8 / max(gauss_data_dist.pdf(gauss_data))

figure, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(12, 5.5),
    gridspec_kw={"width_ratios": [1, 3.5]},
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
    axes[i, 0].legend(loc="upper right", handlelength=1.5)

for i in range(len(seeds)):
    for j in range(len(log_ns)):
        axes[0, 1].plot(
            seeds[i],
            results["discrete"]["pred_dists"][i][j].expect(),
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
        axes[1, 1].plot(
            seeds[:n_gauss][i]
            + gauss_scale * results["continuous"]["pred_dists"][i][j].pdf(gauss_data),
            gauss_data,
            color=colors[j],
            linestyle="--",
            label=(rf"$p_{{10^{log_ns[j]}}}(z)$" if i == 0 else None),
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

figure.tight_layout(w_pad=1, h_pad=1)
figure.savefig(save_dir / "bald_estimation_errors.svg", bbox_inches="tight")
figure.show()
