from typing import Sequence, Tuple

import numpy as np
from scipy.special import digamma
from scipy.stats import gamma, rv_continuous
from scipy.stats import t as student_t


def update_params_for_gaussian_gamma(
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


def estimate_info_quantities_for_gaussian_gaussian_gamma_model(
    data_dist: rv_continuous,
    a_n: float,
    b_n: float,
    k_n: float,
    m_n: float,
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
        a_n_plus_1, b_n_plus_1, k_n_plus_1, _ = update_params_for_gaussian_gamma(
            [data], a_n, b_n, k_n, m_n
        )
        return param_entropy(a_n_plus_1, b_n_plus_1, k_n_plus_1)

    pred_dist = student_t(loc=m_n, scale=np.sqrt(b_n * (k_n + 1) / (a_n * k_n)), df=(2 * a_n))

    expected_cond_pred_entropy = expected_conditional_predictive_entropy(a_n, b_n)

    data_irr_pred_entropy = data_dist.entropy()
    model_irr_pred_entropy = expected_cond_pred_entropy

    data_param_eig = param_entropy(a_n, b_n, k_n) - data_dist.expect(param_posterior_entropy)
    model_param_eig = pred_dist.entropy() - expected_cond_pred_entropy

    return data_irr_pred_entropy, model_irr_pred_entropy, data_param_eig, model_param_eig
