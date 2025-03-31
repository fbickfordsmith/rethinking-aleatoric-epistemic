from typing import Sequence, Tuple

import numpy as np
from scipy.stats import bernoulli, beta, rv_discrete


def update_params_for_beta(
    data: np.ndarray | Sequence[int], a_0: float, b_0: float
) -> Tuple[float, float]:
    """
    References:
        https://en.wikipedia.org/wiki/Conjugate_prior#When_the_likelihood_function_is_a_discrete_distribution
    """
    a_n = a_0 + np.sum(data)
    b_n = b_0 + len(data) - np.sum(data)

    return a_n, b_n


def estimate_info_quantities_for_bernoulli_beta_model(
    data_dist: rv_discrete, a_n: float, b_n: float
) -> Tuple[float, float, float, float]:
    def param_conditional_predictive_entropy(prob: float) -> float:
        return bernoulli(p=prob).entropy()

    def param_posterior_entropy(data: int) -> float:
        a_n_plus_1, b_n_plus_1 = update_params_for_beta([data], a_n, b_n)
        return beta(a=a_n_plus_1, b=b_n_plus_1).entropy()

    param_dist = beta(a=a_n, b=b_n)
    pred_dist = bernoulli(p=param_dist.mean())

    expected_cond_pred_entropy = param_dist.expect(param_conditional_predictive_entropy)

    data_irr_pred_entropy = data_dist.entropy()
    model_irr_pred_entropy = expected_cond_pred_entropy

    data_param_eig = param_dist.entropy() - data_dist.expect(param_posterior_entropy)
    model_param_eig = pred_dist.entropy() - expected_cond_pred_entropy

    return data_irr_pred_entropy, model_irr_pred_entropy, data_param_eig, model_param_eig
