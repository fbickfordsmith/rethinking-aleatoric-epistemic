# %%
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from tqdm import trange

from matplotlib_utils import set_font_size, use_latex_fonts


def estimate_uncertainty_for_weighted_quadratic_loss(
    z_mean: np.ndarray,
    z_std: np.ndarray,
    weight_fn: Callable,
) -> np.ndarray:
    """
    Compute the predictive uncertainty under a weighted-quadratic loss function:
        h[p(z)] = min_a E_{p(z)}[l(a,z)]

    where
        l(a,z) = w(z).(a - z)^2.

    The optimal action is
        a^* = argmin_a E[l(a,z)]
            = E[z.w(z)] / E[w(z)]

    where all expectations are with respect to p(z).

    This follows from setting ∇_a E[l(a,z)] to zero and solving for a:
        ∇_a E[w(z).(a - z)^2] = E[2.w(z).(a - z)]
                              = 2(a.E[w(z)] - E[z.w(z)])
                              = 0.

    The uncertainty is then
        h[p(z)] = E[w(z).(a^* - z)^2]
                = (a^*)^2 E[w(z)] - 2 a^* E[z.w(z)] + E[z^2.w(z)]
                = E[z.w(z)]^2 / E[w(z)] - 2 E[z.w(z)]^2 / E[w(z)] + E[z^2.w(z)]
                = E[z^2.w(z)] - E[z.w(z)]^2 / E[w(z)].

    Here we have Gaussian p(z).
    """
    w = weight_fn
    z_w = lambda z: z * w(z)
    z_sq_w = lambda z: z**2 * w(z)

    uncertainties = np.empty(len(z_mean))

    for i in trange(len(z_mean), desc="Uncertainty"):
        z_dist = norm(loc=z_mean[i], scale=z_std[i])
        uncertainties[i] = z_dist.expect(z_sq_w) - z_dist.expect(z_w) ** 2 / z_dist.expect(w)

    return uncertainties


def estimate_expected_score_for_weighted_quadratic_loss(
    pred_z_mean: np.ndarray,
    pred_z_std: np.ndarray,
    data_z_mean: np.ndarray,
    data_z_std: np.ndarray,
    weight_fn: Callable,
) -> np.ndarray:
    """
    Compute the expected score of a model's optimal action under a weighted-quadratic loss function:
        E_{p_eval(z)}[s(p_n,z)] = E_{p_eval(z)}[l(a_n^*,z)]

    where l(a,z) and a_n^* are defined as in `estimate_uncertainty_for_weighted_quadratic_loss`.

    Here we have Gaussian p_n(z) and p_eval(z).
    """
    w = weight_fn
    z_w = lambda z: z * w(z)

    expected_scores = np.empty(len(pred_z_mean))

    for i in trange(len(pred_z_mean), desc="Evaluation"):
        pred_z_dist = norm(loc=pred_z_mean[i], scale=pred_z_std[i])
        data_z_dist = norm(loc=data_z_mean[i], scale=data_z_std[i])
        pred_optimal_action = pred_z_dist.expect(z_w) / pred_z_dist.expect(w)
        expected_scores[i] = data_z_dist.expect(lambda z: w(z) * (pred_optimal_action - z) ** 2)

    return expected_scores


# %%
rng = np.random.default_rng(seed=0)

weight_fn = lambda z: np.maximum(0, z)

data_mean_fn = lambda x: np.tanh(x)
data_std = 0.1

train_inputs = np.array([-2, 2])
train_labels = data_mean_fn(train_inputs)

plot_inputs = np.linspace(-8, 8, 256 + 1)

kernel = RBF(length_scale=1, length_scale_bounds="fixed")
kernel += WhiteKernel(noise_level=data_std**2, noise_level_bounds="fixed")

model = GaussianProcessRegressor(kernel=kernel, random_state=0)
model.fit(train_inputs.reshape(-1, 1), train_labels)

pred_z_mean, pred_z_std = model.predict(plot_inputs.reshape(-1, 1), return_std=True)
data_z_mean, data_z_std = data_mean_fn(plot_inputs), np.full(len(plot_inputs), data_std)

quad_uncertainty = pred_z_std**2
quad_dispersion = data_z_std**2
quad_discrepancy = np.square(pred_z_mean - data_z_mean)
quad_expected_score = quad_discrepancy + quad_dispersion

wquad_uncertainty = estimate_uncertainty_for_weighted_quadratic_loss(
    pred_z_mean,
    pred_z_std,
    weight_fn,
)
wquad_dispersion = estimate_uncertainty_for_weighted_quadratic_loss(
    data_z_mean,
    data_z_std,
    weight_fn,
)
wquad_expected_score = estimate_expected_score_for_weighted_quadratic_loss(
    pred_z_mean,
    pred_z_std,
    data_z_mean,
    data_z_std,
    weight_fn,
)
wquad_discrepancy = wquad_expected_score - wquad_dispersion


# %%
save_dir = Path(__file__).parent / "results"
save_dir.mkdir(parents=True, exist_ok=True)

use_latex_fonts()
set_font_size(13)

figure, axes = plt.subplots(ncols=4, figsize=(14, 2.7))

[line_0a] = axes[0].plot(
    plot_inputs,
    data_z_mean,
    color="gray",
    alpha=0.7,
    label="reference mean",
)
axes[0].fill_between(
    plot_inputs,
    data_z_mean - data_z_std,
    data_z_mean + data_z_std,
    color="gray",
    alpha=0.1,
)
[line_0b] = axes[0].plot(
    plot_inputs,
    pred_z_mean,
    color="C0",
    label="predictive mean",
)
[line_0c] = axes[0].plot(
    train_inputs,
    train_labels,
    ".",
    color="black",
    markersize=8,
    label="training data",
)
axes[0].fill_between(
    plot_inputs,
    pred_z_mean - pred_z_std,
    pred_z_mean + pred_z_std,
    color="C0",
    alpha=0.1,
)
axes[0].set(
    title="Data, reference distribution\nand predictive distribution",
    xlim=(min(plot_inputs), max(plot_inputs)),
    xlabel=r"$x$",
    ylabel=r"$z$",
)

[line_1a] = axes[1].plot(
    plot_inputs,
    quad_uncertainty,
    color="C1",
    label="predictive uncertainty",
)
[line_1b] = axes[1].plot(
    plot_inputs,
    quad_dispersion,
    color="C2",
    label="data dispersion",
)
[line_1c] = axes[1].plot(
    plot_inputs,
    quad_expected_score,
    color="C3",
    label="expected score",
)
axes[1].set(
    title="Uncertainty and evaluation\nwith loss " + r"$\ell(a,z) = (a - z)^2$",
    xlim=(min(plot_inputs), max(plot_inputs)),
    xlabel=r"$x$",
    ylabel="Value",
)

[line_2a] = axes[2].plot(
    plot_inputs,
    wquad_uncertainty,
    color="C1",
    label="predictive uncertainty",
)
[line_2b] = axes[2].plot(
    plot_inputs,
    wquad_dispersion,
    color="C2",
    label="data dispersion",
)
[line_2c] = axes[2].plot(
    plot_inputs,
    wquad_expected_score,
    color="C3",
    label="expected score",
)
axes[2].set(
    title="Uncertainty and evaluation with\nloss " + r"$\ell(a,z) = \max(0,z)\cdot(a - z)^2$",
    xlim=(min(plot_inputs), max(plot_inputs)),
    xlabel=r"$x$",
    ylabel="Value",
)

lines = [line_0c, line_0a, line_0b, line_1a, line_1b, line_1c]

axes[3].legend(lines, [line.get_label() for line in lines], handlelength=1.5, loc=(0, -0.05))
axes[3].axis("off")

figure.tight_layout(w_pad=1.5)
figure.savefig(save_dir / "uncertainty_and_evaluation.svg", bbox_inches="tight")
figure.show()
