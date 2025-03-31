from matplotlib import pyplot as plt


def use_latex_fonts() -> None:
    """
    References:
        http://phyletica.org/matplotlib-fonts/
        https://stackoverflow.com/a/29013129
    """
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{cmbright}",
        }
    )


def set_font_size(font_size: int = 13) -> None:
    """
    References:
        https://stackoverflow.com/a/39566040
    """
    plt.rcParams.update(
        {
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "figure.titlesize": font_size,
            "font.size": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    )
