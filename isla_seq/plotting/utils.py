import numpy as np
import matplotlib.pyplot as plt

def get_blank_axs_array(
        *,
        naxs: int = None,
        nrows: int = None,
        ncols: int = None,
        ax_w: float,
        ax_h: float,
        dpi: int = 200,
        invisible: bool = False,
        **kwargs,
    ) -> tuple[plt.Figure, np.ndarray]:
    """
    Generate a new figure with an array of empty axes.

    Does the math for number of rows and columns for you, if you only care about the number of
    axes and the number of either rows or columns.

    Parameters
    ---
    naxs : `int | None`, default: `None`
        The minimum number of axes that will be present in the generated figure.
    nrows : `int | None`, default: `None`
        The number of rows of axes in the generated figure.
    ncols : `int | None`, default: `None`
        The number of columns of axes in the generated figure.
    ax_w : `float`
        The width, in inches, of each axis.
    ax_h : `float`
        The height, in inches, of each axis.
    dpi : `int`, default: `200`
        The resolution of the generated figure.
    invisible : `bool`, default: `False`
        Whether to hide all axes by default, calling `.set_visible(False)` on each one.
    **kwargs : `dict`
        Any additional arguments to pass when calling `plt.subplots`
    
    Returns
    ---
    A tuple `(fig, axs)`, where `fig` is the generated matplotlib Figure object, and 
    `axs` is a 2D numpy array of the figure's Axes objects.
    """
    if sum(x is not None for x in (ncols, nrows, naxs)) != 2:
        raise ValueError("Exactly two of [ncols, nrows, naxs] must be specified")
    
    # Determine nrows, ncols
    if nrows is None:
        nrows = naxs // ncols + (naxs % ncols != 0)
    elif ncols is None:
        ncols = naxs // nrows + (naxs % nrows != 0)
    else:
        naxs = nrows * ncols
    
    # Create figure
    figsize = (ncols * ax_w, nrows * ax_h)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, **kwargs)

    # Adjust ax array to correct size if needed
    if naxs == 1:
        axs = np.array([[axs]])
    else:
        axs: np.ndarray = axs.reshape(nrows, ncols)
    
    # Optionally hide axes
    if invisible:
        for ax in axs.flat:
            ax: plt.Axes
            ax.set_visible(False)
    
    return fig, axs
