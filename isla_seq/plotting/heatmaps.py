from typing import Sequence, Literal, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import scanpy as sc

from .. import utils as ut


def correlation(
        adata: sc.AnnData,
        genes: list[str],
        layer: str | None = None,
        cmap: Any = 'coolwarm',
        vmin: float | None = -1,
        vmax: float | None = 1,

    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the Pearson correlation among a list of genes.

    Parameters
    ---
    adata : `AnnData`
        The AnnData object from which to obtain expression data.
    genes : `list[str]`
        The gene names for which to compute the correlation matrix.
    layer : `str | None`, optional
        The layer in `adata.layers` from which to obtain expression.
    cmap : `Any`
        A matplotlib colormap.
    vmin : `float | None`, default `-1.0`
        An artificial minimum imposed upon the data when scaling the color map.
    vmax : `float | None`, default `1.0`
        An artificial maximum imposed upon the data when scaling the color map.
    """
    # Compute correlation
    corr = ut.compute_correlation_matrix(adata, genes, layer)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
    cax = ax.inset_axes([1.02, 0, 0.05, 1])
    im = ax.imshow(corr, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(im, cax)

    # Set tick labels
    ax.set_xticks(np.arange(len(genes)))
    ax.set_yticks(np.arange(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha='right')
    ax.set_yticklabels(genes)

    return fig, ax


def dotplot_multi(
        vals: Sequence[np.ndarray],
        *,
        scaling: Literal['radius', 'area'] = 'radius',
        norm: Literal['all', 'row', 'column'] = 'all',
        minval: float | Literal['auto'] = 0,
        maxval: float | Literal['auto'] = 'auto',
        max_radius: float = 0.4,
        min_radius: float = 0.0,
        fig_inches_per_plot = 0.5,
        fig_dpi: int = 100,
        wedge_rotation: float = 0,
        colors: list | None = None,
        edgecolor: Any = 'k',
        edgewidth: float = 1.0,
        alpha: float = 1.0,
        pie_split: bool = True,
        summary_text: bool = False,
        summary_text_fontdict: dict | None = None,
        ax_text: np.ndarray | None = None,
        ax_text_fontdict: np.ndarray | None = None,
    ):
    """
    Generate a dotplot with multiple values represented per "dot".

    Parameters
    ---
    vals : `Sequence[ndarray]`
        A sequence of 2-dimensional arrays, or a 3-dimensional array to be interpreted as such.
    scaling : `Literal['radius', 'area']`, default: `'radius'`
        Whether dot radius or area should scale with the values in `vals`.
    norm : Literal['all', 'row', 'column'], default: `'all'`
        How to normalize the values in `vals` before plotting.
        * `'all'` : normalize across the entire array, so that all dot sizes are relative to the
          maximum value in `vals`.
        * `'row'` : normalize each row independently.
        * `'column'` : normalize each column independently.
    
    minval : `float | Literal['auto']`, default: `0`
        The minimum value to use when normalizing the values in `vals`.
        If `'auto'`, the minimum value is determined from the data.
    maxval : `float | Literal['auto']`, default: `'auto'`
        The maximum value to use when normalizing the values in `vals`.
        If `'auto'`, the maximum value is determined from the data.
    max_radius : `float`, default: `0.4`
        The maximum radius of the dots in the plot.
    min_radius : `float`, default: `0.0`
        The minimum radius of the dots in the plot.
    fig_inches_per_plot : `float`, default: `0.5`
        The width and height that each dot contributes to the figure size.
        The figure will be `(ncols + 1) * fig_inches_per_plot + 1` inches wide and
        `(nrows + 1) * fig_inches_per_plot + 1` inches tall.
    fig_dpi : `int`, default: `100`
        The resolution of the figure in dots per inch.
    wedge_rotation : `float`, default: `0`
        The rotation of the wedges in degrees, clockwise.
    colors : `list | None`, default: `None`
        Colors to use for each wedge or dot in the plot.
        If `None`, uses the default matplotlib color cycle.
        The length of this list must match the number of layers in `vals`.
    edgecolor : `Any`, default: `'k'`
        The color of the edges of the wedges or dots in the plot.
    edgewidth : `float`, default: `1.0`
        The width of the edges of the wedges or dots in the plot.
    alpha : `float`, default: `1.0`
        The opacity of the wedges or dots in the plot.
    pie_split : `bool`, default: `True`
        If `True`, plots each "dot" as a pie chart, with the value of each
        layer represented as a separate wedge.
        If `False`, plots each "dot" as a circle, with the value of each layer
        corresponding to that circle's size.
    summary_text : `bool`, default: `False`
        Whether to display the maximum value of each row or column, or the total
        maximum value (depending on the `norm` parameter) in the plot.
    summary_text_fontdict : `dict | None`, default: `None`
        Font properties for the maximum value text, if present.
    ax_text : `ndarray | None`, default: `None`
        An array of the same shape as each individual array in `vals`, containing text to display
        at each position in the plot. Useful for significance annotations, for example.
    ax_text_fontdict : `dict | None`, default: `None`
        Font properties for the text in `ax_text`, if present.
    """
    # Validate arrays
    vals: np.ndarray = np.stack([v for v in vals]).astype(float)
    if vals.ndim != 3:
        raise ValueError(f"`vals` must be a sequence of 2D arrays, or a 3D array")
    if vals.size == 0:
        raise ValueError("`vals` must not be empty")

    nlayers, nrows, ncols = vals.shape
    
    # Validate colors
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color'][:nlayers]
        if len(colors) < nlayers:
            raise ValueError(f"`colors` must be specified if the length of `vals` exceeds the number of matplotlib default colors ({len(colors)})")
    elif len(colors) != nlayers:
        raise ValueError(f"`colors` must have the same length as `vals`")
    
    # Validate normalization parameters
    if norm != 'all' and (minval not in [0, 'auto'] or maxval != 'auto'):
        raise ValueError(f"When row or column normalization is specified, minval must be 0 or 'auto' and maxval must be 'auto'")

    # Other parameters
    if ax_text is not None and ax_text.shape != vals.shape[1:]:
        raise ValueError("ax_text must have the same shape as each individual array of values in vals")
    if ax_text_fontdict is None:
        ax_text_fontdict = dict(
            fontfamily = 'arial',
            fontsize = 8,
        )

    # Normalize between 0 and 1
    radii = vals.copy()
    if norm == 'all':
        # Normalize across the entire array
        if minval == 'auto':
            minval = np.nanmin(vals)
        if maxval == 'auto':
            maxval = np.nanmax(vals)
        radii -= minval
        radii /= (maxval - minval)
    elif norm == 'row':
        if minval == 'auto':
            minval = np.nanmin(radii, axis=1)
        maxval = np.nanmax(radii, axis=(0, 2))
        radii = np.swapaxes((np.swapaxes(radii, 1, 2) - minval) / (maxval - minval), 1, 2)
    elif norm == 'column':
        if minval == 'auto':
            minval = np.nanmin(radii, axis=0)
        maxval = np.nanmax(radii, axis=(0, 1))
        radii = (radii - minval) / (maxval - minval)
    else:
        raise ValueError(f"Unrecognized norm parameter '{norm}'")
    radii[radii < 0] = 0
    radii[radii > 1] = 1
    radii: np.ndarray

    # Use appropriate scaling metric
    if scaling == 'radius':
        pass
    elif scaling == 'area':
        np.sqrt(radii, out=radii)
    else:
        raise ValueError(f"Unrecognized scaling parameter '{scaling}'")
    
    # Adjust between minimum and maximum radii values
    radii *= (max_radius - min_radius)
    radii += min_radius

    # Computes thetas based on input data
    def get_thetas(data: np.ndarray) -> list[tuple[float, float] | tuple[None, None]]:
        notnan = ~np.isnan(data)
        notnan_count = np.cumsum(notnan)
        data_nlayers = notnan_count[-1] # number of layers without nan values
        thetas = [
            ((nn_count - 1) * 360 / data_nlayers + wedge_rotation, nn_count * 360 / data_nlayers + wedge_rotation)
            if nn
            else (None, None)
            for nn, nn_count in zip(notnan, notnan_count)
        ] if pie_split else [
            (0.0, 360.0)
            if nn
            else (None, None)
            for nn in notnan
        ]
        return thetas

    # Actually plot!
    figsize = (
        (ncols + 1) * fig_inches_per_plot + 1,
        (nrows + 1) * fig_inches_per_plot + 1,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='w', dpi=fig_dpi)
    ax: plt.Axes

    # Iterate over values in this layer
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            pos_data = radii[:, row_idx, col_idx]
            layer_thetas = get_thetas(pos_data)

            # If not splitting into pies, ensure data is plotted in order of big to small
            if pie_split:
                layer_order = range(nlayers)
            else:
                layer_order = reversed(np.argsort(pos_data))

            for layer_idx in layer_order:
                # Get layer-specific info
                theta1, theta2 = layer_thetas[layer_idx]
                if theta1 is None:
                    continue
                layer_color = colors[layer_idx]

                # Make wedge
                radius = pos_data[layer_idx]
                wedge = Wedge((col_idx, row_idx), radius, theta1, theta2, fc=layer_color, ec=edgecolor, linewidth=edgewidth, alpha=alpha)
                ax.add_artist(wedge)
            
            # Optional significance text:
            if ax_text is not None:
                txt = ax_text[row_idx, col_idx]
                ax.text(col_idx, row_idx-0.41, txt, ax_text_fontdict, horizontalalignment='center', verticalalignment='center')
    
    ax.set_aspect(1)
    ax.set_xlim(-max_radius * 2, ncols + max_radius * 2 - 1)
    ax.set_ylim(-max_radius * 2, nrows + max_radius * 2 - 1)
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_ylim(*reversed(ax.get_ylim()))

    if summary_text:
        def exp_fmt(num: float, sigfigs: int = 3) -> str:
            """
            Healthy coding practices :)
            """
            pow10 = int(np.floor(np.log10(num)))
            disp_num = num / 10**pow10
            return f"{disp_num:.{sigfigs - 1}f}×$10^{pow10}$"

        if norm == 'all':
            ax.text(ncols + 0.1, 0.1, exp_fmt(maxval), fontdict=summary_text_fontdict)
        elif norm == 'row':
            for row_idx in range(nrows):
                ax.text(ncols + 0.1, row_idx+0.1, exp_fmt(maxval[row_idx]), fontdict=summary_text_fontdict)
        elif norm == 'col':
            for col_idx in range(ncols):
                ax.text(col_idx + 0.1, nrows + 0.1, exp_fmt(maxval[col_idx]), rotation=45, ha='right', va='top', fontdict=summary_text_fontdict)

    return fig, ax
