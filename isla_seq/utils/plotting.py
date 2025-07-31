from typing import Literal, Sequence, Any

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge

from .anndatas import get_expr_matrix


def plot_umap_ax(
        adata: sc.AnnData,
        ax: plt.Axes,
        feature: str,
        *,
        layer: str | None = None,
        cmap: str | matplotlib.colors.Colormap = 'viridis',
        cat_cmap: str | matplotlib.colors.Colormap | list[str] | None = None,
        dot_size: float = 8.0,
        dot_edgewidth: float = 0.1,
        dot_edgecolor = 'k',
        obs_filt: np.ndarray | pd.Series = None,
        shuffle_rng: int | None = None,
        sort_numeric: Literal['ascending', 'descending'] | None = None,
        obsm_key: str = 'X_umap',
        cat_autotext: bool = False,
        cat_legend: bool = True,

        beautify: bool = True,
    ):
    """
    Plots a UMAP (or other projection) with nice formatting.

    Parameters
    ---
    adata : `scanpy.AnnData`
        The AnnData from which to plot the projection.
    ax : `matplotlib.pyplot.Axes`
        The `axes` object onto which to plot the projection.
    feature : `str`
        The name of a row in `adata.var`, or column in `adata.obs`, whose data to plot.
        If a row in `var`, plots expression data for that feature.
        If a column in `obs`, plots that column's data.
    layer : `str | None`, default: `None`
        When plotting expression data, the data will be drawn from `adata.layers[layer]` when
        `layer` is not `None`, otherwise expression data will be drawn from `adata.X`.
    cmap : `str | Colormap`
        When plotting numeric data, data points will be colored according to this colormap.
    cat_cmap : `str | Colormap | list[str] | None`, default: `None`
        Used when plotting categorical or non-numeric data.
        If `str`, uses the matplotlib colormap by that name.
        If `Colormap`, colors the categories across the whole dynamic range of the colormap,
        in sorted order of the category names.
        If `list`, assumes list elements are valid matplotlib colors. Cycles through the list
        in order when assigning category colors, and will wrap in the case of overflow.
        By default, chooses the smallest appropriate default color list from `scanpy.pl.palettes`.
    dot_size : `float`
        Size of the scatterplot points.
    dot_edgewidth : `float`
        Width of the edges of the scatterplot points.
    dot_edgecolor :
        Color of the edges of the scatterplot points.
    obs_filt : `numpy.ndarray | pandas.Series`, default: `None`
        Optional filter on the points to plot.
    shuffle_rng : `int | None`, default: `None`
        When provided, seeds a random number generator and shuffles the order in which to plot
        points. Helpful when a dataset is ordered by conditions, and one condition is covering
        up another as a result. When data is numeric and `sort_numeric` is not None, this
        argument is ignored.
    sort_numeric : `Literal['ascending', 'descending'] | None`, default: `None`
        When specified AND data is numeric, sorts values ascending or descending before plotting.
        `'ascending'` means the highest values will be plotted on top; `'descending'` means the
        lowest values will be plotted on top.
    obsm_key : `str`, default: `'X_umap'`
        The key in `adata.obsm` from which to obtain the X,Y coordinates of the projection.
        By default, this is scanpy's default key for storing UMAP coordinates.
    cat_autotext : `bool`, default: `False`
        When plotting categorical data (e.g. clusters), labels the centroid (mean position)
        of each category on the plot.
    cat_legend : `bool`, default: `True`
        When plotting categorical data, specifies whether to plot a legend of categories.
    beautify : `bool`, default: `True`
        Whether to make the plots look a little extra bonita. Adds a title and eliminates
        the X and Y ticks.
    """
    # Check parameters
    if obsm_key not in adata.obsm:
        raise ValueError(f"Embedding key '{obsm_key}' not found; have you computed the UMAP yet?")
    if isinstance(cmap, str):
        cmap: matplotlib.colors.Colormap = plt.get_cmap(cmap)
    if isinstance(cat_cmap, str):
        cat_cmap: matplotlib.colors.Colormap = plt.get_cmap(cat_cmap)
    if obs_filt is not None and len(obs_filt) != len(adata.obs):
        raise ValueError("The length of `obs_filt` must match the length of `adata.obs`")
    if isinstance(obs_filt, pd.Series):
        if not obs_filt.index.equals(adata.obs.index):
            raise ValueError("When `obs_filt` is a Series, its index must match that of `adata.obs`")
        obs_filt = obs_filt.values

    # Determine whether feature is categorical or numeric
    cdata = None
    mapping_type: Literal['categorical', 'numeric'] = None
    if feature in adata.var.index:
        # gene -- easy; just plot expression over cmap
        cdata = get_expr_matrix(adata, feature, layer=layer)[:, 0]
        mapping_type = 'numeric'
    elif feature in adata.obs.columns:
        # obs column -- must determine whether numeric or categorical
        data = adata.obs[feature]
        if pd.api.types.is_numeric_dtype(data.dtype):
            # If numeric, easy; just plot over cmap
            cdata = data
            mapping_type = 'numeric'
        else:
            mapping_type = 'categorical'
            # If categorical, must check whether color mapping already exists in adata
            # First get all unique values of the column...
            if isinstance(data.dtype, pd.CategoricalDtype):
                cat_unique = list(data.dtype.categories)
            else:
                cat_unique = sorted(data.unique())
            # Then attempt to find existing color assignment for those values...
            uns_key = f"{feature}_colors"
            if uns_key in adata.uns:
                # If an entry exists for the feature in `uns`, make sure it has the right number of color entries
                cat_colors = adata.uns[uns_key]
                if len(cat_colors) != len(cat_unique):
                    raise ValueError(
                        f"Color mapping for categorical feature '{feature}' in `uns` ('{uns_key}') is not one-to-one with said feature."
                        f"'{feature}' has {len(cat_unique)} unique values, while '{uns_key}' has {len(cat_colors)} entries!"
                    )
            else:
                # If no entry exists in `uns`, we must make our own categorical color mapping.
                # If no cmap specified, choose a default list from scanpy
                if cat_cmap is None:
                    if len(cat_unique) <= 20:
                        cat_cmap = sc.pl.palettes.default_20
                    elif len(cat_unique) <= 28:
                        cat_cmap = sc.pl.palettes.default_28
                    else:
                        cat_cmap = sc.pl.palettes.default_102
                # If cmap is a list, simply select colors from that list in order
                if isinstance(cat_cmap, list):
                    cat_colors = [cat_cmap[i % len(cat_cmap)] for i in range(len(cat_unique))]
                else:
                    cat_colors = cat_cmap(np.linspace(0, 1, len(cat_unique), endpoint=True))
    else:
        # Feature does not exist
        raise ValueError(f"Could not find feature '{feature}' in either `adata.var.index` or `adata.obs.columns`")

    # Generate colors!
    if mapping_type == 'numeric':
        cdata_norm = cdata.copy()
        cdata_norm -= cdata_norm.min()
        cdata_norm /= cdata_norm.max()
        final_colors = cmap(cdata_norm)
    elif mapping_type == 'categorical':
        cat_mapping = {
            cat_unique[i] : cat_colors[i]
            for i in range(len(cat_unique))
        }
        final_colors = [cat_mapping[x] for x in data]
    else:
        raise ValueError("This line should not execute lol")

    # Get UMAP coords
    umap_coord_arr = adata.obsm[obsm_key][:, :2]

    # Perform optional filtering
    final_colors = np.array(final_colors)
    if obs_filt is not None:
        umap_coord_arr = umap_coord_arr[obs_filt]
        final_colors = final_colors[obs_filt]
        if mapping_type == 'categorical':
            data = data[obs_filt]

    # Optional shuffle and/or sorting
    reorder_indices = None
    if sort_numeric is not None and mapping_type == 'numeric':
        if sort_numeric == 'ascending':
            reorder_indices = np.argsort(cdata)
        elif sort_numeric == 'descending':
            reorder_indices = np.argsort(cdata)[::-1]
    elif shuffle_rng is not None:
        reorder_indices = np.arange(len(final_colors), dtype=int)
        np.random.RandomState(shuffle_rng).shuffle(reorder_indices)
    if reorder_indices is not None:
        umap_coord_arr = umap_coord_arr[reorder_indices]
        final_colors = final_colors[reorder_indices]
        if mapping_type == 'categorical':
            data = data.iloc[reorder_indices]

    # Plot!
    umap_x, umap_y = umap_coord_arr.T
    ax.scatter(
        umap_x, umap_y,
        color=final_colors,
        linewidths=dot_edgewidth,
        edgecolors=dot_edgecolor,
        s=dot_size
    )
    ax.set_box_aspect(1)

    # Optionally mark categories over the figure
    if mapping_type == 'categorical' and cat_autotext:
        for cat_value in cat_unique:
            cat_filt = data == cat_value
            centroid_x, centroid_y = umap_coord_arr[cat_filt].mean(axis=0)
            ax.text(centroid_x, centroid_y, str(cat_value), fontsize=8, fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    # Create colorbar or legend
    if mapping_type == 'numeric':
        cax = ax.inset_axes([1.01, 0, 0.05, 1])
        norm = matplotlib.colors.Normalize(vmin=cdata.min(), vmax=cdata.max())
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(mappable, cax=cax)
    elif mapping_type == 'categorical' and cat_legend:
        cat_represented_values = data.unique()
        legend_elements = [
            Line2D([0], [0], label=cat_unique[i], linewidth=0, marker='o', markerfacecolor=cat_colors[i], markersize=dot_size+4.0, markeredgewidth=dot_edgewidth, markeredgecolor=dot_edgecolor)
            for i in range(len(cat_unique))
            if cat_unique[i] in cat_represented_values
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=[1, 1])
    
    # Optional beautification
    if beautify:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(feature, fontweight='bold')


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


def transform_square_to_circular(
        x: np.ndarray,
        y: np.ndarray,
        center_method: Literal['mean', 'median', 'stretch'] = 'stretch',
        *,
        transform_weight: float = 1.0,
        inplace: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts square coordinates to circular coordinates using an elliptical transform,
    as described [here](https://squircular.blogspot.com/2015/09/mapping-circle-to-square.html).

    Parameters
    ---
    x : `ndarray`
        The X coordinate array.
    y : `ndarray`
        The Y coordinate array.
    center_method : `Literal['mean', 'median', 'stretch']`, default: `'stretch'`
        How to center and normalize the data to the unit square.
        * `'mean'` : centers data around the mean X and Y coordinates; does not change X/Y aspect ratio.
        * `'median'` : centers data around the median X and Y coordinates; does not change X/Y aspect ratio.
        * `'stretch'` : stretches data such that it takes up the full range [-1, 1] in both dimensions. May
          distort X/Y aspect ratio.
    transform_weight : `float`, default: `1.0`
        When transforming the data, how much to weight the target coordinate compared to the old
        coordinate. A weight of 1.0 applies the transform completely; a weight of 0.0 does not
        alter the input coordinates at all.
    inplace : `bool`, default: `False`
        Whether to modify the input arrays in place.
    """
    # Check params
    if not 0 <= transform_weight <= 1.0:
        raise ValueError("correction_factor must be in the range [0, 1]")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    # Ensure data is numpy array
    xy = np.array([x, y]).T

    # Center & normalize to range [-1, 1] in both dimensions
    if center_method == 'stretch':
        xy -= xy.min(axis=0)
        xy /= xy.max(axis=0) / 2
        xy -= 1
    elif center_method == 'mean':
        xy -= xy.mean(axis=0)
        xy /= np.abs(xy).max()
    elif center_method == 'median':
        xy -= np.median(xy, axis=0)
        xy /= np.abs(xy).max()
    else:
        raise ValueError(f"Invalid center_method: '{center_method}'")

    # Apply final transform
    x1, y1 = xy.T
    x1, y1 = (
        x1 * ((1 - 0.5 * y1**2) ** (transform_weight/2)),
        y1 * ((1 - 0.5 * x1**2) ** (transform_weight/2)),
    )

    # Optionally modify in-place
    if inplace:
        x[:] = x1
        y[:] = y1

    return x1, y1
