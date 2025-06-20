from typing import Optional, Literal

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.lines import Line2D

from .anndatas import get_expr_matrix


def plot_umap_ax(
        adata: sc.AnnData,
        ax: plt.Axes,
        feature: str,
        *,
        layer: Optional[str] = None,
        cmap: str | matplotlib.colors.Colormap = 'viridis',
        cat_cmap: str | matplotlib.colors.Colormap | list[str] | None = None,
        dot_size: float = 8.0,
        dot_edgewidth: float = 0.1,
        dot_edgecolor = 'k',
        obs_filt: np.ndarray | pd.Series = None,
        shuffle_rng: Optional[int] = None,
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
        up another as a result.
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

    # Optional shuffle
    if shuffle_rng is not None:
        rng = np.random.RandomState(shuffle_rng)
        shuffle_indices = np.arange(len(final_colors), dtype=int)
        rng.shuffle(shuffle_indices)
        umap_coord_arr = umap_coord_arr[shuffle_indices]
        final_colors = final_colors[shuffle_indices]
        if mapping_type == 'categorical':
            data = data.iloc[shuffle_indices]

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


def get_blank_axs_array(
        *,
        naxs: int = None,
        nrows: int = None,
        ncols: int = None,
        ax_w: float,
        ax_h: float,
        dpi: int = 200,
        invisible: bool = False,
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
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)

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
