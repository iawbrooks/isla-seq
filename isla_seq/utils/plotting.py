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
        obsm_key: str = 'X_umap',

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
    obsm_key : `str`, default: `'X_umap'`
        The key in `adata.obsm` from which to obtain the X,Y coordinates of the projection.
        By default, this is scanpy's default key for storing UMAP coordinates.
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
        if isinstance(data.dtype, np.number):
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
    
    # Create colorbar or legend
    if mapping_type == 'numeric':
        cax = ax.inset_axes([1.01, 0, 0.05, 1])
        norm = matplotlib.colors.Normalize(vmin=cdata.min(), vmax=cdata.max())
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(mappable, cax=cax)
    elif mapping_type == 'categorical':
        cat_represented_values = data.unique()
        legend_elements = [
            Line2D([0], [0], label=cat_unique[i], linewidth=0, marker='o', markerfacecolor=cat_colors[i], markersize=dot_size+2.0, markeredgewidth=dot_edgewidth, markeredgecolor=dot_edgecolor)
            for i in range(len(cat_unique))
            if cat_unique[i] in cat_represented_values
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=[1, 1])
    
    # Optional beautification
    if beautify:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(feature, fontweight='bold')
