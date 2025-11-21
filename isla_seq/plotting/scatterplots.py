from typing import Literal, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.figure
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from umap import UMAP

from ..utils import get_expr_matrix, ColumnPath, ColumnCondition

from .utils import get_blank_axs_array

def plot_umaps(
        adata: sc.AnnData,
        features: list[str | ColumnPath],
        *,
        ncols: int = 4,
        dpi: float = 200,
        ax_w: float = 5,
        ax_h: float = 5,
        **kwargs
    ) -> tuple[plt.Figure, np.ndarray]:
    """
    Plots a grid of UMAPs, calling plot_umap_ax for each one.

    Parameters
    ---
    adata : `scanpy.AnnData`
        The AnnData from which to plot the projection.
    features : `str`
        The names of rows in `adata.var` or columns in `adata.obs`, or ColumnPaths.
    ncols : `int`, default: 4
        The number of columns for the generated axes grid.
    dpi : `float`, default: 200
        The dots per inch of the generated figure.
    ax_w : `float`, default: 5
        The width, in inches, to allocate per axis
    ax_h : `float`, default: 5
        The height, in inches, to allocate per axis
    kwargs:
        Keyword arguments to pass along to each call of `plot_umap_ax`
    """
    fig, axs = get_blank_axs_array(naxs=len(features), ncols=ncols, ax_w=ax_w, ax_h=ax_h, dpi=dpi, invisible=True)
    for feature, ax in zip(features, axs.flat):
        ax: plt.Axes
        ax.set_visible(True)
        ax.set_aspect(1)
        plot_umap_ax(adata, ax, feature, **kwargs)
    
    return fig, axs


def plot_umap_ax(
        adata: sc.AnnData,
        ax: plt.Axes,
        feature: str | ColumnPath,
        *,
        layer: str | None = None,
        cmap: str | matplotlib.colors.Colormap = 'viridis',
        cat_cmap: str | matplotlib.colors.Colormap | list[str] | None = None,
        dot_size: float = 8.0,
        dot_edgewidth: float = 0,
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
        The name of a row in `adata.var` or column in `adata.obs`, or a ColumnPath.
    layer : `str | None`, default: `None`
        When plotting expression data, the data will be drawn from `adata.layers[layer]` if
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

    # Get feature data
    if isinstance(feature, ColumnPath):
        feature_series = feature.get(adata)
    elif feature in adata.var.index:
        feature_series = get_expr_matrix(adata, feature, layer=layer, ret_type='pandas')[feature]
    elif feature in adata.obs.columns:
        feature_series = adata.obs[feature]
    else:
        raise ValueError(f"Feature '{feature}' not found in either `adata.var.index` or `adata.obs.columns`")

    # Get final colors
    feature_is_numeric = pd.api.types.is_numeric_dtype(feature_series.dtype)
    if feature_is_numeric:
        # Numeric: Simply normalize data and use colormap to get colors
        cdata: np.ndarray = feature_series.to_numpy(dtype=float, copy=True)
        cdata -= cdata.min()
        cdata /= cdata.max()
        final_colors = cmap(cdata)
    else:
        # Categorical: Determine order of categories
        if isinstance(feature_series.dtype, pd.CategoricalDtype):
            cat_unique = list(feature_series.dtype.categories)
        else:
            cat_unique = sorted(feature_series.unique())
        # Get categorical cmap
        if cat_cmap is None:
            # Attempt to look for existing colors in adata.uns
            uns_key = f"{feature_series.name}_colors"
            if uns_key in adata.uns and len(adata.uns[uns_key]) == len(cat_unique):
                cat_cmap = adata.uns[uns_key]
            else:
                if len(cat_unique) <= 20:
                    cat_cmap = sc.pl.palettes.default_20
                elif len(cat_unique) <= 28:
                    cat_cmap = sc.pl.palettes.default_28
                else:
                    cat_cmap = sc.pl.palettes.default_102
        elif isinstance(cat_cmap, str):
            cat_cmap = plt.get_cmap(cat_cmap)
        # Get colors of each category
        if isinstance(cat_cmap, (list, np.ndarray)):
            cat_colors = [cat_cmap[i % len(cat_cmap)] for i in range(len(cat_unique))]
        else:
            cat_colors = cat_cmap(np.linspace(0, 1, len(cat_unique), endpoint=True))
        # Map categorical data to colors
        cat_mapping = {
            cat_unique[i] : cat_colors[i]
            for i in range(len(cat_unique))
        }
        final_colors = [cat_mapping[x] for x in feature_series]

    # Get UMAP coords
    umap_coord_arr = adata.obsm[obsm_key][:, :2]

    # Perform optional filtering
    final_colors = np.array(final_colors)
    if obs_filt is not None:
        feature_series = feature_series[obs_filt]
        umap_coord_arr = umap_coord_arr[obs_filt]
        final_colors   = final_colors[obs_filt]

    # Optional shuffle and/or sorting
    reorder_indices = None
    if sort_numeric is not None and feature_is_numeric:
        if sort_numeric == 'ascending':
            reorder_indices = np.argsort(feature_series)
        elif sort_numeric == 'descending':
            reorder_indices = np.argsort(feature_series)[::-1]
    elif shuffle_rng is not None:
        reorder_indices = np.arange(len(final_colors), dtype=int)
        np.random.RandomState(shuffle_rng).shuffle(reorder_indices)
    if reorder_indices is not None:
        feature_series = feature_series.iloc[reorder_indices]
        umap_coord_arr = umap_coord_arr[reorder_indices]
        final_colors   = final_colors[reorder_indices]

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
    if cat_autotext and not feature_is_numeric:
        for cat_value in cat_unique:
            cat_filt = feature_series == cat_value
            centroid_x, centroid_y = umap_coord_arr[cat_filt].mean(axis=0)
            ax.text(centroid_x, centroid_y, str(cat_value), fontsize=8, fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    # Create colorbar or legend
    if feature_is_numeric:
        cax = ax.inset_axes([1.01, 0, 0.05, 1])
        norm = matplotlib.colors.Normalize(vmin=feature_series.min(), vmax=feature_series.max())
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(mappable, cax=cax)
    elif cat_legend:
        cat_represented_values = [x for x in cat_unique]
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
        ax.set_title(feature_series.name, fontweight='bold')


def compute_umap_pretty(
        adata: sc.AnnData,
        *,
        # PCA settings
        pca_use_rep: str | None = None,
        n_pcs = 100,
        bias_vector_key: str = 'log1p_n_genes_by_counts',
        max_bias_vector_correlation = 0.7,
        # UMAP settings
        n_neighbors: int = 25,
        negative_sample_rate: int = 5,
        min_dist: float = 0.4,
        # Downsample settings
        cluster_key: str = 'leiden',
        max_cells_per_cluster = 1000,
        max_cells_total: int = 250000,
        impute_method: Literal['knn', 'transform'] = 'knn',
        impute_knn: int = 15,
        # Filtering settings
        layer: str | None = 'CPM_log1p',
        var_filt: str = 'highly_variable',
        # Miscellaneous
        random_state: int = 42,
        verbose: bool = True,
        key_added: str | None = None,
    ) -> np.ndarray:
    """
    Implements the UMAP computation methodology from [Yao et al., 2023](https://www.nature.com/articles/s41586-023-06812-z),
    as described in their methods section. Specifically, this function downsamples the dataset in a cluster-aware manner,
    performs PCA while removing principal components that are highly correlated with a "bias vector" (by default,
    the log of the number of genes expressed in each cell), computes UMAP on the downsampled data, and then imputes
    UMAP coordinates for the excluded cells using a KNN regressor.

    This function is significantly faster than computing UMAP directly on all cells, especially for large datasets.
    Whether the resulting UMAP is "pretty" is left to the user to adjudicate.

    Parameters
    ---
    adata : `AnnData`
        The AnnData object from which to compute the UMAP.
    pca_use_rep : `str | None`, default: `None`
        If not `None`, uses the PCA representation stored in `adata.obsm[pca_use_rep]` instead of computing PCA.
        If `None`, computes PCA on the expression matrix.
    n_pcs : `int`, default: `100`
        The number of principal components to use.
    bias_vector_key : `str`, default: `'log1p_n_genes_by_counts'`
        The name of a column in `adata.obs` with the "bias vector", defined in Yao et al. 2023 as the log of
        the number of genes expressed in each cell.
    max_bias_vector_correlation : `float`, default: `0.7`
        The maximum absolute correlation between the bias vector and the principal components.
        Principal components with a correlation greater than this value will be removed.
    n_neighbors : `int`, default: `25`
        The number of neighbors to use in UMAP computation.
    negative_sample_rate : `int`, default: `5`
        The negative sample rate to use in UMAP computation.
    min_dist : `float`, default: `0.4`
        The minimum distance between points in UMAP computation.
    cluster_key : `str`, default: `'leiden'`
        The key in `adata.obs` that contains the cluster labels.
    max_cells_per_cluster : `int`, default: `1000`
        The maximum number of cells to include from each cluster when computing UMAP.
    max_cells_total : `int`, default: `250000`
        The maximum number of cells to include in UMAP computation.
    impute_method : `Literal['knn', 'transform']`, default: `'knn'`
        The method to use for imputing UMAP coordinates for excluded cells.
        * `'knn'` - uses a KNN regressor to predict UMAP coordinates based on the PCA coordinates.
        * `'transform'` - uses the UMAP transform method to predict UMAP coordinates.
    
    impute_knn : `int`, default: `15`
        The number of nearest neighbors to use for imputing UMAP coordinates when `impute_method` is `'knn'`.
    layer : `str | None`, default: `'CPM_log1p'`
        The layer from which to obtain the expression matrix.
        If `None`, uses `adata.X`. Irrelevant if `pca_use_rep` is not `None`.
    var_filt : `str`, default: `'highly_variable'`
        The name of a column in `adata.var` that indicates which genes to use for UMAP computation.
        Only genes with `adata.var[var_filt]` set to `True` will be used. Irrelevant if `pca_use_rep`
        is not `None`.
    random_state : `int`, default: `42`
        The random state to use for reproducibility. Relevant for downsampling, PCA, and UMAP.
    verbose : `bool`, default: `True`
        Whether to print progress messages.
    key_added : `str | None`, default: `None`
        If not `None`, the computed UMAP coordinates will be stored in `adata.obsm[key_added]`.

    Returns
    ---
    `numpy.ndarray`
        The computed (and imputed) UMAP coordinates, as a 2D array of shape (n_cells, 2).
        If `key_added` is not `None`, the UMAP coordinates will also be stored in `adata.obsm[key_added]`.
    """
    # Check params
    vprint = print if verbose else lambda x: None

    # Get expression matrix
    genes = adata.var.index[adata.var[var_filt]]
    expr = get_expr_matrix(adata, genes, layer=layer, ret_type='numpy')
    n_cells_total = len(expr)

    # Downsample, complicatedly
    vprint("Downsampling...")
    rng = np.random.RandomState(random_state)
    ## Cluster downsampling
    cluster_values = adata.obs[cluster_key].values.copy()
    indices_all = np.arange(len(cluster_values), dtype=int)
    rng.shuffle(indices_all)
    cluster_values = cluster_values[indices_all]
    indices_by_cluster = defaultdict(list)
    for idx, clust in zip(indices_all, cluster_values):
        cluster_idx_list = indices_by_cluster[clust]
        if len(cluster_idx_list) < max_cells_per_cluster:
            cluster_idx_list.append(idx)
    ## Overall downsampling
    indices_downsampled: np.ndarray = np.r_[*list(indices_by_cluster.values())]
    indices_downsampled_set = set(indices_downsampled)
    indices_excluded = np.array([x for x in range(n_cells_total) if x not in indices_downsampled_set])
    if len(indices_downsampled) > max_cells_total:
        rng.shuffle(indices_downsampled)
        indices_downsampled = indices_downsampled[:max_cells_total]
    np.sort(indices_downsampled)
    ## Final downsampled array
    expr_downsampled : np.ndarray = expr[indices_downsampled]
    expr_excluded    : np.ndarray = expr[indices_excluded]
    n_cells_downsampled = len(expr_downsampled)
    vprint(f" -> Downsampled to {n_cells_downsampled}/{n_cells_total} cells")

    # PCA
    vprint("PCA...")
    if pca_use_rep is None:
        pca_fitter = PCA(n_pcs, random_state=random_state)
        pca = pca_fitter.fit_transform(expr_downsampled) # (n_cells, n_pcs)
        pca_excluded = pca_fitter.transform(expr_excluded)
        vprint(" -> PCA computed")
    else:
        if verbose and adata.obsm[pca_use_rep].shape[1] < n_pcs:
            vprint(f" -> \033[31mRepresentation '{pca_use_rep}' has only {adata.obsm[pca_use_rep].shape[1]} "
                   f"components, but n_pcs = {n_pcs}\033[0m")
        pca          = adata.obsm[pca_use_rep][indices_downsampled, :n_pcs]
        pca_excluded = adata.obsm[pca_use_rep][indices_excluded,    :n_pcs]
        n_pcs = pca.shape[1]
        vprint(f" -> Using representation '{pca_use_rep}'")

    # Remove biased PCs
    bias_vector = adata.obs[bias_vector_key].values[indices_downsampled]
    corr: np.ndarray = np.corrcoef(pca.T, bias_vector)
    bias_corr = corr[-1, :-1] # last row, all except the last column (only want bias_vector x pca correlation values)
    pca_indices, = np.where(bias_corr <= max_bias_vector_correlation)
    pca          = pca         [:, pca_indices]
    pca_excluded = pca_excluded[:, pca_indices]
    vprint(f" -> {pca.shape[1]}/{n_pcs} principal components remaining after filtering")

    # UMAP
    vprint("UMAP...")
    umap_fitter = UMAP(
        n_neighbors = n_neighbors,
        negative_sample_rate = negative_sample_rate,
        min_dist = min_dist,
        random_state = random_state,
    )
    umap = umap_fitter.fit_transform(pca)
    vprint(" -> UMAP done")

    # Impute UMAP coordinates if necessary
    # It may seem silly to not just call umap_fitter.transform(), but imputing coordinates
    # from nearest neighbors is the methodology used in Yao et al. 2023
    if n_cells_downsampled < n_cells_total:
        vprint("Imputing UMAP coordinates")
        indices_recombined = np.r_[indices_downsampled, indices_excluded]
        indices_argsort = np.argsort(indices_recombined)
        if impute_method == 'knn':
            ## Predict UMAP coordinates based on neighbors in PCA space
            knn_fitter = KNeighborsRegressor(impute_knn, weights='distance')
            knn_fitter.fit(pca, umap)
            umap_excluded = knn_fitter.predict(pca_excluded)
        elif impute_method == 'transform':
            ## Transform directly from UMAP object
            umap_excluded = umap_fitter.transform(pca_excluded)
        else:
            raise ValueError(f"Unrecognized impute_method '{impute_method}'")
        ## Concatenate final coordinates
        umap_recombined = np.r_[umap, umap_excluded]
        umap_recombined = umap_recombined[indices_argsort]


    # Optionally save result in obsm
    if key_added is not None:
        adata.obsm[key_added] = umap_recombined

    return umap_recombined


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


def anchor_umap_by_condition(
        adata: sc.AnnData,
        obsm_key: str = 'X_umap',
        *,
        columns: tuple[int, int] = (0, 1),
        center_method: Literal['mean', 'median', 'bounds'] = 'mean',
        left: ColumnCondition | None = None,
        right: ColumnCondition | None = None,
        top: ColumnCondition | None = None,
        bottom: ColumnCondition | None = None,
        inplace: bool = True,
    ) -> np.ndarray:
    """
    Flip a UMAP or other `.obsm` array such that certain conditions or subsets
    of cells appear more on one side of the array than the other. Modifies the
    positions of *all* cells, not just those that meet the specified condition,
    so all relative distances are maintained.

    For example, to make sure that *Cxcl14*-positive cells appear more on the bottom
    and left sides of a UMAP, one would write:
    ```python
    # Make a condition with which to anchor
    cond = ColumnPath("var::Cxcl14") > 0
    
    anchor_umap_by_condition(adata, left=cond, bottom=cond)
    ```

    Parameters
    ---
    adata : `AnnData`
        The AnnData object whose UMAP or array to modify.
    obsm_key : `str`, default `"X_umap"`
        The key of the array in `adata.obsm` to modify.
    columns : `tuple[int, int]`, default `(0, 1)`
        Which columns of the array in `.obsm` to use as the X (horizontal) and Y
        (vertical) axes when interpreting the `left`, `right`, `top`, `bottom`
        parameters.
    center_method : `'mean' | 'median' | 'bounds'`, default `mean`
        Whether to anchor the cells belonging to a condition according to that
        condition's mean location, median location, or halfway between its minumum
        and maximum edges.
    left : `ColumnCondition | None`, optional
        Anchor a condition to the left side. Mutually exclusive with `right`.
    right : `ColumnCondition | None`, optional
        Anchor a condition to the right side. Mutually exclusive with `left`.
    top : `ColumnCondition | None`, optional
        Anchor a condition to the top side. Mutually exclusive with `bottom`.
    bottom : `ColumnCondition | None`, optional
        Anchor a condition to the bottom side. Mutually exclusive with `top`.
    inplace : `bool`, default `True`
        Whether to modify the specified array in-place. Always returns a 2D
        numpy array consisting of the two columns specified in `columns`
        regardless of this parameter.
    """
    # Check params
    if left is not None and right is not None:
        raise ValueError("Only one of `right` or `left` may be specified")
    if top is not None and bottom is not None:
        raise ValueError("Only one of `top` or `bottom` may be specified")
    umap = adata.obsm[obsm_key]
    umap = umap[:, columns].copy()

    # Get dataset centers
    def get_centers(arr: np.ndarray) -> tuple[float, float]:
        match center_method:
            case 'mean':
                centers = np.mean(arr, axis=0)
            case 'median':
                centers = np.median(arr, axis=0)
            case 'bounds':
                centers = (arr.max(axis=0) + arr.min(axis=0)) / 2.0
        return tuple(centers)
    
    centers = get_centers(umap)

    # Perform left-right adjustment
    filt_x = None
    if right is not None:
        filt_x = ~(right.eval(adata))
    elif left is not None:
        filt_x = left.eval(adata)
    
    if filt_x is not None:
        x, _ = get_centers(umap[filt_x])
        if x > centers[0]:
            umap[:, 0] *= -1
    
    # Perform top-bottom adjustment
    filt_y = None
    if top is not None:
        filt_y = ~(top.eval(adata))
    elif bottom is not None:
        filt_y = bottom.eval(adata)
    
    if filt_y is not None:
        _, y = get_centers(umap[filt_y])
        if y > centers[1]:
            umap[:, 1] *= -1
    
    # Optionally modify inplace
    if inplace:
        adata.obsm[obsm_key][:, columns] = umap
    
    return umap
