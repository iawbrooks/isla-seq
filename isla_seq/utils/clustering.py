import itertools
import sys
from typing import Optional, Sequence
from warnings import warn

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from numba import njit

from . import anndatas, preprocessing


def cluster_recursively(
        adata: sc.AnnData,
        *,
        n_genes: int,
        batch_key: str | None = None,
        corr_merge_thresh: float,
        leiden_resolution: float,
        verbose: bool = True,
        logfile = sys.stdout,
    ) -> pd.DataFrame:
    """
    Recursively subclusters and recombines `adata` using a combination of the Leiden
    clustering algorithm and cluster cross-correlations.

    The algorithm can be summarized as follows:
    1. Assign all cells to one large origin cluster, `0`. Mark this cluster with the flag `final=False`.
    2. For each cluster, `C`, for which `final=False`:
       - **(a)** Split it with leiden, generating `N` subclusters named `C,S` (where `S` starts counting from 0)
       - **(b)** If there is only one subcluster (that is, `C,S` = `C`), mark `C` with flag `final=True` and 
                 move onto the next cluster.
       - **(b)** Compute the `N x N` cross-correlation (xcorr) matrix between all subclusters `C,S`.
       - **(c)** Select the highest (non-diagonal) value in the xcorr matrix. If it exceeds `corr_merge_thresh`,
                 merge its two corresponding subclusters `C,i` and `C,j` and return to step **2b**
                 That is, `C,i <- Union(C,i; C,j)` and `C,j <- ∅`.
       - **(d)** If no xcorr values exceed the merge threshold, add all remaining subclusters `C,S` to the
                 global cluster set with the flag `final=False`.
    3. If there are any clusters for which `final=False`, return to step **2**. Otherwise, exit!
    """
    def print_verbose(x):
        if verbose:
            print(x, file=logfile)

    # Compute HVGs
    gene_meta = sc.pp.highly_variable_genes(adata, layer='CPM_log1p', n_top_genes=n_genes, flavor='seurat', inplace=False, batch_key=batch_key)
    gene_meta.set_index(adata.var.index, inplace=True)
    highly_variable_genes = gene_meta.index.values[gene_meta['highly_variable']]

    # Set everything to the same leiden cluster in the beginning
    adata.obs['leiden_base'] = pd.Series(index=adata.obs.index, data='0').astype("category")
    leiden_is_final = dict[str, bool]()
    for leiden in adata.obs['leiden_base'].unique():
        leiden_is_final[leiden] = False

    # Main clustering loop
    all_leidens_final = False
    adata.obs['leiden_new'] = '' # Column for keeping track of each leiden's subclustering
    round_counter = 0
    hidden_leiden_steps = [] # Keep track of each cell's identity history
    while not all_leidens_final:
        all_leidens_final = True
        round_counter += 1

        # Do this to avoid weird dtype=category error
        adata.obs['leiden_new'] = adata.obs['leiden_new'].astype(str)

        # Get all leidens which have not been deemed final
        all_leidens = sorted(adata.obs['leiden_base'].unique())
        nonfinal_leidens = [x for x in all_leidens if not leiden_is_final[x]]
        print_verbose(f"ROUND {round_counter}: {len(nonfinal_leidens)} non-final, {len(all_leidens) - len(nonfinal_leidens)} final leidens")

        # Subclustering loop
        for leiden_base in nonfinal_leidens:
            # Subcluster this leiden
            sc.tl.leiden(
                adata,
                resolution=leiden_resolution,
                restrict_to=('leiden_base', [leiden_base]),
                key_added='leiden_temp',
                flavor='leidenalg',
                random_state=42,
            )

            # Helpful filter for this iteration
            filt_leiden_base = adata.obs['leiden_base'] == leiden_base

            ### EXIT CASE: no subclusters were generated, so mark this cluster as FINAL
            initial_subleidens = adata.obs['leiden_temp'][filt_leiden_base].unique()
            if len(initial_subleidens) == 1:
                new_final_cluster = initial_subleidens[0]
                adata.obs.loc[filt_leiden_base, 'leiden_new'] = new_final_cluster
                leiden_is_final[new_final_cluster] = True
                print_verbose(f" -> New FINAL cluster '{new_final_cluster}' (no subclusters generated)")
                continue

            # Iteratively compute xcorrs and combine
            any_recombined = True
            final_cluster_identified = False
            while any_recombined:
                any_recombined = False
                adata_leiden = adata[filt_leiden_base]

                ### EXIT CASE: all clusters combined, so mark remaining cluster as FINAL
                initial_subleidens = adata_leiden.obs['leiden_temp'].unique()
                if len(initial_subleidens) == 1:
                    new_final_cluster = initial_subleidens[0]
                    adata.obs.loc[filt_leiden_base, 'leiden_new'] = new_final_cluster
                    leiden_is_final[new_final_cluster] = True
                    print_verbose(f" -> New FINAL cluster '{new_final_cluster}' (all subclusters recombined)")
                    final_cluster_identified = True
                    break
                
                # Compute xcorrs
                subleidens_grouped = anndatas.get_expr_grouped_by(
                    adata = adata_leiden,
                    groupby = 'leiden_temp',
                    genes = highly_variable_genes,
                    layer = 'CPM_log1p'
                )
                subleiden_means = subleidens_grouped.mean()
                subleidens = subleiden_means.index.values
                subleiden_xcorrs = np.corrcoef(subleiden_means)
                subleiden_xcorrs[np.identity(len(subleidens), dtype=bool)] = 0 # zero the diagonal

                # Get highest-correlated pair of clusters
                max_corr_index = np.unravel_index(np.argmax(subleiden_xcorrs), subleiden_xcorrs.shape)
                max_corr_val = subleiden_xcorrs[max_corr_index]

                # If nobody is correlated enough, exit the loop!
                if max_corr_val < corr_merge_thresh:
                    break

                # ...Otherwise, merge the highest-correlated cluster pair
                any_recombined = True
                subleiden_0 = subleidens[max_corr_index[0]]
                subleiden_1 = subleidens[max_corr_index[1]]
                subleiden_0, subleiden_1 = min(subleiden_0, subleiden_1), max(subleiden_0, subleiden_1)
                adata.obs.loc[adata.obs['leiden_temp'] == subleiden_1, 'leiden_temp'] = subleiden_0
                print_verbose(f" -> Merged clusters '{subleiden_1}' -> '{subleiden_0}' (corr = {max_corr_val})")

            # No final clusters identified; Add contents of leiden_temp to leiden_new and mark clusters as non-final
            if not final_cluster_identified:
                all_leidens_final = False
                adata.obs.loc[filt_leiden_base, 'leiden_new'] = adata.obs.loc[filt_leiden_base, 'leiden_temp']
                for subleiden in adata.obs.loc[filt_leiden_base, 'leiden_temp'].unique():
                    if subleiden in leiden_is_final:
                        raise ValueError(f"leiden '{subleiden}' is somehow already registered as final={leiden_is_final[subleiden]}")
                    leiden_is_final[subleiden] = False
        
        # Move leiden_new to leiden_base
        adata.obs['leiden_base'] = adata.obs['leiden_new'].astype("category")
        hidden_step = adata.obs['leiden_new'].copy()
        hidden_step.name = f"leiden_round_{round_counter}"
        hidden_leiden_steps.append(hidden_step)
    
    hidden_leiden_steps_df = pd.concat(hidden_leiden_steps, axis=1)
    return hidden_leiden_steps_df


def mnn_correct_conditions(
        adata: sc.AnnData,
        *,
        # General settings
        condition_variables: str | Sequence[str],
        k: int,
        sigma: float = 1.0,
        max_mnns: Optional[int] = None,
        random_state: int = 42,
        # MNN space
        mnn_genes: Sequence[str],
        mnn_layer: Optional[str] = None,
        mnn_cosine_normalize: bool,
        mnn_key_added: str = 'mnn_space_corrected',
        # Inquiry space
        inq_layer: Optional[str] = None,
        inq_genes: Sequence[str],
        inq_cosine_normalize: bool,
        inq_key_added: str = 'inquiry_space_corrected',
        # Misc
        verbose: bool = False,
    ):
    """
    Aligns subsets of the provided dataset to a common, cosine-normalized space using mutual
    nearest neighbors (MNNs).

    Registers this aligned space in `adata.obsm[key_added]`.

    Parameters
    ---
    adata : `sc.AnnData`
        The dataset to integrate.
    condition_variables : `str | list[str]`
        A column or list of columns in `adata.obs` by which to separate the expression matrix of
        `adata` into separate conditions for MNN alignment. If two columns are provided, each with two
        conditions, `adata` will be split into at most 2x2 = 4 conditions for sequential alignment.
    k : `int`
        How many nearest neighbors to compute at each step of the batch integration.
    sigma : `float`, default: 1.0
        Gaussian weighting parameter used when computing weights between each unintegrated cell and
        MNN pairs.
    max_mnns : `int`, default: `None`
        Maximum number of mutual nearest neighbor pairs to be used for each batch correction.
        Setting a limit can drastically improve performance.
    mnn_genes : `Sequence[str]`
        A list of genes (members of `adata.var.index`) upon which to compute MNN pairs.
        Typically, highly variable genes are used.
    mnn_layer : `str | Sequence[str]`, default: `None`
        The layer whose expression values to use for MNN computation. If `mnn_cosine_normalize`
        is True, there is no difference between using raw counts and CPM, or any other linear
        per-cell total normalization.
        When not provided, uses `adata.X`.
    mnn_cosine_normalize : `bool`
        Whether to cosine-normalize the MNN space before computing MNNs and performing batch correction.
        This is HIGHLY recommended to be True.
    mnn_key_added : `str`, default: `'mnn_space_corrected'`
        The index into `adata.obsm` in which to store the final corrected MNN space array.
    inq_genes : `Sequence[str]`
        A list of genes (members of `adata.var.index`) to integrate in the "inquiry" space.
        This can be any set of genes of interest, and may optionally overlap with `mnn_genes`.
        MNN pairs computed `mnn_genes` are used to integrate the space defined by genes.
    inq_layer : `str | Sequence[str]`, default: `None`
        The layer whose expression values to use for the inquiry space. If `mnn_cosine_normalize`
        is True, there is no difference between using raw counts and CPM, or any other linear
        per-cell total normalization.
        When not provided, uses `adata.X`.
    inq_cosine_normalize : `bool`
        Whether to cosine-normalize the inquiry space before batch correction.
    inq_key_added : `str`, default: `'mnn_space_corrected'`
        The index into `adata.obsm` in which to store the final corrected inquiry space array.
    verbose : `bool`, default: `False`
        Whether to print information about the batch correction as it is being performed.
    """
    # Check params
    if isinstance(condition_variables, str):
        condition_variables = [condition_variables]
    
    # Get filters for each categorical variable specified
    variable_filters: list[list[np.ndarray]] = []
    variable_values: list[list] = []
    for variable in condition_variables:
        col = adata.obs[variable]
        if col.isna().any():
            raise ValueError(f"Condition variable '{variable}' contains None or NaN values")

        # Get ordering of unique values
        unique_unordered = col.unique()
        if isinstance(col.dtype, pd.CategoricalDtype):
            unique_final = [x for x in col.dtype.categories if x in unique_unordered]
        else:
            warn(f"Condition variable '{variable}' is not categorical and will be batch-corrected in order of appearance!")
            unique_final = unique_unordered
        variable_values.append(unique_final)
        # Generate filters
        filt_list = []
        for unique_val in unique_final:
            filt_list.append(col.values == unique_val)
        variable_filters.append(filt_list)
        print(f"Variable '{variable}': Identified {len(filt_list)} conditions")
    
    # Get final filters as a cross-product of individual variable filters
    final_filters = []
    for filts, filt_var_values in zip(itertools.product(*variable_filters), itertools.product(*variable_values)):
        final_filter = np.logical_and.reduce(filts)
        # Only use filter if any cells match it!
        if final_filter.sum() == 0:
            msg_var_vals = "; ".join(f"{var} == {val}" for var, val in zip(condition_variables, filt_var_values))
            print(f"WARNING: No cells match filter {msg_var_vals}")
        else:
            final_filters.append(final_filter)
    print(f"{len(final_filters)} final filters")

    # Get expression arrays
    mnn_expr_all = anndatas.get_expr_matrix(adata, mnn_genes, layer=mnn_layer, ret_type='numpy')
    inq_expr_all = anndatas.get_expr_matrix(adata, inq_genes, layer=inq_layer, ret_type='numpy')

    # Optionally cosine normalize
    if mnn_cosine_normalize:
        mnn_expr_all = preprocessing.get_cosine_normalized(mnn_expr_all)
    if inq_cosine_normalize:
        inq_expr_all = preprocessing.get_cosine_normalized(inq_expr_all)

    # Get arrays to batch-correct
    mnn_expr_list = []
    inq_expr_list = []
    obs_index_arrays = []
    for filt in final_filters:
        mnn_expr_list.append(mnn_expr_all[filt])
        inq_expr_list.append(inq_expr_all[filt])
        obs_index_arrays.append(adata.obs.index.values[filt])
    concat_obs_index = np.concatenate(obs_index_arrays)

    # Batch-correct in sequence
    mnn_expr_iter = iter(mnn_expr_list)
    inq_expr_iter = iter(inq_expr_list)
    mnn_expr_concat = next(mnn_expr_iter)
    inq_expr_concat = next(inq_expr_iter)
    for mnn_expr_next, inq_expr_next in zip(mnn_expr_iter, inq_expr_iter):
        mnn_expr_concat, inq_expr_concat, = mutual_nearest_neighbors_transform(
            mnn_space_1 = mnn_expr_concat,
            mnn_space_2 = mnn_expr_next,
            inq_space_1 = inq_expr_concat,
            inq_space_2 = inq_expr_next,
            k = k,
            sigma = sigma,
            inplace = False,
            concat = True,
            max_mnns = max_mnns,
            random_state = random_state,
            verbose = verbose,
        )

    # Reorder to original index
    concat_index_to_iindex = {x : i for i, x in enumerate(concat_obs_index)}
    concat_reorder_iindices = adata.obs.index.map(concat_index_to_iindex).values
    mnn_expr_concat = mnn_expr_concat[concat_reorder_iindices].copy()
    inq_expr_concat = inq_expr_concat[concat_reorder_iindices].copy()

    # Add to obsm
    adata.obsm[mnn_key_added] = mnn_expr_concat
    adata.obsm[inq_key_added] = inq_expr_concat


def mutual_nearest_neighbors(arr1: np.ndarray, arr2: np.ndarray, k: int) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Finds indices of mutual nearest neighbors (MNNs) between the two arrays.

    Returns
    ---
    A tuple of lists `(mnn_pairs, mnn_disances)`:
    * `mnn_pairs`: a list of one tuple `(idx1, idx2)` for each MNN pair where 
      `idx1` is the index in `arr1` and `idx2` is the index in `arr2`.
    * `mnn_dists`: A list of distances between the MNN pairs
    """

    # Train neighbor graphs
    knn1 = NearestNeighbors(n_neighbors=k)
    knn2 = NearestNeighbors(n_neighbors=k)
    knn1.fit(arr1)
    knn2.fit(arr2)

    # Get neighbors
    dists1, neigh1 = knn2.kneighbors(arr1, return_distance=True) # Each arr1 point's neighbors (& distances) in arr2
    dists2, neigh2 = knn1.kneighbors(arr2, return_distance=True) # Each arr2 point's neighbors (& distances) in arr1

    # Search!
    mnn_pairs = []
    mnn_dists = []
    neigh1_sets = [set(row) for row in neigh1]
    for idx2 in range(len(neigh2)): # idx2 is an index into arr2/neigh2/dists2
        for idx1, dist in zip(neigh2[idx2], dists2[idx2]): # idx1 is an index into arr1/neigh1/dists1
            # If this arr2 point is present in its arr1 neighbor, MNN pair has been found!
            if idx2 in neigh1_sets[idx1]:
                mnn_pairs.append((idx1, idx2))
                mnn_dists.append(dist)
    
    return mnn_pairs, mnn_dists


def mutual_nearest_neighbors_transform(
        *,
        mnn_space_1: np.ndarray,
        mnn_space_2: np.ndarray,
        inq_space_1: Optional[np.ndarray] = None,
        inq_space_2: Optional[np.ndarray] = None,
        k: int,
        sigma = 1.0,
        inplace = False,
        concat: bool = False,
        min_mnns: int = 10,
        max_mnns: Optional[int] = None,
        random_state: int = 42,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Transforms the datasets `inq_space_2` and `mnn_space_2` to the same spaces as `inq_space_1`
    and `mnn_space_1`, respectively, using the mutual nearest neighbors (MNN) correction algorithm. MNN
    pairs are computed between `mnn_space_1` and `mnn_space_2`.

    Expects `mnn_space_1` and `mnn_space_2` to be cosine-normalized. If they are not,
    the `sigma` parameter may need to be adjusted significantly.
    
    Returns
    ---
    The corrected MNN and inquiry spaces, in a tuple. If `concat == True`, these will be the concatenated
    `_1` and `_2` spaces, otherwise they will just be the corrected `_2` spaces. If no inquiry space was
    provided, returns None for the corrected inquiry space. 
    """
    verbose_print = print if verbose else lambda x: None
    
    # Check parameters
    inq_space_provided = True
    if (inq_space_1 is None) != (inq_space_2 is None):
        raise ValueError("Both or neither of inq_space_1 and inq_space_2 must be provided")
    if inq_space_1 is None:
        inq_space_provided = False
        inq_space_1 = mnn_space_1
        inq_space_2 = mnn_space_2
        verbose_print("No inquiry space provided")
    if not 2 == mnn_space_1.ndim == mnn_space_2.ndim == inq_space_1.ndim == inq_space_2.ndim:
        raise ValueError("Input arrays must be 2-dimensional")
    if len(mnn_space_1) != len(inq_space_1):
        raise ValueError("mnn_space_1 and inq_space_1 must have the same number of rows")
    if len(mnn_space_2) != len(inq_space_2):
        raise ValueError("mnn_space_2 and inq_space_2 must have the same number of rows")
    if mnn_space_1.shape[1] != mnn_space_2.shape[1]:
        raise ValueError("mnn_space_1 and mnn_space_2 must have the same number of columns")
    if inq_space_1.shape[1] != inq_space_2.shape[1]:
        raise ValueError("inq_space_1 and inq_space_2 must have the same number of columns")
    if concat and inplace:
        raise ValueError("`concat` and `inplace` cannot both be True")

    # Copy if not inplace
    if not inplace:
        mnn_space_2 = mnn_space_2.copy()
        inq_space_2 = inq_space_2.copy()

    # Obtain MNNs
    verbose_print(f"Computing MNNs")
    (
        neigh_pairs, # list of tuples of MNN pairs. Element 0 indexes `mnn_space_1` and `inq_space_1`; Element 1 indexes `mnn_space_2` and `inq_space_2`
        neigh_dists, # distances between MNN pairs (not used here)
    ) = mutual_nearest_neighbors(mnn_space_1, mnn_space_2, k)
    npairs = len(neigh_pairs)
    verbose_print(f"Found {npairs} MNN pairs")

    # Ensure some MNNs were found
    if npairs < min_mnns:
        raise ValueError(f"Less than the required number of mutual nearest neighbors were found ({npairs} vs. {min_mnns})")
    
    # Optionally downsample to subset of MNNs
    if max_mnns is not None and npairs > max_mnns:
        verbose_print("Downsampling MNNs")
        # Generate random subset of indices into MNN pairs
        indices = np.arange(npairs)
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)
        indices = indices[:max_mnns]
        # Perform actual downsampling
        neigh_pairs = [neigh_pairs[i] for i in indices]
        neigh_dists = [neigh_dists[i] for i in indices]
        npairs = max_mnns

    # Compute the weights matrix between each MNN pair cell in `mnn_space_2` (rows)
    # and every other cell in `mnn_space_2` (columns)
    verbose_print("Computing weights matrix")
    weights = _mnn_compute_weights(
        arr = mnn_space_2,
        row_indices = [idx_2 for _, idx_2 in neigh_pairs],
        sigma = sigma,
    )

    # Create two vector-difference matrices between dataset 2 and dataset 1,
    # one each for the MNN space and the inquiry space.
    # These matrices have one row per MNN pair.
    verbose_print("Computing vector difference matrix")
    mnn_vectors = _mnn_compute_vectors(
        mnn_pairs = neigh_pairs,
        space_1 = mnn_space_1,
        space_2 = mnn_space_2,
    )
    if inq_space_provided:
        inq_vectors = _mnn_compute_vectors(
            mnn_pairs = neigh_pairs,
            space_1 = inq_space_1,
            space_2 = inq_space_2,
        )

    # Perform batch correction!
    verbose_print("Computing MNN space batch correction")
    _mnn_correct_inplace(
        arr     = mnn_space_2,
        vectors = mnn_vectors,
        weights = weights,
    )
    if inq_space_provided:
        verbose_print("Computing inquiry space batch correction")
        _mnn_correct_inplace(
            arr     = inq_space_2,
            vectors = inq_vectors,
            weights = weights,
        )

    # Optionally concatenate
    if concat:
        mnn_space_ret = np.concatenate([mnn_space_1, mnn_space_2], axis=0)
        inq_space_ret = np.concatenate([inq_space_1, inq_space_2], axis=0) if inq_space_provided else None
    else:
        mnn_space_ret = mnn_space_2
        inq_space_ret = inq_space_2 if inq_space_provided else None

    return mnn_space_ret, inq_space_ret


@njit
def _mnn_compute_weights(arr: np.ndarray, row_indices: list[int], sigma: float):
    """
    Given an NxG matrix of N cells and G genes, and a list of L indices into
    the rows (cells) of said matrix, computes the gaussian weights between
    each cell in the index list and all other cells in the matrix.

    Returns an LxN matrix of weights.
    """
    # Weights matrix has shape (# weights, # rows)
    weights = np.zeros((len(row_indices), len(arr)))

    # Refrain from computing weights for the same cell twice
    computed_weights_idx_mapping: dict[int, int] = dict()

    # Arrays reused each loop iteration
    diff = np.zeros_like(arr)

    for i, row_idx in enumerate(row_indices):
        # If this row_idx was already computed, just reuse it and skip to next
        if row_idx in computed_weights_idx_mapping:
            weights[i] = weights[computed_weights_idx_mapping[row_idx]]
            continue
        else:
            computed_weights_idx_mapping[row_idx] = i

        ### Vector difference between all rows and the selected row
        # diff = to_project - to_project[mnn_proj_idx]
        np.subtract(arr, arr[row_idx], diff)

        ### Euclidean distance
        # distances = (diff ** 2).sum(axis=1) ** 0.5
        diff **= 2
        distances = diff.sum(axis=1)
        distances **= 0.5

        ### Weights computation
        # weights[i] = exp(-0.5 * (distances / sigma)**2)
        distances *= 1.0/sigma
        distances **= 2
        distances *= -0.5
        np.exp(distances, distances)

        # Save weights for this row
        weights[i] = distances
    
    return weights


@njit
def _mnn_compute_vectors(
        *,
        mnn_pairs: Sequence[tuple[int, int]],
        space_1: np.ndarray,
        space_2: np.ndarray,
    ):
    _, ncols = space_1.shape
    vectors = np.zeros((len(mnn_pairs), ncols), dtype=np.float32)
    for i, (idx_1, idx_2) in enumerate(mnn_pairs):
        vectors[i] = space_2[idx_2] - space_1[idx_1]

    return vectors


@njit
def _mnn_correct_inplace(arr: np.ndarray, vectors: np.ndarray, weights: np.ndarray):
    _, ncols = vectors.shape
    vectors_weighted = np.zeros_like(vectors)
    
    for i in range(len(arr)):
        weights_this_row = weights[:, i]
        
        # vector_differences_weighted = (vector_differences.T * weights_this_row).T
        np.multiply(vectors.T, weights_this_row, vectors_weighted.T)
        
        # to_project[proj_idx] -= vector_differences_weighted.sum(axis=0) / weights_this_row
        final_correction = vectors_weighted.sum(axis=0)
        final_correction *= 1.0 / weights_this_row.sum()
        arr[i] -= final_correction
