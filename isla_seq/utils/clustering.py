import sys

import numpy as np
import pandas as pd
import scanpy as sc

from .anndatas import get_expr_grouped_by



def cluster_recursively(adata: sc.AnnData,
                        *,
                        n_genes: int,
                        corr_merge_thresh: float,
                        leiden_resolution: float,
                        verbose: bool = True,
                        logfile = sys.stdout,
                        ):
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
    gene_meta = sc.pp.highly_variable_genes(adata, layer='CPM_log1p', n_top_genes=n_genes, flavor='seurat', inplace=False)
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
                subleidens_grouped = get_expr_grouped_by(
                    adata = adata_leiden,
                    genes = highly_variable_genes,
                    obs_key = 'leiden_temp',
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
