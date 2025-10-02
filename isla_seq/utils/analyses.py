from typing import Sequence

import numpy as np
import pandas as pd
import scanpy as sc

from .anndatas import get_expr_matrix

def compute_coexpression_matrix(
        adata: sc.AnnData,
        genes: Sequence[str],
        layer: str | None = None,
        min_expr: float = 1,
    ) -> pd.DataFrame:
    """
    Computes a matrix where for each (row, column) gene pair, the contained
    value answers the question "among cells expressing {row}, what proportion
    additionally expresses {column}?"
    """
    expr = get_expr_matrix(adata, genes, layer=layer, ret_type='numpy')
    n_genes = len(genes)

    # Precompute filters
    filters = [
        (expr[:, col] >= min_expr)
        for col in range(n_genes)
    ]

    # Precompute total cells expressing each gene
    gene_sums = [filt.sum() for filt in filters]
    
    # Compute matrix!
    ret_mtx = np.zeros((n_genes, n_genes), dtype=float)
    for idx1, filt1 in zip(range(n_genes), filters):
        ret_mtx[idx1, idx1] = gene_sums[idx1]
        for idx2, filt2 in zip(range(idx1), filters):
            count = (filt1 & filt2).sum()
            ret_mtx[idx1, idx2] = count
            ret_mtx[idx2, idx1] = count
    
    # Normalize
    ret_mtx = (ret_mtx.T / np.array(gene_sums)).T

    return pd.DataFrame(ret_mtx, index=genes, columns=genes)
