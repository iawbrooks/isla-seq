from typing import Optional, Literal, overload
from warnings import warn

import numpy as np
import pandas as pd
import scanpy as sc

import scipy.sparse


@overload
def get_expr_matrix(adata: sc.AnnData, genes: list[str], *, layer: Optional[str] = None, ret_type: Literal['pandas'], copy: bool = True) -> pd.DataFrame: ...
@overload
def get_expr_matrix(adata: sc.AnnData, genes: list[str], *, layer: Optional[str] = None, ret_type: Literal['numpy'], copy: bool = True) -> np.ndarray: ...

def get_expr_matrix(
        adata: sc.AnnData,
        genes: list[str],
        *,
        layer: Optional[str] = None,
        ret_type: Literal['pandas', 'numpy'] = 'numpy',
        copy: bool = True,
    ) -> np.ndarray | pd.DataFrame:
    """
    Obtain an expression matrix for the specified genes.
    """
    if isinstance(genes, str):
        genes = [genes]
    
    # Construct index array
    gene_indices = np.full(len(genes), np.iinfo(np.int32).max)
    for i, gene in enumerate(genes):
        gene_index = adata.var.index.get_loc(gene)
        if not isinstance(gene_index, (int, np.int32, np.int64)):
            raise ValueError(f"Non-unique key '{gene}' results in more than one loc index: {gene_index}")
        gene_indices[i] = gene_index

    # Determine source array
    source_arr = adata.X if layer is None else adata.layers[layer]

    # Slice
    arr = source_arr[:, gene_indices]
    if copy:
        arr = arr.copy()

    # Convert to numpy array
    if isinstance(arr, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        if not copy:
            warn("copy = False, but a copy must be made because adata.X is a sparse matrix")
        arr = arr.toarray()

    # Return depending on `ret_type`
    if ret_type == 'pandas':
        return pd.DataFrame(
            data=arr,
            columns=genes,
            index=adata.obs.index.values
        )
    elif ret_type == 'numpy':
        return arr
