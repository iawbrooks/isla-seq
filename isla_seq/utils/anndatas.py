from typing import Optional, Literal, Sequence, overload
from warnings import warn

import numpy as np
import pandas as pd
import pandas.api.typing
import scanpy as sc

import scipy.sparse


@overload
def get_expr_matrix(adata: sc.AnnData, genes: Sequence[str], *, layer: Optional[str] = None, ret_type: Literal['pandas'], copy: bool = True) -> pd.DataFrame: ...
@overload
def get_expr_matrix(adata: sc.AnnData, genes: Sequence[str], *, layer: Optional[str] = None, ret_type: Literal['numpy'], copy: bool = True) -> np.ndarray: ...

def get_expr_matrix(
        adata: sc.AnnData,
        genes: Sequence[str],
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


def get_expr_grouped_by(
        adata: sc.AnnData,
        obs_key: str,
        *,
        genes: Optional[Sequence[str]] = None,
        layer: Optional[str] = None,
    ) -> pandas.api.typing.DataFrameGroupBy:
    """
    Returns a pandas GroupBy object on the specified genes and layer in `adata`,
    with grouping done according to the categorical column name provided in `obs_key`.
    """
    if genes is None:
        genes = adata.var.index.values
    expr_df: pd.DataFrame = get_expr_matrix(adata, genes, layer=layer, ret_type='pandas')
    expr_grouped = expr_df.groupby(adata.obs[obs_key], observed=True)
    return expr_grouped


def sparse_matrix_to_numpy(arr) -> np.ndarray:
    """
    Converts scipy sparse matrices to uncompressed numpy format.
    If a numpy array is passed, simply returns the array.
    """
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        return arr.toarray()
    else:
        raise ValueError(f"Unrecognized type could not be converted from sparse matrix to numpy: '{type(arr)}'")
