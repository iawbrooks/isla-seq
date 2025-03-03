from typing import overload

import numpy as np
import pandas as pd
import scanpy as sc


def generate_normalized_layers(
        adata: sc.AnnData,
        *,
        scale_max_std: float = 10.0,
        include_unitvar: bool = False,
    ):
    """
    Normalize counts for clustering and further processing. Assumes adata.X is in raw counts.

    Modifies `adata` in-place, but saves a complete copy of each level
    of processing in the layers `"counts"`, `"CPM"`, `"CPM_log1p"`, and `"CPM_log1p_unitvar"`.

    You can optionally exclude unit-variance log-CPM computation by setting the
    `include_unitvar` parameter to `False`. This is recommended when working with
    massive datasets in sparse matrix format, as the resulting array cannot be
    stored in sparse matrix format and could therefore require a substantial
    amount of memory.
    """
    if 'counts' in adata.layers:
        print("counts already saved")
    else:
        adata.layers['counts'] = adata.X.copy()

    # CPM
    if 'CPM' in adata.layers:
        print('CPM already saved')
    else:
        adata.X = adata.layers['counts'].copy()
        sc.pp.normalize_total(adata, target_sum=1e6)
        adata.layers['CPM'] = adata.X.copy()

    # Log(CPM+1)
    if 'CPM_log1p' in adata.layers:
        print('CPM_log1p already saved')
    else:
        adata.X = adata.layers['CPM'].copy()
        sc.pp.log1p(adata)
        adata.layers['CPM_log1p'] = adata.X.copy()

    # Unit variance scaling
    if include_unitvar:
        if 'CPM_log1p_unitvar' in adata.layers:
            print('CPM_log1p_unitvar already saved')
        else:
            adata.X = adata.layers['CPM_log1p'].copy()
            sc.pp.scale(adata, max_value=scale_max_std)
            adata.layers['CPM_log1p_unitvar'] = adata.X.copy()


@overload
def get_cosine_normalized(expr: pd.DataFrame) -> pd.DataFrame: ...
@overload
def get_cosine_normalized(expr: np.ndarray) -> np.ndarray: ...

def get_cosine_normalized(expr: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """
    Cosine-normalizes the rows of `expr`.
    That is, each row in `expr` will be linearly scaled such that the sum of the squares of the
    values in any given row will sum to one.
    """
    # ret = (expr.T / np.sqrt(np.power(expr, 2).sum(axis=1))).T
    ret: np.ndarray = np.empty_like(expr)
    np.square(expr, out=ret)
    divisor = ret.sum(axis=1)
    divisor[divisor == 0] = 1 # avoid dividing by zero
    np.sqrt(divisor, out=divisor)
    np.divide(expr.T, divisor, out=ret.T)

    if isinstance(expr, pd.DataFrame):
        return pd.DataFrame(ret, index=expr.index, columns=expr.columns)
    else:
        return ret


def get_unitvar_normalized(expr: np.ndarray) -> np.ndarray:
    """
    Normalizes `expr` such that columns are z-scored.
    """
    ret = expr.copy()
    ret -= ret.mean(axis=0)
    std = ret.std(axis=0)
    std[std == 0] = np.inf
    ret /= std
    return ret
