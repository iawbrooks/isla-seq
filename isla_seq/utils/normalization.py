from __future__ import annotations

from typing import overload
from enum import IntEnum

import numpy as np
import pandas as pd
import scanpy as sc


class ExprNorm(IntEnum):
    """
    Normalization options for scRNA-seq data
    """

    COUNTS = 0
    COSINE = 1
    CPM = 2
    CPM_LOG1P = 3

    def __str__(self):
        match self:
            case ExprNorm.COUNTS:
                return "counts"
            case ExprNorm.COSINE:
                return "cosine"
            case ExprNorm.CPM:
                return "CPM"
            case ExprNorm.CPM_LOG1P:
                return "CPM_log1p"
            case _:
                return str(int(self))
    
    def __eq__(self, other):
        # Helps prevent accidental comparisons to strings when writing code .-.
        if not isinstance(other, ExprNorm):
            raise ValueError(f"Invalid type comparison: ExprNorm == {type(other)}")
        return int(self) == int(other)
    
    def convert(arr: np.ndarray | pd.DataFrame, frm: ExprNorm, to: ExprNorm) -> np.ndarray | pd.DataFrame:
        """
        Convert the normalization of an expression array.

        The only impossible conversion is to `ExprNorm.COUNTS` from any other value;
        all other conversions are allowed.

        Parameters
        arr : 
        """
        ret = arr.copy()
        del arr

        if frm == to:
            return ret

        # Converting to counts can only happen from counts
        if to == ExprNorm.COUNTS:
            raise ValueError("No normalization can recreate counts (except counts)")
        
        # Converting to CPM and cosine can happen from any normalization
        elif to in [ExprNorm.CPM, ExprNorm.COSINE]:
            if frm == ExprNorm.CPM_LOG1P:
                np.power(np.e, ret, out=ret)
                ret -= 1
                frm = ExprNorm.CPM

            if to == ExprNorm.CPM:
                return get_cpm_normalized(ret)
            elif to == ExprNorm.COSINE:
                return get_cosine_normalized(ret)

        # Converting to CPM_log1p can happen from any normalization
        elif to == ExprNorm.CPM_LOG1P:
            if frm != ExprNorm.CPM:
                ret = get_cpm_normalized(ret)
                frm = ExprNorm.CPM

            np.log1p(ret, out=ret)
            return ret
        
        # Shouldn't happen, but check just in case
        else:
            raise ValueError(f"Invalid normalization for parameter `to`: '{to}'")


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
    Copies and cosine-normalizes the rows of `expr`.
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


@overload
def get_cpm_normalized(expr: pd.DataFrame, sum: float = 1e6) -> pd.DataFrame: ...
@overload
def get_cpm_normalized(expr: np.ndarray, sum: float = 1e6) -> np.ndarray: ...

def get_cpm_normalized(expr: pd.DataFrame | np.ndarray, sum: float = 1e6) -> pd.DataFrame | np.ndarray:
    """
    Copies and CPM-normalizes the rows of `expr`.
    That is, each row in `expr` will be linearly scaled such that it sums to `sum`, which is 1e6
    by default.
    """
    # ret = (expr.T / (expr.sum(axis=1) / sum)).T
    ret: np.ndarray = np.empty_like(expr)
    divisor = np.array(expr.sum(axis=1))
    divisor /= sum
    divisor[divisor == 0] = 1 # avoid dividing by zero
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
