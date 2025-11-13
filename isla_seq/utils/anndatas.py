from __future__ import annotations

from typing import Literal, Sequence, Iterable, Any, Callable, overload
from warnings import warn

import numpy as np
import pandas as pd
import pandas.api.typing
import scanpy as sc
import scipy.sparse

from .arrays import random_filter


def get_layer(adata: sc.AnnData, layer: str | None) -> sc.anndata.typing.ArrayDataStructureType:
    return adata.X if layer is None else adata.layers[layer]


@overload
def get_expr_matrix(adata: sc.AnnData, genes: str | Sequence[str], *, layer: str | None = None, ret_type: Literal['pandas'], copy: bool = True) -> pd.DataFrame: ...
@overload
def get_expr_matrix(adata: sc.AnnData, genes: str | Sequence[str], *, layer: str | None = None, ret_type: Literal['numpy'], copy: bool = True) -> np.ndarray: ...

def get_expr_matrix(
        adata: sc.AnnData,
        genes: str | Sequence[str],
        *,
        layer: str | None = None,
        ret_type: Literal['numpy', 'np', 'pandas', 'pd'] = 'numpy',
        copy: bool = True,
        raise_forcecopy: bool = False,
    ) -> np.ndarray | pd.DataFrame:
    """
    Obtain an expression matrix for the specified gene or genes.

    Parameters
    ---
    adata : `AnnData`
        The AnnData object from which to obtain expression data.
    genes : `str | Sequence[str]`
        A gene or list of genes for which to obtain an expression matrix.
    layer : `str`, optional
        The layer key in `adata.layers` to use. Defaults to using `adata.X` if not provided.
    ret_type : `'numpy' | 'np' | 'pandas' | 'pd'`, default `'numpy'`
        Whether to return a numpy array (`'np' | 'numpy'`) or pandas DataFrame (`'pd' | 'pandas'`).
    copy : `bool`, default `True`
        Whether to copy the data before returning. Will always copy if the underlying array is
        a scipy sparse array.
    raise_forcecopy: `bool`, default `False`
        Whether to raise an error when being forced to copy the underlying data but `copy == False`.
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
    source_arr = get_layer(adata, layer)

    # Slice
    arr = source_arr[:, gene_indices]
    if copy:
        arr = arr.copy()

    # Convert to numpy array
    if isinstance(arr, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        if raise_forcecopy and not copy:
            raise ValueError("copy = False, but a copy must be made because the data is stored as a sparse matrix")
        arr = arr.toarray()

    # Return depending on `ret_type`
    if ret_type in ['pd', 'pandas']:
        return pd.DataFrame(
            data=arr,
            columns=genes,
            index=adata.obs.index.values
        )
    elif ret_type in ['np', 'numpy']:
        return arr
    else:
        raise ValueError(f"Invalid ret_type parameter '{ret_type}'")


def get_expr_df(
        adata: sc.AnnData,
        genes: str | Sequence[str],
        *,
        layer: str | None = None,
        copy: bool = True,
        raise_forcecopy: bool = False,
    ) -> pd.DataFrame:
    """
    Obtain a DataFrame of expression for the specified gene or genes.

    Parameters
    ---
    adata : `AnnData`
        The AnnData object from which to obtain expression data.
    genes : `str | Sequence[str]`
        A gene or list of genes for which to obtain an expression DataFrame.
    layer : `str`, optional
        The layer key in `adata.layers` to use. Defaults to using `adata.X` if not provided.
    copy : `bool`, default `True`
        Whether to copy the data before returning. Will always copy if the underlying array is
        a scipy sparse array.
    raise_forcecopy: `bool`, default `False`
        Whether to raise an error when `copy == False` but a copy must be made of the underlying data.
    """
    return get_expr_matrix(
        adata = adata,
        genes = genes,
        layer = layer,
        ret_type = 'pandas',
        copy = copy,
        raise_forcecopy = raise_forcecopy,
    )


def get_expr_grouped_by(
        adata: sc.AnnData,
        groupby: ColumnPath | str,
        *,
        genes: Sequence[str] | str | None = None,
        layer: str | None = None,
    ) -> pandas.api.typing.DataFrameGroupBy:
    """
    Returns a pandas GroupBy object on the specified expression data.

    Parameters
    ---
    adata : `AnnData`
        The AnnData object from which to group expression data.
    groupby : `ColumnPath | str`
        Either a ColumnPath indexing a categorical column in `adata`, or the string
        name of a categorical column in `adata.obs`.
    genes : `Sequence[str] | str`, optional
        The gene or genes whose expression to group. Defaults to all genes if not provided.
    layer : `str`, optional
        The layer key in `adata.layers` to use. Defaults to using `adata.X` if not provided.
    """
    # Check params
    if genes is None:
        genes = adata.var.index.values
    if isinstance(groupby, str):
        groupby = ColumnPath.obs(groupby)
    
    # Group expression
    expr_df: pd.DataFrame = get_expr_df(adata, genes, layer=layer)
    expr_grouped = expr_df.groupby(groupby.get(adata, copy=False), observed=True)
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


class ColumnCondition():
    lambda_eval: Callable[[sc.AnnData], "pd.Series[bool]"] | None
    invert: bool

    def __init__(
            self,
            lambda_eval: Callable[[sc.AnnData], "pd.Series[bool]"] | None = None,
            *,
            invert: bool = False
        ):
        self.lambda_eval = lambda_eval
        self.invert = invert

    def __and__(self, other: ColumnCondition):
        if not isinstance(other, ColumnCondition):
            raise ValueError(f"Logic operations not permitted between ColumnCondition and {type(other)}")
        return ColumnCondition(lambda adata: self.eval(adata) & other.eval(adata))

    def __or__(self, other: ColumnCondition):
        if not isinstance(other, ColumnCondition):
            raise ValueError(f"Logic operations not permitted between ColumnCondition and {type(other)}")
        return ColumnCondition(lambda adata: self.eval(adata) | other.eval(adata))

    def __invert__(self):
        return ColumnCondition(self.lambda_eval, invert = not self.invert)

    def eval(self, adata: sc.AnnData) -> "pd.Series[bool]":
        """
        Evaluate the condition on an AnnData object, generating a boolean filter
        on `adata.obs`.
        """
        ret = self.lambda_eval(adata)
        if self.invert:
            ret = ~ret
        return ret

    def subset(self, adata: sc.AnnData, copy: bool = False) -> sc.AnnData:
        """
        Get a subset of an AnnData object by filtering for this condition.
        """
        ret = adata[self.eval(adata)]
        if copy:
            ret = ret.copy()
        return ret
    
    def intersection(*conditions: ColumnCondition) -> ColumnCondition:
        """
        Obtain a single condition representing the logical AND (intersection) of a list of conditions.
        """
        if len(conditions) == 0:
            raise ValueError("Must provide at least one condition")
        elif len(conditions) == 1 and isinstance(conditions[0], (list, set)):
            conditions = tuple(conditions[0])
        cond = conditions[0]
        for x in conditions[1:]:
            cond = cond & x
        return cond
    
    def union(*conditions: ColumnCondition) -> ColumnCondition:
        """
        Obtain a single condition representing the logical OR (union) of a list of conditions.
        """
        if len(conditions) == 0:
            raise ValueError("Must provide at least one condition")
        elif len(conditions) == 1 and isinstance(conditions[0], (list, set)):
            conditions = tuple(conditions[0])
        cond = conditions[0]
        for x in conditions[1:]:
            cond = cond | x
        return cond

    @property
    def FALSE() -> ColumnCondition:
        return ColumnCondition(lambda adata: pd.Series(False, index=adata.obs.index))

    @property
    def TRUE() -> ColumnCondition:
        return ColumnCondition(lambda adata: pd.Series(True, index=adata.obs.index))

    def random(prop: float = 0.5, rng: int | np.random.RandomState | None = None, same: bool = True) -> ColumnCondition:
        """
        Sample a random subopulation.

        Parameters
        ---
        prop : `float`, between 0 and 1, inclusive
            The proportion of samples to subset. 
        rng : `int` | `RandomState` | `None`, default: `None`
            An integer seed or RandomState object to use for randomly subsetting an AnnData. If a
            RandomState object is provided, a copy will be made such that the passed object is
            not modified.
        same : `bool`, default: `True`
            Whether to generate the same subset each time the ColumnCondition is used. When True,
            always produces the exact same pseudorandom sample. Otherwise, repeated calls of `.eval()`
            or `.subset()` will generate different, but still predictable (according to the `rng`
            parameter) pseudorandom subsets.
        """
        # Create reusable random number generator object
        if isinstance(rng, np.random.RandomState):
            state = rng.get_state()
            rng = np.random.RandomState()
            rng.set_state(state)
        else:
            rng = np.random.RandomState(rng)
        
        # Create random filter generator function
        def rand(adata: sc.AnnData) -> "pd.Series[bool]":
            if same:
                use_rng = np.random.RandomState()
                use_rng.set_state(rng.get_state())
            else:
                use_rng = rng

            return pd.Series(random_filter(len(adata), prop, use_rng), index=adata.obs.index)

        return ColumnCondition(rand)


class ColumnPath():
    _lambda_eval: Callable[[sc.AnnData], pd.Series]

    _NONEPATH = "::NONEPATH::"

    def __init__(
            self,
            path: str,
            *,
            layer: str | None = None,
        ):
        """
        Index a column in an AnnData object.

        The format of the provided path must be either the name of an index in `var`, or of the
        format `"{resource}::{subpath}"`, where:
        * `resource` is one of `'var'`, `'obs'`, or `'obsm'`
        * `subpath` is the name of an index in `var` or a column in `obs`, or is of the format
          `"{key}::{index}"`, where `key` is the name of an array or DataFrame in `obsm`, and
          `index` is the numerical index or string name of a column of that array or DataFrame.
        
        Examples
        ---
        For an AnnData object, `adata`:
        ```
        # Indexes the first column of adata.obsm['X_pca']
        cpath = ColumnPath("obsm::X_pca::0")

        # Indexes adata.layers["counts"][:, adata.var.index.get_loc("Cxcl14")]
        cpath = ColumnPath("Cxcl14", layer="counts")
        cpath = ColumnPath("var::Cxcl14", layer="counts")

        # Indexes adata.obs["leiden"]
        cpath = ColumnPath("obs::leiden")
        ```
        """
        # If NONEPATH was provided, assume everything else will be handled by the wrapping function
        if path == ColumnPath._NONEPATH:
            return

        # Parse path
        items = tuple(path.split('::'))
        resource = items[0]
        subpath = items[1:]

        # Check params
        if len(subpath) == 0:
            subpath = (resource, )
            resource = 'var'
        elif resource not in ['var', 'obs', 'obsm']:
            raise ValueError(f"Invalid resource '{resource}'")
        
        if layer is not None and resource != 'var':
            raise ValueError("When `layer` is specified, the resource must be 'var'")
        
        # Set the evaluation lambda
        match resource:
            case 'var':
                if len(subpath) != 1:
                    raise ValueError(
                        "When keying from var or obs, one and only one subpath element must be specified: "
                        "1) A gene name, or 2) an obs column name, respectively"
                    )
                self._lambda_eval = self._get_var_lambda(gene=subpath[0], layer=layer)
            case 'obs':
                if len(subpath) != 1:
                    raise ValueError(
                        "When keying from var or obs, one and only one subpath element must be specified: "
                        "1) A gene name, or 2) an obs column name, respectively"
                    )
                self._lambda_eval = self._get_obs_lambda(col=subpath[0])
            case 'obsm':
                if len(subpath) != 2:
                    raise ValueError(
                        "When keying from obsm, exactly two subpath elements must be specified: "
                        "The key of an array in obsm, and the name or index of the column to be accessed in said array"
                    )
                self._lambda_eval = self._get_obsm_lambda(key=subpath[0], col_idx=subpath[1])

    def _get_var_lambda(self, gene: str, layer: str | None = None):
        return lambda adata: get_expr_matrix(adata, gene, layer=layer, copy=False, ret_type='pandas')[gene]

    def _get_obs_lambda(self, col: str):
        return lambda adata: adata.obs[col]

    def _get_obsm_lambda(self, key: str, col_idx: str | int):
        def get_obsm(adata: sc.AnnData, key: str, col_idx: str | int):
            arr = adata.obsm[key]
            if isinstance(arr, pd.DataFrame):
                return arr[col_idx]
            elif isinstance(arr, np.ndarray):
                try:
                    col_idx = int(col_idx)
                except ValueError:
                    raise ValueError(
                        f"Indexed element, `adata.obsm[{key}]`, is a numpy array, "
                        f"but the provided column index is not convertible to int: '{col_idx}'"
                    )
                return pd.Series(arr[:, col_idx], index=adata.obs.index, copy=False)
            else:
                raise ValueError(f"Invalid type of indexed array '{type(arr)}'")
        return lambda adata: get_obsm(adata, key, col_idx)

    def get(self, adata: sc.AnnData, copy: bool = True) -> pd.Series:
        """
        Get the keyed data from `adata`, optionally copying it.

        The returned Series will have the same index as `adata.obs`.
        """
        # Get data
        ret = self._lambda_eval(adata)

        # Optionally copy
        if copy:
            ret = ret.copy()
        
        return ret
    
    def __eq__(self, value: Any) -> ColumnCondition:
        if isinstance(value, ColumnPath):
            return ColumnCondition(lambda adata: self.get(adata, copy=False) == value.get(adata, copy=False))
        else:
            return ColumnCondition(lambda adata: self.get(adata, copy=False) == value)
    
    def __ne__(self, value: Any) -> ColumnCondition:
        if isinstance(value, ColumnPath):
            return ColumnCondition(lambda adata: self.get(adata, copy=False) != value.get(adata, copy=False))
        else:
            return ColumnCondition(lambda adata: self.get(adata, copy=False) != value)
    
    def __lt__(self, value: Any) -> ColumnCondition:
        if isinstance(value, ColumnPath):
            return ColumnCondition(lambda adata: self.get(adata, copy=False) < value.get(adata, copy=False))
        else:
            return ColumnCondition(lambda adata: self.get(adata, copy=False) < value)
    
    def __le__(self, value: Any) -> ColumnCondition:
        if isinstance(value, ColumnPath):
            return ColumnCondition(lambda adata: self.get(adata, copy=False) <= value.get(adata, copy=False))
        else:
            return ColumnCondition(lambda adata: self.get(adata, copy=False) <= value)
    
    def __gt__(self, value: Any) -> ColumnCondition:
        if isinstance(value, ColumnPath):
            return ColumnCondition(lambda adata: self.get(adata, copy=False) > value.get(adata, copy=False))
        else:
            return ColumnCondition(lambda adata: self.get(adata, copy=False) > value)
    
    def __ge__(self, value: Any) -> ColumnCondition:
        if isinstance(value, ColumnPath):
            return ColumnCondition(lambda adata: self.get(adata, copy=False) >= value.get(adata, copy=False))
        else:
            return ColumnCondition(lambda adata: self.get(adata, copy=False) >= value)
    
    def bool(self) -> ColumnCondition:
        return ColumnCondition(lambda adata: self.get(adata, copy=False).astype(bool))
    
    def isin(self, values: Iterable) -> ColumnCondition:
        values = set(values)
        return ColumnCondition(lambda adata: self.get(adata, copy=False).isin(values))
    
    ###########################
    ### Static Constructors ###
    ###########################

    def _blank() -> ColumnPath:
        return ColumnPath(ColumnPath._NONEPATH)

    def var(name: str, layer: str | None = None) -> ColumnPath:
        return ColumnPath(f"var::{name}", layer=layer)
    
    gene = var

    def obs(name: str) -> ColumnPath:
        return ColumnPath(f"obs::{name}")
    
    def obsm(key: str, index: str) -> ColumnPath:
        return ColumnPath(f"obsm::{key}::{index}")
    
    def sum(genes: list[str] | None = None, layer: str | None = None) -> ColumnPath:
        cpath: ColumnPath = ColumnPath._blank()
        if genes is None:
            cpath._lambda_eval = lambda adata: np.asarray(get_layer(adata, layer).sum(axis=1)).reshape(-1)
        else:
            cpath._lambda_eval = lambda adata: get_expr_df(adata, genes, layer=layer, copy=False).sum(axis=1)
        return cpath


def var(name: str, layer: str | None = None) -> ColumnPath:
    return ColumnPath.var(name=name, layer=layer)

gene = var

def obs(name: str) -> ColumnPath:
    return ColumnPath.obs(name=name)

def obsm(key: str, index: str) -> ColumnPath:
    return ColumnPath.obsm(key=key, index=index)
