from typing import Optional, Literal, Sequence, Iterable, Any, Callable, Self, overload
from warnings import warn

import numpy as np
import pandas as pd
import pandas.api.typing
import scanpy as sc

import scipy.sparse


@overload
def get_expr_matrix(adata: sc.AnnData, genes: str | Sequence[str], *, layer: Optional[str] = None, ret_type: Literal['pandas'], copy: bool = True) -> pd.DataFrame: ...
@overload
def get_expr_matrix(adata: sc.AnnData, genes: str | Sequence[str], *, layer: Optional[str] = None, ret_type: Literal['numpy'], copy: bool = True) -> np.ndarray: ...

def get_expr_matrix(
        adata: sc.AnnData,
        genes: str | Sequence[str],
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

    def __and__(self, other: Self):
        if not isinstance(other, ColumnCondition):
            raise ValueError(f"Logic operations not permitted between ColumnCondition and {type(other)}")
        return ColumnCondition(lambda adata: self.eval(adata) & other.eval(adata))

    def __or__(self, other: Self):
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


class ColumnPath():
    resource: Literal['var', 'obs', 'obsm']
    subpath: tuple[str, ...]
    layer: str | None

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
        
        if resource in ['var', 'obs'] and len(subpath) != 1:
            raise ValueError(
                "When keying from var or obs, one and only one subpath element must be specified: "
                "1) A gene name, or 2) an obs column name, respectively"
            )
        elif resource == 'obsm' and len(subpath) != 2:
            raise ValueError(
                "When keying from obsm, exactly two subpath elements must be specified: "
                "The key of an array in obsm, and the name or index of the column to be accessed in said array"
            )

        # Set members
        self.resource = resource
        self.subpath = subpath
        self.layer = layer

    
    def get(self, adata: sc.AnnData, copy: bool = True) -> pd.Series:
        """
        Get the keyed column from `adata`, optionally copying it.

        The returned Series will have the same index as `adata.obs`.
        """
        # Get data
        match self.resource:
            case 'var':
                gene = self.subpath[0]
                ret = get_expr_matrix(adata, gene, layer=self.layer, copy=False, ret_type='pandas')[gene]
            case 'obs':
                ret = adata.obs[self.subpath[0]]
            case 'obsm':
                arr = adata.obsm[self.subpath[0]]
                if isinstance(arr, pd.DataFrame):
                    ret = arr[self.subpath[1]]
                if isinstance(arr, np.ndarray):
                    try:
                        col_idx = int(self.subpath[1])
                    except ValueError:
                        raise ValueError(
                            f"Indexed element, `adata.obsm[{self.subpath[0]}]`, is a numpy array, "
                            f"but the provided column index is not convertible to int: '{self.subpath[1]}'"
                        )
                    ret = pd.Series(arr[:, col_idx], index=adata.obs.index, copy=False)

        # Optionally copy
        if copy:
            ret = ret.copy()
        
        return ret
    
    def __eq__(self, value: Any) -> ColumnCondition:
        return ColumnCondition(lambda adata: self.get(adata, copy=False) == value)
    
    def __ne__(self, value: Any) -> ColumnCondition:
        return ColumnCondition(lambda adata: self.get(adata, copy=False) != value)
    
    def __lt__(self, value: Any) -> ColumnCondition:
        return ColumnCondition(lambda adata: self.get(adata, copy=False) < value)
    
    def __le__(self, value: Any) -> ColumnCondition:
        return ColumnCondition(lambda adata: self.get(adata, copy=False) <= value)
    
    def __gt__(self, value: Any) -> ColumnCondition:
        return ColumnCondition(lambda adata: self.get(adata, copy=False) > value)
    
    def __ge__(self, value: Any) -> ColumnCondition:
        return ColumnCondition(lambda adata: self.get(adata, copy=False) >= value)
    
    def bool(self) -> ColumnCondition:
        return ColumnCondition(lambda adata: self.get(adata, copy=False).astype(bool))
    
    def isin(self, values: Iterable) -> ColumnCondition:
        try:
            values = values.copy()
        except AttributeError:
            raise ValueError("`values` must be copyable with a .copy() method")
        return ColumnCondition(lambda adata: self.get(adata, copy=False).isin(values))
    
    ###########################
    ### Static Constructors ###
    ###########################

    def var(name: str, layer: str | None = None) -> Self:
        return ColumnPath(f"var::{name}", layer=layer)
    
    gene = var

    def obs(name: str) -> Self:
        return ColumnPath(f"obs::{name}")
    
    def obsm(key: str, index: str) -> Self:
        return ColumnPath(f"obsm::{key}::{index}")

CPath = ColumnPath
