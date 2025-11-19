from __future__ import annotations

from typing import Sequence, Generator
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import yaml
from pydeseq2.ds import DeseqDataSet, DeseqStats
from sklearn.neighbors import KNeighborsRegressor

from .. import internal

from .anndatas import ColumnCondition, ColumnPath, Literal, get_expr_df, get_expr_matrix

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


def impute_genes(
        adata_ref: sc.AnnData,
        adata_query: sc.AnnData,
        *,
        k: int = 5,
        layer_knn: str | None,
        layer_impute: str | None,
        knn_genes: list[str] | None = None,
        impute_genes: list[str] | None = None,
        knn_metric: Literal['cosine', 'euclidean'] = 'cosine',
        knn_weights: Literal['uniform', 'distance'] = 'uniform',
    ) -> pd.DataFrame:
    """
    Compute the nearest-neighbors-predicted expression of genes in a query dataset
    based off of a reference dataset.

    Parameters
    ---
    adata_ref : `AnnData`
        The reference dataset from which to impute expression,
    adata_query : `AnnData`
        The querying dataset for whose cells expression values will be generated.
    k : `int`, default 5
        The number of nearest neighbors to compute in the reference dataset for each
        cell in the query dataset.
    layer_knn : `str | None`
        The layer key to use for computing nearest neighbors between the two datasets.
        If not provided, defaults to using expression data stored in `.X`.
    layer_impute: `str | None`
        The layer key to use for imputing expression.
        If not provided, defaults to using expression data stored in `.X`.
    knn_genes : `list[str] | None`, default `None`
        List of genes to use for neighbor computation. Defaults to all shared genes.
    impute_genes: `list[str] | None`, default `None`
        List of genes for which to impute expression. Defaults to all genes in `adata_ref`.
    knn_metric: `Literal['cosine', 'euclidean']`, default `'cosine'`
        Controls how distances are computed when finding nearest neighbors. Supplied as the
        `metric` argument in the `sklearn.neighbors.KNeighborsRegressor` constructor.
    knn_weights: `Literal['uniform', 'distance']`, default `'uniform'`
        Controls how neighbors are weighted when imputing expression. Supplied as the
        `weights` argument in the `sklearn.neighbors.KNeighborsRegressor` constructor.
    """
    # Check params
    if knn_genes is None:
        knn_genes = list(adata_query.var.index.intersection(adata_ref.var.index))
    if impute_genes is None:
        impute_genes = list(adata_ref.var.index)
    
    # Get necessary expression data
    arr_ref_impute = get_expr_matrix(adata_ref,   impute_genes, layer=layer_impute, copy=False)
    arr_ref_knn    = get_expr_matrix(adata_ref,   knn_genes,    layer=layer_knn,    copy=False)
    arr_query_knn  = get_expr_matrix(adata_query, knn_genes,    layer=layer_knn,    copy=False)

    # Impute!
    knn = KNeighborsRegressor(k, metric=knn_metric, weights=knn_weights)
    knn.fit(arr_ref_knn, arr_ref_impute)
    imputed_arr = knn.predict(arr_query_knn)

    return pd.DataFrame(
        imputed_arr,
        index = adata_query.obs.index,
        columns = impute_genes,
    )


###############################
### Differential Expression ###
###############################


class DiffexpResult():
    results: pd.DataFrame
    var: pd.DataFrame

    title: str
    name_1: str
    name_2: str

    def __init__(
            self,
            results: pd.DataFrame,
            title: str,
            name_1: str,
            name_2: str,
            *,
            var: pd.DataFrame | None = None,
        ):
        """
        
        """
        # Check params
        if var is None:
            var = pd.DataFrame(index=results.index)
        if not var.index.equals(results.index):
            raise ValueError("Index of `results` and `var` must match")

        # Set members
        self.results = results
        self.var = var
        self.title = title
        self.name_1 = name_1
        self.name_2 = name_2


    @property
    def filestem(self) -> str:
        return f"{self.title}_stats"


    def save_results(
            self,
            output_dir: Path = Path('./'),
            *,
            include_var: bool = False
        ):
        """
        Saves the complete results DataFrame, optionally concatenated with `self.var`.
        """
        results = self.results
        if include_var:
            results = pd.concat([results, self.var], axis=1)
        
        results.to_csv(output_dir / f"{self.filestem}.csv")


    def save_results_genes(
            self,
            genes: list[str],
            output_dir: Path = Path('./'),
            suffix: str = "genes",
        ):
        """
        Saves a results DataFrame of the specified genes under the name
        `"{self.title}_stats_{suffix}.csv"`, in the output directory.
        """
        save_df = self.results.loc[genes]
        save_df.to_csv(output_dir / f"{self.title}_stats_{suffix}.csv")


    def save_results_significance(
            self,
            significance_level: float,
            output_dir: Path = Path('./'),
            pval_column: Literal['padj', 'pvalue'] = 'padj', 
            suffix: str | None = None
        ):
        """
        Saves a results DataFrame of significantly differentially expressed
        genes under the name `"{self.title}_stats_{suffix}.csv"`, in the output
        directory. Will automatically construct a `suffix` by the format
        `"{pval_column}<={significance_level}"` if not provided.
        """
        if suffix is None:
            suffix = f"{pval_column}<={significance_level:g}"

        save_df = self.results[self.results[pval_column] <= significance_level]
        save_df.to_csv(output_dir / f"{self.filestem}_{suffix}.csv")


    def save_bokeh(
            self,
            output_dir: Path | str,
            *,
            # Scatterplot settings
            min_size: float = 3.0,
            max_size: float = 9.0,
            l2fc_range_size = 2.0,
            l2fc_range_color = 1.5,
            cmap: str | matplotlib.colors.Colormap = 'coolwarm',
            hover_tooltips: list[tuple[str, str]] | None = None,

            # Table settings
            table_addl_data: pd.DataFrame | None = None,
            table_columns: list | None = None,
            table_sortby_columns: list[str] | None = None,
        ):
        """
        Save a bokeh plot as an HTML file in the output directory.

        Parameters
        ---
        min_size : float
            Minimum point size in the scatterplot.
        max_size : float
            Maximum point size in the scatterplot.
        l2fc_range_size : float
            The range of log2 fold change values to map to point sizes.
            Specifically, a point with |log2FoldChange| >= `l2fc_range_size`
            will be drawn at `max_size`, and a point with log2FoldChange = 0
            will be drawn at `min_size`.
        l2fc_range_color : float
            The range of log2 fold change values to map to point colors.
            Specifically, a point with log2FoldChange <= -`l2fc_range_color`
            will be drawn with the first color in the colormap, a point with
            log2FoldChange >= `l2fc_range_color` will be drawn with the last
            color in the colormap, and a point with log2FoldChange = 0 will
            be drawn with the middle color in the colormap.
        cmap : str | matplotlib.colors.Colormap
            Colormap to use for point colors. If a string is provided, it
            should be a valid matplotlib colormap name.
        hover_tooltips : list[tuple[str, str]] | None
            List of (label, value) tuples to show in the hover tooltip. If None,
            defaults to showing gene name, p-value, and mean CPM in each condition.
            Value strings should be in the format accepted by bokeh's HoverTool,
            e.g. the tuple `("Gene", "@gene"` to show the value of `column_name` in a
            ColumnDataSource.
        """
        try:
            import bokeh.plotting
            import bokeh.layouts
            from bokeh.models import ColumnDataSource, DataTable, TableColumn, StringFormatter, NumberFormatter, ScientificFormatter, WheelZoomTool, Column
        except ImportError:
            raise ImportError("Bokeh not found; please install it to run `save_bokeh`")

        # Check params
        output_dir = Path(output_dir)

        # Add basic info to results
        res_df = self.results.copy()
        res_df = res_df[~res_df['log2FoldChange'].isna()]
        res_df.loc[res_df['padj'].isna(), 'padj'] = 1.0
        res_df['-log10padj'] = -np.log10(res_df['padj'])
        res_df.loc[res_df['-log10padj'] == np.inf, '-log10padj'] = -np.log10(np.finfo(np.float64).tiny) # cheaty thing for when pval == 0
        res_df['gene'] = res_df.index

        # Compute point colors
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        cdata = res_df['log2FoldChange'].to_numpy(copy=True)
        cdata = (cdata + l2fc_range_color) / (l2fc_range_color * 2)
        cdata = np.clip(cdata, 0, 1)
        res_df['color'] = [tuple(x) for x in (cmap(cdata) * 255.99).astype(np.uint8)]
        # Compute point sizes
        sdata = np.abs(res_df['log2FoldChange'].values)
        sdata /= l2fc_range_size
        sdata[sdata > 1] = 1
        sdata *= (max_size - min_size)
        sdata += min_size
        res_df['size'] = sdata

        # Add optional info to res_df
        if table_addl_data is not None:
            if len(res_df.index.intersection(table_addl_data.index)) != len(res_df.index):
                raise ValueError("The supplied additional data is missing entries for at least one gene")
            res_df.drop(columns=res_df.columns.intersection(table_addl_data.columns), inplace=True)
            res_df = pd.concat([res_df, table_addl_data.loc[res_df.index]], axis=1)
        res_df = pd.concat([res_df, self.var.loc[res_df.index].drop(columns=res_df.columns.intersection(self.var.columns))], axis=1)
            
        # Create final source
        if table_sortby_columns is None:
            table_sortby_columns = ['log2FoldChange']
        res_df.sort_values(table_sortby_columns, inplace=True)
        source = ColumnDataSource(res_df)

        # Make plot!
        TOOLS="hover,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,examine,help"
        p = bokeh.plotting.figure(tools=TOOLS, width=800, height=800)
        p.toolbar.active_scroll = p.select_one(WheelZoomTool)
        p.title = f"Differential Expression -- {self.name_2} vs. {self.name_1}"
        p.background_fill_color = "#CCCCCC"
        p.axis.axis_label_text_font_size = '14pt'
        p.axis.axis_label_text_font_style = 'bold'
        p.axis.axis_label_text_color = (0, 0, 0)
        p.yaxis.axis_label = f"{self.name_2} Enrichment Index\n(Log2 Fold Change)"
        p.xaxis.axis_label = "Confidence (-log10 pval)"
        p.scatter(
            x="-log10padj",
            y="log2FoldChange",
            fill_color="color",
            fill_alpha=0.6,
            line_color=None,
            size="size",
            source=source
        )
        if hover_tooltips is None:
            hover_tooltips = [
                ("Gene", "@gene"),
                ("P-value", "@padj"),
                (f"{self.name_1} CPM", f"@{{mean_CPM_{self.name_1}}}"),
                (f"{self.name_2} CPM", f"@{{mean_CPM_{self.name_2}}}"),
            ]
        p.hover.tooltips = hover_tooltips

        # Make table!
        if table_columns is None:
            table_columns = [
                TableColumn(width=120, field="gene", title="Gene", formatter=StringFormatter(font_style="bold")),
                TableColumn(width=300, field="log2FoldChange", title=f"{self.name_2} Enrichment Index", formatter=NumberFormatter(font_style="bold", format="0.000", text_color="color")),
                TableColumn(width=150, field="padj", title="P Value", formatter=ScientificFormatter(font_style="bold", precision=4)),
                TableColumn(width=300, field=f"mean_CPM_{self.name_1}", title=f"{self.name_1} CPM", formatter=NumberFormatter(font_style="bold", format="0.00")),
                TableColumn(width=300, field=f"mean_CPM_{self.name_2}", title=f"{self.name_2} CPM", formatter=NumberFormatter(font_style="bold", format="0.00")),
            ]
        data_table = DataTable(source=source, columns=table_columns, editable=False, width=900, height=800, index_position=None)

        # Make final figure
        show_col = bokeh.layouts.row(p, Column(width=100), data_table)
        bokeh.plotting.save(
            show_col,
            output_dir / f"{self.filestem}.html",
            title=f"Differential Expression - {self.name_1} vs. {self.name_2}"
        )



class Diffexp:
    # Init parameters
    title: str
    sources: list[sc.AnnData] | None
    condition_1: ColumnCondition | None
    condition_2: ColumnCondition | None

    # Computed in the constructor
    genes: list[str]
    name_1: str | None
    name_2: str | None
    var: pd.DataFrame


    def __init__(
            self,
            title: str,
            sources: sc.AnnData | list[sc.AnnData],
            *,
            condition_1: ColumnCondition,
            condition_2: ColumnCondition,
            name_1: str,
            name_2: str,
            var: pd.DataFrame | None = None,
        ):
        """
        Parameters
        ---
        title : str
            Title for this analysis, used in output filenames.
        sources : AnnData | list[AnnData]
            One or more AnnData objects to draw cells from.
        condition_1 : ColumnCondition | None
            Condition defining the first population of cells to compare.
        condition_2 : ColumnCondition | None
            Condition defining the second population of cells to compare.
        name_1 : str | None
            Name for the first condition.
        name_2 : str | None
            Name for the second condition.
        var : pd.DataFrame | None
            A DataFrame of gene metadata, indexed by gene name. If None, will be
            generated automatically by taking the intersection of all `sources`
            var DataFrames and retaining only columns that are present across
            all `sources`.
        """
        # Check params
        if isinstance(sources, sc.AnnData):
            sources = [sources]

        # Set params
        self.title       = title
        self.sources     = list(sources)
        self.condition_1 = condition_1
        self.condition_2 = condition_2

        self.results_df = None

        # Compute set of common genes
        gene_set = set(sources[0].var.index)
        for adata in sources[1:]:
            gene_set &= set(adata.var.index)
        self.genes = [x for x in sources[0].var.index if x in gene_set]

        # Check condition names
        if name_1 == name_2:
            raise ValueError(f"Conditions have the same name '{self.name_1}'")
        self.name_1 = name_1
        self.name_2 = name_2
    
        # Generate common 'var' DataFrame
        if var is None:
            var_cols = sources[0].var.columns
            var_cols_final = []
            for col in var_cols:
                for adata in sources[1:]:
                    if not (col in adata.var and sources[0].var[col].equals(adata.var[col])):
                        break
                else:
                    var_cols_final.append(col)
            var = sources[0].var.loc[self.genes, var_cols_final].copy()
        else:
            var = var.loc[self.genes]
        self.var = var


    def get_condition_expr(self, which_condition: Literal[1, 2]) -> pd.DataFrame:
        """
        Returns a DataFrame of raw counts for all cells matching the specified condition.
        """
        if which_condition == 1:
            condition = self.condition_1
        elif which_condition == 2:
            condition = self.condition_2
        else:
            raise ValueError("Selected condition must be `1` or `2`")
        
        exprs = []
        for adata in self.sources:
            expr = get_expr_df(condition.subset(adata), self.genes, layer='counts')
            exprs.append(expr)

        return pd.concat(exprs, axis=0)


    def get_combined_adata(self, *, enforce_mutual_exclusivity: bool = True) -> sc.AnnData:
        """
        Constructs and returns a combined AnnData object containing all cells
        from both conditions.

        Parameters
        ---
        enforce_mutual_exclusivity : bool
            If True, raises an error if any cell is found to be present in both conditions.
        """
        # adata.X
        expr_1 = self.get_condition_expr(1)
        expr_2 = self.get_condition_expr(2)

        if enforce_mutual_exclusivity and len(expr_1.index.intersection(expr_2.index)) != 0:
            raise ValueError("The two conditions are not mutually exclusive")
        
        expr_concat = pd.concat([expr_1, expr_2])

        # adata.obs
        obs = pd.DataFrame(index=expr_concat.index, columns=['condition'])
        obs.loc[expr_1.index, 'condition'] = self.name_1
        obs.loc[expr_2.index, 'condition'] = self.name_2

        # adata.var
        var = pd.DataFrame(index=expr_concat.columns)

        return sc.AnnData(
            X = expr_concat,
            obs = obs,
            var = var
        )


    def run_pydeseq2(self, quiet: bool = False) -> DiffexpResult:
        """
        Runs the full PyDESeq2 workflow and returns the statistical results.
        """
        # Run PyDESeq2
        adata = self.get_combined_adata()
        dds = DeseqDataSet(adata = adata, design = "~condition", quiet = quiet)
        dds.deseq2()
        ds = DeseqStats(dds, contrast=['condition', self.name_2, self.name_1], quiet = quiet)
        ds.summary()

        # Add helpful info
        res_df = ds.results_df.copy()
        res_df = res_df.loc[adata.var.index]
        res_df[f'mean_counts_{self.name_1}'] = adata.X[(adata.obs['condition'] == self.name_1).values].mean(axis=0)
        res_df[f'mean_counts_{self.name_2}'] = adata.X[(adata.obs['condition'] == self.name_2).values].mean(axis=0)
        sc.pp.normalize_total(adata, target_sum=1e6)
        res_df[f'mean_CPM_{self.name_1}'] = adata.X[(adata.obs['condition'] == self.name_1).values].mean(axis=0)
        res_df[f'mean_CPM_{self.name_2}'] = adata.X[(adata.obs['condition'] == self.name_2).values].mean(axis=0)
        
        return DiffexpResult(
            res_df,
            title = self.title,
            name_1 = self.name_1,
            name_2 = self.name_2,
            var = self.var,
        )


    def _get_condition_intersection_from_filters(filters: dict) -> ColumnCondition:
        """
        Create a ColumnCondition from the intersection of a set of filters.

        The `filters` dictionary's key-value pairs will be used to construct a
        series of `ColumnCondition` filters as follows:
        * **key**: If it contains a double-colon (`::`), it will be interpreted as
            a `ColumnPath`-indexed resource. Otherwise, it will be interpreted as a
            column in `obs`.
        * **value**: If a list or set, any contained element will be acceptable
            for the filter. Otherwise, the resource indexed by `key` must match
            `value`.
        """
        conditions = []
        for key, val in filters.items():
            # Get path to resource
            if '::' in key:
                cpath = ColumnPath(key)
            else:
                cpath = ColumnPath.obs(key)

            # Construct condition
            if isinstance(val, (list, set)):
                ccond = cpath.isin(val)
            else:
                ccond = cpath == val
            
            conditions.append(ccond)
        
        return ColumnCondition.intersection(conditions)


    def generate_condition_from_dict(cond_dict: dict | list[dict]) -> ColumnCondition:
        """
        For a condition dictionary loaded from a correctly formatted YAML file of conditions,
        constructs a ColumnCondition that applies the filters underlying said conditions.
        """
        if isinstance(cond_dict, dict):
            cond_dict = [cond_dict]
        
        conds_processed = [
            Diffexp._get_condition_intersection_from_filters(cond['filters'])
            for cond in cond_dict
        ]
        
        return ColumnCondition.union(*conds_processed)
    

    def generate_condition_name_from_dict(cond_dict:  dict | list[dict], join_str: str = ' | ') -> str:
        """
        For a condition dictionary loaded from a correctly formatted YAML file of conditions,
        generates a name that represents those conditions.
        """
        if isinstance(cond_dict, dict):
            cond_dict = [cond_dict]
        
        return join_str.join(c['name'] for c in cond_dict)


    def iter_from_yaml(
            yml: Path | str | dict,
            sources: sc.AnnData | list[sc.AnnData],
            definitions_key: str = 'definitions',
            diffexp_key: str = 'diffexp',
        ) -> Generator[Diffexp, None, None]:
        if isinstance(yml, (Path, str)):
            with Path(yml).open('rb') as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = yml
        cfg = internal.resolve_yaml_definitions(cfg, definitions_key=definitions_key)

        # Ensure unique title names
        titles = set()
        runlist: list[dict] = cfg[diffexp_key]
        for runme in runlist:
            title = runme['title']
            if title in titles:
                raise ValueError(f"Title is not unique: '{title}'")
            titles.add(title)

        # Run!
        for runme in runlist:
            title = runme['title']
            cond_1 = Diffexp.generate_condition_from_dict(runme['condition_1'])
            cond_2 = Diffexp.generate_condition_from_dict(runme['condition_2'])
            name_1 = runme.get('name_1', Diffexp.generate_condition_name_from_dict(runme['condition_1']))
            name_2 = runme.get('name_2', Diffexp.generate_condition_name_from_dict(runme['condition_2']))

            diffexp = Diffexp(
                title = title,
                sources = sources,
                condition_1 = cond_1,
                condition_2 = cond_2,
                name_1 = name_1,
                name_2 = name_2,
                var = None, # TODO
            )

            yield diffexp
