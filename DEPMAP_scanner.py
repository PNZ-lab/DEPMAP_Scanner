#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% ===========================================================================
# 1. Description
# =============================================================================
Description='''
DEPMAP_scanner.py is a tool to explore the DepMap data for different cell lines and cancers.
In section 4 you can create a plot and a csv file including every genes dependency for a subset of the cell lines in DepMap.
    - Optionally, you can plot a set of genes on the plot and calculate the Mann-Whitney U statistic for this distribution.
Section 5 through 7 contain functions that can be launched in section 8.
    - Here we generate boxplots of all different cancer's (in DepMap) dependencies on genes'
    - For a single gene
    - For the n genes with the lowest median scores for a specific cancer
    - For all the genes where one specific cancer has the lowest median score
Section 9 generates heatmaps for all or a subset of cancers for a subset of genes
    - 
'''

#%% =============================================================================
# 2. Setup and Settings
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
import numpy as np
from KTC_functions import KTC_GetGeneSet
from tqdm import tqdm
from adjustText import adjust_text
from typing import Literal



in_dir = '/Volumes/cmgg_pnlab/Kasper/Data/Interesting_Lists'

# in_dm = os.path.join(in_dir, 'CRISPRGeneEffect_Q42024.csv')
in_dm = os.path.join(in_dir, 'CRISPRGeneEffect_2025Q3.csv')

# in_cl = os.path.join(in_dir, 'DepMap_Cellline_annotation_Q42024.csv')
in_cl = os.path.join(in_dir, 'DepMap_CellLine_annotation_2025Q3.csv')

out   = '/Users/kachrist/Desktop/out_dir'

df_dm = pd.read_csv(in_dm, low_memory=False)
df_cl = pd.read_csv(in_cl)

df_dm.rename(columns={df_dm.columns[0]: 'Depmap Id'}, inplace=True)
df_dm.columns = df_dm.columns.str.replace(r'\s\(\d+\)', '', regex=True)

df_cl['OncotreeSubtype'] = df_cl['OncotreeSubtype'].replace({
    'Undifferentiated Pleomorphic Sarcoma/Malignant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma':
    'UPS/MFHS/Spindle Cell Sarcoma'
})

# Optional preload for Section 10 (expression vs drug sensitivity).
# This is loaded once with Section 2 and reused in Section 10.
preload_section10_data = True
df_expr_s10 = None
df_drug_s10 = None
df_prism_cell_meta_s10 = None
df_drug_s10_source = None  # 'prism24q2_matrix' or 'long_table'

if preload_section10_data:
    try:
        expr_candidates = [
            os.path.join(in_dir, 'OmicsExpressionTPMLogp1HumanProteinCodingGenes26Q1.csv'),
            os.path.join(in_dir, 'OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv'),
        ]
        long_drug_candidates = [
            os.path.join(in_dir, 'PRISMRepurposingPrimaryScreenDoseResponse.csv'),
            os.path.join(in_dir, 'PRISMRepurposingSecondaryScreenDoseResponse.csv'),
        ]
        prism_matrix = os.path.join(in_dir, 'Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv')
        prism_compounds = os.path.join(in_dir, 'Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv')
        prism_cell_meta = os.path.join(in_dir, 'Repurposing_Public_24Q2_Cell_Line_Meta_Data.csv')

        in_expr_preload = next((f for f in expr_candidates if os.path.exists(f)), None)
        in_long_preload = next((f for f in long_drug_candidates if os.path.exists(f)), None)

        if in_expr_preload is not None:
            print(f'Section 2 preload: loading expression file: {in_expr_preload}')
            df_expr_s10 = pd.read_csv(in_expr_preload, low_memory=False)
        else:
            print('Section 2 preload: expression file not found (Section 10 will attempt on-demand load).')

        # Prefer 24Q2 matrix if present.
        if os.path.exists(prism_matrix) and os.path.exists(prism_compounds):
            print(f'Section 2 preload: loading PRISM 24Q2 matrix: {prism_matrix}')
            print(f'Section 2 preload: loading PRISM 24Q2 compounds: {prism_compounds}')
            _mat = pd.read_csv(prism_matrix, low_memory=False)
            _cmp = pd.read_csv(prism_compounds, low_memory=False)

            # Convert matrix to long format once during preload.
            _id_col = _mat.columns[0]
            _mat = _mat.rename(columns={_id_col: 'IDs'})
            _model_cols = [c for c in _mat.columns if c != 'IDs']
            _long = _mat.melt(id_vars='IDs', value_vars=_model_cols, var_name='ModelID', value_name='auc')
            _long['auc'] = pd.to_numeric(_long['auc'], errors='coerce')
            _long = _long.dropna(subset=['auc'])

            _cmp_cols = [c for c in ['IDs', 'Drug.Name', 'screen', 'dose', 'MOA', 'repurposing_target'] if c in _cmp.columns]
            _cmp_small = _cmp[_cmp_cols].copy()
            _long = _long.merge(_cmp_small, on='IDs', how='left')
            if 'Drug.Name' in _long.columns:
                _long['inhibitor'] = _long['Drug.Name'].fillna(_long['IDs'])
            else:
                _long['inhibitor'] = _long['IDs']

            _long['ModelID'] = _long['ModelID'].astype(str)
            _long['inhibitor'] = _long['inhibitor'].astype(str)

            df_drug_s10 = _long
            df_drug_s10_source = 'prism24q2_matrix'

            if os.path.exists(prism_cell_meta):
                print(f'Section 2 preload: loading PRISM 24Q2 cell metadata: {prism_cell_meta}')
                df_prism_cell_meta_s10 = pd.read_csv(prism_cell_meta, low_memory=False)

        elif in_long_preload is not None:
            print(f'Section 2 preload: loading long-format drug file: {in_long_preload}')
            df_drug_s10 = pd.read_csv(in_long_preload, low_memory=False)
            df_drug_s10_source = 'long_table'
        else:
            print('Section 2 preload: no drug file found (Section 10 will attempt on-demand load).')

    except Exception as _e:
        print(f'Section 2 preload warning: {_e}')
        print('Section 10 can still run with on-demand loading.')

#%% =============================================================================
# Functions    
# =============================================================================

def PlotOneGene(
    gene,
    highlighted_cancer_types,
    df_cl_col='DepmapModelType', # E.g. OncotreeSubtype, OncotreeLineage, OncotreePrimaryDisease, DepmapModelType
    fig_width=30,
    fig_height=8,
    min_samples=5,
    horizontal=False
):
    # Step 1: Select only the columns for DepmapID and the gene of interest from df_dm
    df_dm_gene = df_dm[['Depmap Id', gene]].rename(columns={'Depmap Id': 'ModelID'})
    
    # Step 2: Merge the gene dependency data with the clinical mapping data
    df_merged = pd.merge(df_dm_gene, df_cl[['ModelID', df_cl_col]], on='ModelID')
    
    # Step 3: Filter out subtypes with fewer than min_samples
    subtype_counts = df_merged[df_cl_col].value_counts()
    valid_subtypes = subtype_counts[subtype_counts >= min_samples].index
    df_merged = df_merged[df_merged[df_cl_col].isin(valid_subtypes)]

    # Step 4: Calculate medians by OncotreeSubtype and sort them
    median_sorted = df_merged.groupby(df_cl_col)[gene].median().sort_values().index
    
    # Define cancer types to recolor and their corresponding colors
    color_palette = [
        '#d62728' if subtype in highlighted_cancer_types else '#1f77b4'
        for subtype in median_sorted
    ]
    
    # Step 5: Create boxplots with custom colors for selected cancer types
    plt.figure(figsize=(fig_width, fig_height), dpi=200)

    if horizontal:
        sns.boxplot(
            y=df_cl_col,
            x=gene,
            data=df_merged,
            order=median_sorted,
            palette=color_palette
        )
        plt.ylabel(df_cl_col)
        plt.xlabel(f'{gene} Dependency Factor')
    else:
        sns.boxplot(
        x=df_cl_col,
        y=gene,
        data=df_merged,
        order=median_sorted,
        hue=df_cl_col,                      # tell seaborn the grouping variable
        hue_order=median_sorted,           # keep same order
        palette=dict(zip(median_sorted, color_palette)),
        dodge=False,
        legend=False                       # we don’t need a legend (hue=x)
        )
        plt.xticks(rotation=90)
        plt.xlabel(df_cl_col)
        plt.ylabel(f'{gene} Dependency Factor')
    
    plt.title(f'Dependency Factor of {gene} Across Cancer Types (samples >= {min_samples})')
    plt.tight_layout()
    plt.show()

def Graph_n_write(gene_set, filter_column, filter_content):

    df_cl_filtered = df_cl[df_cl[filter_column] == filter_content]

    df_merged = pd.merge(df_cl_filtered, df_dm, left_on='ModelID', right_on='Depmap Id')

    gene_columns = df_dm.select_dtypes(include='number').columns
    mean_values =  df_merged[gene_columns].mean().sort_values()

    df_means = pd.DataFrame({
        'Gene': mean_values.index,
        'mean_CRISPR_score': mean_values.values
    })

    fig, ax = plt.subplots(figsize=(8,5), dpi=200)
    plt.scatter(range(len(mean_values)), mean_values, color='black', alpha=1, s=10, zorder=2, label='gene dependency score')
    plt.xlabel('Ranked depmap average', fontsize=18)
    plt.ylabel('mean CRISPR score', fontsize=18)
    plt.tick_params(axis='both', labelsize=12)
    plt.title('DepMap means filtered by: %s' %(filter_content), fontsize=20)
    plt.hlines(0,0, len(gene_columns), color='black', alpha=0.5)

    highlight_genes = KTC_GetGeneSet(gene_set)
    if len(highlight_genes) != 0:
        genes  = mean_values.index
        ranks  = range(len(genes))
        scores = mean_values.values
        highlight_indices = [i for i, gene in enumerate(genes) if gene in highlight_genes]
        highlight_ranks = np.array(ranks)[highlight_indices]
        highlight_scores = np.array(scores)[highlight_indices]
        background_scores = [r for r in scores if r not in highlight_scores]
        highlight_scores = [score for score in highlight_scores if not np.isnan(score)]
        background_scores = [score for score in background_scores if not np.isnan(score)]

        stat, p_value = mannwhitneyu(highlight_scores, background_scores, alternative='two-sided')
        for rank, score in zip(highlight_ranks, highlight_scores):
            plt.vlines(x=rank, ymin=min(scores), ymax=max(scores), color='red', linewidth=0.5, zorder=1, label= '%s (%.0f, p=%f)' %(gene_set, stat, p_value))


    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)
    plt.show()

    out_file = os.path.join(out, 'DEPMAP_%s_filtered.csv' %(filter_content))
    print(f'file written to {out_file}')
    print(df_means)
    df_means.to_csv(out_file, index=False)


def HeatmapSpecificGenes(gene_set, df_cl_col='DepmapModelType', cancers='All', min_samples=10, fig_width=8, fig_height=6, annotate_values=True, title='Cancer dependency (DepMap)'):
    dict_input = {}
    if cancers=='All':
        qualifying_cancers = df_cl[df_cl_col].value_counts()[df_cl[df_cl_col].value_counts()>min_samples].index.tolist()
        for cancer in qualifying_cancers:
            # if cancer not in dict_input:
            dict_input[cancer] = [df_cl_col, cancer]
    else:
        for cancer in cancers:
            dict_input[cancer] = [df_cl_col, cancer]

    # Initialize a dataframe to store mean values for each cancer type
    mean_expression_data = pd.DataFrame()
    
    # Loop through each cancer type to filter, merge, and compute mean scores
    for cancer, filter_params in dict_input.items():
        filter_column = filter_params[0]
        filter_content = filter_params[1]
        
        # Filter and merge data
        df_cl_filtered = df_cl[df_cl[filter_column] == filter_content]
        df_merged = pd.merge(df_cl_filtered, df_dm, left_on='ModelID', right_on='Depmap Id')
        
        # Select only numeric columns representing gene expression data
        gene_columns = df_dm.select_dtypes(include='number').columns
        
        # Compute mean CRISPR score for each gene
        mean_values = df_merged[gene_columns].mean()
        
        # Filter for genes in the specified gene set
        mean_values = mean_values[mean_values.index.isin(gene_set)]
    
        
        # Append results to mean_expression_data with cancer type as index
        mean_expression_data[cancer] = mean_values

    # No need to transpose since genes should go on the y-axis now
    # mean_expression_data will already have genes as rows and cancers as columns
    
    if cancers=='All':
        desired_order = ['AML', 'BALL', 'TALL']
        
        #Combine desired columns with the remaining columns
        remaining_columns = [col for col in mean_expression_data.columns if col not in desired_order]
        reordered_columns = desired_order + remaining_columns
        
        #Reorder the DataFrame columns
        mean_expression_data = mean_expression_data[reordered_columns]
    
    # Plot the heatmap with genes on the y-axis and cancers on the x-axis
    plt.figure(figsize=(fig_width, fig_height), dpi=200)
    # sns.heatmap(mean_expression_data, cmap='coolwarm_r', annot=annotate_values)
    sns.heatmap(mean_expression_data, cmap='coolwarm_r', annot=annotate_values, fmt=".2f")

    plt.title(title)
    # plt.xlabel('Cancer Types')
    # plt.ylabel('Genes')
    out_path = os.path.join(out, 'DepMap_Heatmap_speficic_genes.svg')
    plt.savefig(out_path)
    plt.show()


def FindTopNGenesWithLowestMedianForCancerType(cancer_type, n=5):
    # Merge df_dm with df_cl to include OncotreeSubtype for each ModelID
    df_dm_merged = pd.merge(
        df_dm.rename(columns={'Depmap Id': 'ModelID'}), 
        df_cl[['ModelID', 'OncotreeSubtype']], 
        on='ModelID'
    )
    
    # Filter to include only rows with the specified cancer type
    cancer_type_df = df_dm_merged[df_dm_merged['OncotreeSubtype'] == cancer_type]
    
    # Select only numeric columns representing gene expression (excluding ModelID and OncotreeSubtype)
    gene_columns = cancer_type_df.select_dtypes(include='number').columns
    
    # Calculate the median dependency score for each gene within the specific cancer type
    medians = {gene: cancer_type_df[gene].median(skipna=True) for gene in gene_columns}

    # Remove genes with NaN median scores
    medians = {gene: median for gene, median in medians.items() if pd.notna(median)}

    # Sort the medians dictionary by values (median scores) in ascending order
    sorted_medians = sorted(medians.items(), key=lambda item: item[1])  # This gives you a list of (gene, median) tuples
    
    # Get the top n lowest genes
    top_n_lowest_genes = [gene for gene, _ in sorted_medians[:n]]
    
    return top_n_lowest_genes

def get_gene_medians(gene, df_cl_col='OncotreeSubtype', min_samples=5):
    df_dm_gene = df_dm[['Depmap Id', gene]].rename(columns={'Depmap Id': 'ModelID'})
    df_merged  = pd.merge(df_dm_gene, df_cl[['ModelID', df_cl_col]], on='ModelID')

    # Per-gene sample filtering
    subtype_counts = df_merged[df_cl_col].value_counts()
    valid_subtypes = subtype_counts[subtype_counts >= min_samples].index
    df_merged      = df_merged[df_merged[df_cl_col].isin(valid_subtypes)]

    med = df_merged.groupby(df_cl_col)[gene].median()
    return med

def FindGenesWhereCancerTypeHasLowestMedian(
    target,
    df_cl_col='OncotreeSubtype',
    min_samples=5,
    unique_min=False,     # if True: require target to be the *only* lowest
    max_score=None        # e.g. -1.0 -> require median(target) <= -1
):
    # All gene columns = everything except ID columns
    gene_columns = [c for c in df_dm.columns if c not in ['Depmap Id', 'ModelID']]

    genes_with_lowest = []

    for gene in gene_columns:
        med = get_gene_medians(gene, df_cl_col=df_cl_col, min_samples=min_samples)

        if target not in med.index:
            continue

        target_median = med[target]
        if pd.isna(target_median):
            continue

        # Threshold on strength of dependency
        if max_score is not None and target_median > max_score:
            # remember: more negative = stronger dependency
            continue

        # Check if target has the lowest median
        if target_median == med.min():
            if unique_min and (med == med.min()).sum() > 1:
                # skip ties if we require uniqueness
                continue
            genes_with_lowest.append(gene)

    return genes_with_lowest

def Scatter_CellLines(
    df_dm: pd.DataFrame,
    df_cl: pd.DataFrame,
    gene: str,
    *,
    model_type: str | None = None,
    oncotree_subtype: str | None = None,
    oncotree_lineage: str | None = None,
    query: str | None = None,

    dep_id_col: str = "Depmap Id",
    meta_id_col: str = "ModelID",
    name_col: str = "StrippedCellLineName",

    sort_ascending: bool = True,
    highlight: set[str] | None = None,
    label: str | None = "highlight",     # "highlight", "all", "none"
    top_n_labels: int = 0,

    figsize: tuple[int, int] = (9, 5),
    dpi: int = 200,
    point_size: int = 25,
    highlight_color: str = "tab:orange",
    yline0: bool = True,
    title: str | None = None,
):
    """
    Plot Chronos gene dependency for a subset of DepMap models.
    More negative Chronos => stronger dependency.
    """

    if gene not in df_dm.columns:
        raise ValueError(f"Gene '{gene}' not found in df_dm columns.")

    # --- Dependency matrix ---
    df_dep = df_dm.copy()
    if dep_id_col in df_dep.columns:
        df_dep = df_dep.set_index(dep_id_col)
    df_dep.index = df_dep.index.astype(str)

    # --- Metadata ---
    df_meta = df_cl.copy()
    df_meta[meta_id_col] = df_meta[meta_id_col].astype(str)

    subset = df_meta

    if model_type is not None:
        subset = subset[subset["DepmapModelType"] == model_type]

    if oncotree_subtype is not None:
        subset = subset[subset["OncotreeSubtype"] == oncotree_subtype]

    if oncotree_lineage is not None:
        subset = subset[subset["OncotreeLineage"] == oncotree_lineage]

    if query is not None:
        subset = subset.query(query)

    print("Models in subset:", len(subset))

    common_ids = pd.Index(subset[meta_id_col]).intersection(df_dep.index)
    print("Overlap with dependency matrix:", len(common_ids))

    if len(common_ids) == 0:
        raise ValueError("No overlapping models.")

    subset = subset[subset[meta_id_col].isin(common_ids)].copy()
    dep_sub = df_dep.loc[common_ids].copy()

    name_map = subset.set_index(meta_id_col)[name_col].to_dict()
    dep_sub["CellLine"] = [name_map.get(mid, mid) for mid in dep_sub.index]
    dep_sub["ModelID"] = dep_sub.index

    dep_plot = dep_sub[["ModelID", "CellLine", gene]].copy()
    dep_plot[gene] = pd.to_numeric(dep_plot[gene], errors="coerce")
    dep_sorted = dep_plot.sort_values(gene, ascending=sort_ascending).reset_index(drop=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    x = np.arange(len(dep_sorted))
    y = dep_sorted[gene].values

    ax.scatter(x, y, s=point_size, alpha=0.8)
    ax.grid(True, linewidth=0.5, alpha=0.4)

    if yline0:
        ax.axhline(0, linewidth=1, alpha=0.6)

    hl = set(highlight or [])
    label_mode = (label or "highlight").lower()

    def should_label(cellline: str, rank: int):
        if label_mode == "all":
            return True
        if label_mode == "none":
            return False
        if cellline in hl:
            return True
        if top_n_labels and rank < top_n_labels:
            return True
        return False

    # Highlight points
    if hl:
        for i, row in dep_sorted.iterrows():
            if row["CellLine"] in hl:
                ax.scatter(i, row[gene], s=point_size * 3, color=highlight_color)

    # --- Create text objects ---
    texts = []
    for i, row in dep_sorted.iterrows():
        if should_label(str(row["CellLine"]), i):
            txt = ax.text(i, row[gene], row["CellLine"], fontsize=9)
            texts.append(txt)

    # --- Adjust text positions ---
    if texts:
        adjust_text(
            texts,
            ax=ax,
            expand_points=(1.2, 1.4),
            expand_text=(1.2, 1.4),
            arrowprops=dict(arrowstyle="-", linewidth=0.5, alpha=0.6),
        )

    ax.set_ylabel("Chronos Gene Effect (lower = stronger dependency)")
    ax.set_xticks([])

    subset_desc = []
    if model_type: subset_desc.append(f"DepmapModelType={model_type}")
    if oncotree_subtype: subset_desc.append(f"OncotreeSubtype={oncotree_subtype}")
    if oncotree_lineage: subset_desc.append(f"OncotreeLineage={oncotree_lineage}")
    if query: subset_desc.append(f"query={query}")

    default_title = f"{gene} dependency"
    if subset_desc:
        default_title += " (" + ", ".join(subset_desc) + ")"

    ax.set_title(title or default_title)

    plt.tight_layout()
    plt.show()

    return dep_sorted


def _bh_fdr(p_values):
    """Benjamini-Hochberg FDR correction for a 1D array-like of p-values."""
    p = pd.Series(p_values, dtype='float64')
    out = pd.Series(np.nan, index=p.index, dtype='float64')

    valid = p.notna()
    if valid.sum() == 0:
        return out.values

    pv = p[valid].values
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    q_unsorted = np.empty_like(q)
    q_unsorted[order] = q
    out.loc[valid] = q_unsorted
    return out.values


def BuildDrugLong(
    df_drug,
    sample_col_candidates=('ModelID', 'model_id', 'Depmap_ID', 'depmap_id', 'Depmap Id'),
    drug_col_candidates=('inhibitor', 'drug', 'compound', 'Drug', 'Compound'),
    auc_col_candidates=('auc', 'AUC')
):
    """Standardize a drug table into: ModelID, inhibitor, auc."""
    def pick_col(cols, cands, label):
        for c in cands:
            if c in cols:
                return c
        raise ValueError(f"Could not find a {label} column among {cands}. Available: {list(cols)}")

    sample_col = pick_col(df_drug.columns, sample_col_candidates, 'sample ID')
    drug_col = pick_col(df_drug.columns, drug_col_candidates, 'drug name')
    auc_col = pick_col(df_drug.columns, auc_col_candidates, 'AUC')

    out = df_drug.rename(
        columns={sample_col: 'ModelID', drug_col: 'inhibitor', auc_col: 'auc'}
    ).copy()

    out = out.dropna(subset=['ModelID', 'inhibitor', 'auc'])
    out['auc'] = pd.to_numeric(out['auc'], errors='coerce')
    out = out.dropna(subset=['auc'])
    out['ModelID'] = out['ModelID'].astype(str)
    out['inhibitor'] = out['inhibitor'].astype(str)
    return out


def BuildDrugLongFromExtendedPrimary(
    df_matrix,
    df_compounds,
    df_cell_meta=None,
    culture_filter=None,
):
    """
    Convert PRISM 24Q2 Extended Primary matrix (wide LFC table) to long format.

    Output is standardized to columns: ModelID, inhibitor, auc.
    Here, 'auc' stores LFC values so existing scan functions can be reused.
    """
    mat = df_matrix.copy()

    # Matrix first column is typically unnamed and contains treatment IDs.
    id_col = mat.columns[0]
    mat = mat.rename(columns={id_col: 'IDs'})

    model_cols = [c for c in mat.columns if c != 'IDs']
    long = mat.melt(id_vars='IDs', value_vars=model_cols, var_name='ModelID', value_name='auc')
    long['auc'] = pd.to_numeric(long['auc'], errors='coerce')
    long = long.dropna(subset=['auc'])

    cmp_cols = [c for c in ['IDs', 'Drug.Name', 'screen', 'dose', 'MOA', 'repurposing_target'] if c in df_compounds.columns]
    cmp = df_compounds[cmp_cols].copy()
    long = long.merge(cmp, on='IDs', how='left')

    if 'Drug.Name' in long.columns:
        long['inhibitor'] = long['Drug.Name'].fillna(long['IDs'])
    else:
        long['inhibitor'] = long['IDs']

    if df_cell_meta is not None and culture_filter is not None:
        culture_vals = [culture_filter] if isinstance(culture_filter, str) else list(culture_filter)
        if 'depmap_id' not in df_cell_meta.columns or 'culture' not in df_cell_meta.columns:
            raise ValueError("df_cell_meta must contain 'depmap_id' and 'culture' when culture_filter is used.")
        valid_ids = df_cell_meta[df_cell_meta['culture'].isin(culture_vals)]['depmap_id'].astype(str).unique().tolist()
        long = long[long['ModelID'].astype(str).isin(valid_ids)].copy()

    long['ModelID'] = long['ModelID'].astype(str)
    long['inhibitor'] = long['inhibitor'].astype(str)
    return long


def _get_expression_vector(df_expr, gene):
    """Return a standardized expression vector table: ModelID, expression."""
    expr_cols = list(df_expr.columns)

    id_col = None
    for c in ('ModelID', 'Depmap Id', 'Unnamed: 0'):
        if c in expr_cols:
            id_col = c
            break
    if id_col is None:
        raise ValueError("No ID column found in expression dataframe. Expected one of: ModelID, Depmap Id, Unnamed: 0")

    gene_match = [c for c in expr_cols if str(c).split(' (')[0].upper() == gene.upper()]
    if len(gene_match) == 0:
        raise ValueError(f"Gene '{gene}' was not found in expression dataframe columns.")

    gene_col = gene_match[0]

    expr_vec = df_expr[[id_col, gene_col]].copy()
    expr_vec = expr_vec.rename(columns={id_col: 'ModelID', gene_col: 'expression'})
    expr_vec['ModelID'] = expr_vec['ModelID'].astype(str)
    expr_vec['expression'] = pd.to_numeric(expr_vec['expression'], errors='coerce')
    return expr_vec.dropna(subset=['ModelID', 'expression'])


def ScanExpressionDrugSensitivity(
    gene,
    df_expr,
    df_drug,
    subset_col=None,
    subset_values=None,
    min_n=25,
    corr_method: Literal['pearson', 'spearman', 'both']='both',
    split_high_low=True,
):
    """
    Scan all drugs for association between one gene's expression and drug response.

    Notes:
    - Uses AUC where lower means higher sensitivity.
    - Also reports sens_score = -AUC so positive correlation implies sensitization.
    """
    expr_vec = _get_expression_vector(df_expr, gene)
    drug_long = BuildDrugLong(df_drug)

    df = pd.merge(drug_long, expr_vec, on='ModelID', how='inner')

    if subset_col is not None and subset_values is not None:
        if isinstance(subset_values, str):
            subset_values = [subset_values]
        allowed = df_cl[df_cl[subset_col].isin(subset_values)]['ModelID'].astype(str).tolist()
        df = df[df['ModelID'].isin(allowed)].copy()

    df['sens_score'] = -df['auc']

    def corr_with_p(x, y, method):
        pair = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(pair) < 3:
            return np.nan, np.nan

        # Avoid scipy ConstantInputWarning for zero-variance vectors.
        if pair['x'].nunique() <= 1 or pair['y'].nunique() <= 1:
            return np.nan, np.nan

        if method == 'pearson':
            r, p = pearsonr(pair['x'], pair['y'])
        else:
            r, p = spearmanr(pair['x'], pair['y'])
        return float(r), float(p)

    results = []
    for inhibitor, g in df.groupby('inhibitor', dropna=True):
        g2 = g.dropna(subset=['expression', 'auc']).copy()
        n = int(g2.shape[0])
        if n < min_n:
            continue

        row = {'gene': gene, 'inhibitor': inhibitor, 'n': n}

        if corr_method in ('pearson', 'both'):
            r_auc, p_auc = corr_with_p(g2['expression'], g2['auc'], 'pearson')
            r_sens, p_sens = corr_with_p(g2['expression'], g2['sens_score'], 'pearson')
            row['r_pearson_expr_auc'] = r_auc
            row['p_pearson_expr_auc'] = p_auc
            row['r_pearson_expr_sens'] = r_sens
            row['p_pearson_expr_sens'] = p_sens

        if corr_method in ('spearman', 'both'):
            r_auc, p_auc = corr_with_p(g2['expression'], g2['auc'], 'spearman')
            r_sens, p_sens = corr_with_p(g2['expression'], g2['sens_score'], 'spearman')
            row['r_spearman_expr_auc'] = r_auc
            row['p_spearman_expr_auc'] = p_auc
            row['r_spearman_expr_sens'] = r_sens
            row['p_spearman_expr_sens'] = p_sens

        if split_high_low:
            med = float(g2['expression'].median())
            hi = g2[g2['expression'] >= med]['auc']
            lo = g2[g2['expression'] < med]['auc']
            row['delta_mean_auc_hi_minus_lo'] = float(hi.mean() - lo.mean())
            if len(hi) >= 2 and len(lo) >= 2:
                row['p_mannwhitney_hi_vs_lo_auc'] = float(mannwhitneyu(hi, lo, alternative='two-sided').pvalue)
            else:
                row['p_mannwhitney_hi_vs_lo_auc'] = np.nan

        results.append(row)

    res = pd.DataFrame(results)
    if res.empty:
        return res

    for p_col in ('p_spearman_expr_sens', 'p_pearson_expr_sens', 'p_mannwhitney_hi_vs_lo_auc'):
        if p_col in res.columns:
            res[f'fdr_{p_col}'] = _bh_fdr(res[p_col])

    strength_metric_col = None
    strength_p_col = None
    strength_fdr_col = None

    if 'r_spearman_expr_sens' in res.columns:
        res['strength'] = res['r_spearman_expr_sens'].astype(float)
        strength_metric_col = 'r_spearman_expr_sens'
        strength_p_col = 'p_spearman_expr_sens'
        strength_fdr_col = 'fdr_p_spearman_expr_sens'
    elif 'r_pearson_expr_sens' in res.columns:
        res['strength'] = res['r_pearson_expr_sens'].astype(float)
        strength_metric_col = 'r_pearson_expr_sens'
        strength_p_col = 'p_pearson_expr_sens'
        strength_fdr_col = 'fdr_p_pearson_expr_sens'
    elif 'r_spearman_expr_auc' in res.columns:
        res['strength'] = -res['r_spearman_expr_auc'].astype(float)
        strength_metric_col = 'r_spearman_expr_auc'
        strength_p_col = 'p_spearman_expr_auc'
    else:
        res['strength'] = -res['r_pearson_expr_auc'].astype(float)
        strength_metric_col = 'r_pearson_expr_auc'
        strength_p_col = 'p_pearson_expr_auc'

    res['strength_metric_col'] = strength_metric_col
    if strength_metric_col is not None and strength_metric_col in res.columns:
        res['strength_metric_value'] = pd.to_numeric(res[strength_metric_col], errors='coerce')
    else:
        res['strength_metric_value'] = np.nan

    if strength_p_col is not None and strength_p_col in res.columns:
        res['strength_p'] = pd.to_numeric(res[strength_p_col], errors='coerce')
    else:
        res['strength_p'] = np.nan

    if strength_fdr_col is not None and strength_fdr_col in res.columns:
        res['strength_fdr'] = pd.to_numeric(res[strength_fdr_col], errors='coerce')
    else:
        res['strength_fdr'] = np.nan

    return res.sort_values(['strength', 'n'], ascending=[False, False]).reset_index(drop=True)


def PlotExpressionVsDrug(
    gene,
    inhibitor,
    df_expr,
    df_drug,
    subset_col=None,
    subset_values=None,
    show_regression=True,
    label_n=True,
    fig_width=6,
    fig_height=5,
):
    """Scatter plot: expression of one gene versus drug AUC for one inhibitor."""
    expr_vec = _get_expression_vector(df_expr, gene)
    drug_long = BuildDrugLong(df_drug)
    df = pd.merge(drug_long, expr_vec, on='ModelID', how='inner')

    if subset_col is not None and subset_values is not None:
        if isinstance(subset_values, str):
            subset_values = [subset_values]
        allowed = df_cl[df_cl[subset_col].isin(subset_values)]['ModelID'].astype(str).tolist()
        df = df[df['ModelID'].isin(allowed)].copy()

    df = df[df['inhibitor'] == inhibitor].dropna(subset=['expression', 'auc']).copy()
    if df.empty:
        raise ValueError(f"No overlapping data for inhibitor '{inhibitor}' after filtering.")

    x = df['expression'].values
    y = df['auc'].values

    plt.figure(figsize=(fig_width, fig_height), dpi=200)
    plt.scatter(x, y, alpha=0.6)

    title = f"{gene} expression vs {inhibitor} AUC"
    if subset_col is not None and subset_values is not None:
        if isinstance(subset_values, str):
            subset_label = subset_values
        else:
            subset_label = ', '.join([str(v) for v in subset_values])
        title += f"\nsubset: {subset_col} = {subset_label}"

    if show_regression and len(df) >= 3:
        if pd.Series(x).nunique() > 1 and pd.Series(y).nunique() > 1:
            slope, intercept = np.polyfit(x, y, 1)
            xfit = np.linspace(x.min(), x.max(), 100)
            yfit = slope * xfit + intercept
            plt.plot(xfit, yfit, color='black')
            r, p = pearsonr(x, y)
            title += f" | R={r:.2f}, p={p:.2e}"
        else:
            title += " | R=NA (constant input)"

    if label_n:
        title += f" (n={len(df)})"

    plt.xlabel(f"{gene} expression")
    plt.ylabel('AUC (lower = more sensitive)')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def RunExpressionDrugSensitivityScan(
    gene_of_interest,
    subset_col='DepmapModelType',
    subset_values=None,
    min_n=15,
    corr_method='both',
    split_high_low=True,
    prism_culture_filter=None,
    plot_single_inhibitor=False,
    inhibitor_for_plot='Venetoclax',
):
    """Load expression + PRISM data and run expression-vs-sensitivity scan."""
    global df_expr_s10, df_drug_s10, df_prism_cell_meta_s10, df_drug_s10_source

    if df_expr_s10 is not None and df_drug_s10 is not None:
        print('Using preloaded Section 2 data for expression and drug response.')
        df_expr = df_expr_s10
        df_drug = df_drug_s10.copy()

        if prism_culture_filter is not None and df_drug_s10_source == 'prism24q2_matrix':
            if df_prism_cell_meta_s10 is None:
                raise FileNotFoundError(
                    'prism_culture_filter was requested, but Repurposing_Public_24Q2_Cell_Line_Meta_Data.csv was not loaded in Section 2.'
                )
            culture_vals = [prism_culture_filter] if isinstance(prism_culture_filter, str) else list(prism_culture_filter)
            if 'depmap_id' not in df_prism_cell_meta_s10.columns or 'culture' not in df_prism_cell_meta_s10.columns:
                raise ValueError("PRISM cell metadata must contain 'depmap_id' and 'culture'.")
            valid_ids = df_prism_cell_meta_s10[df_prism_cell_meta_s10['culture'].isin(culture_vals)]['depmap_id'].astype(str).unique().tolist()
            df_drug = df_drug[df_drug['ModelID'].astype(str).isin(valid_ids)].copy()

    else:
        print('Section 2 preloaded data not available. Falling back to on-demand loading.')

        expression_file_candidates = [
            os.path.join(in_dir, 'OmicsExpressionTPMLogp1HumanProteinCodingGenes26Q1.csv'),
            os.path.join(in_dir, 'OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv'),
        ]

        drug_file_candidates = [
            os.path.join(in_dir, 'PRISMRepurposingPrimaryScreenDoseResponse.csv'),
            os.path.join(in_dir, 'PRISMRepurposingSecondaryScreenDoseResponse.csv'),
        ]

        prism_24q2_matrix_candidates = [
            os.path.join(in_dir, 'Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv'),
        ]

        prism_24q2_compound_candidates = [
            os.path.join(in_dir, 'Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv'),
        ]

        prism_24q2_cell_meta_candidates = [
            os.path.join(in_dir, 'Repurposing_Public_24Q2_Cell_Line_Meta_Data.csv'),
        ]

        in_expr = next((f for f in expression_file_candidates if os.path.exists(f)), None)
        in_drug = next((f for f in drug_file_candidates if os.path.exists(f)), None)
        in_prism_matrix = next((f for f in prism_24q2_matrix_candidates if os.path.exists(f)), None)
        in_prism_compounds = next((f for f in prism_24q2_compound_candidates if os.path.exists(f)), None)
        in_prism_cell_meta = next((f for f in prism_24q2_cell_meta_candidates if os.path.exists(f)), None)

        if in_expr is None:
            raise FileNotFoundError(f'No expression file found. Tried: {expression_file_candidates}')
        if in_prism_matrix is None and in_drug is None:
            raise FileNotFoundError(
                f'No drug response file found. Tried 24Q2 matrix: {prism_24q2_matrix_candidates} and long-format files: {drug_file_candidates}'
            )
        if in_prism_matrix is not None and in_prism_compounds is None:
            raise FileNotFoundError(f'Found 24Q2 matrix but no 24Q2 compound list. Tried: {prism_24q2_compound_candidates}')

        print(f'Loading expression file: {in_expr}')
        df_expr = pd.read_csv(in_expr, low_memory=False)

        if in_prism_matrix is not None:
            print(f'Loading PRISM 24Q2 matrix: {in_prism_matrix}')
            print(f'Loading PRISM 24Q2 compounds: {in_prism_compounds}')
            df_matrix = pd.read_csv(in_prism_matrix, low_memory=False)
            df_compounds = pd.read_csv(in_prism_compounds, low_memory=False)
            df_cell_meta = pd.read_csv(in_prism_cell_meta, low_memory=False) if in_prism_cell_meta is not None else None
            df_drug = BuildDrugLongFromExtendedPrimary(
                df_matrix=df_matrix,
                df_compounds=df_compounds,
                df_cell_meta=df_cell_meta,
                culture_filter=prism_culture_filter,
            )
            print('Using LFC from PRISM 24Q2 matrix as response metric (stored in auc column for compatibility).')
        else:
            print(f'Loading long-format drug file: {in_drug}')
            df_drug = pd.read_csv(in_drug, low_memory=False)

    df_drug_scan = ScanExpressionDrugSensitivity(
        gene=gene_of_interest,
        df_expr=df_expr,
        df_drug=df_drug,
        subset_col=subset_col,
        subset_values=subset_values,
        min_n=min_n,
        corr_method=corr_method,
        split_high_low=split_high_low,
    )

    out_scan_file = os.path.join(out, f'{gene_of_interest}_expression_vs_drug_sensitivity_scan.csv')
    df_drug_scan.to_csv(out_scan_file, index=False)
    print(f'Wrote scan results to: {out_scan_file}')
    top5_cols = [
        c for c in [
            'inhibitor',
            'n',
            'strength',
            'strength_metric_col',
            'strength_metric_value',
            'strength_p',
            'strength_fdr',
        ]
        if c in df_drug_scan.columns
    ]
    print('Top 5 inhibitors (ranked by strength):')
    print(df_drug_scan[top5_cols].head(5).to_string(index=False))

    if not df_drug_scan.empty:
        top_inhibitor = str(df_drug_scan.iloc[0]['inhibitor'])
        print(f"Auto-plotting #1 hit: {top_inhibitor}")
        PlotExpressionVsDrug(
            gene=gene_of_interest,
            inhibitor=top_inhibitor,
            df_expr=df_expr,
            df_drug=df_drug,
            subset_col=subset_col,
            subset_values=subset_values,
            show_regression=True,
            label_n=True,
        )
    else:
        print('No hits passed filters, so no top-hit plot was generated.')

    if plot_single_inhibitor:
        PlotExpressionVsDrug(
            gene=gene_of_interest,
            inhibitor=inhibitor_for_plot,
            df_expr=df_expr,
            df_drug=df_drug,
            subset_col=subset_col,
            subset_values=subset_values,
            show_regression=True,
            label_n=True,
        )

    return df_drug_scan


def _resolve_inhibitor_name(df_drug, inhibitor_query):
    """Resolve an inhibitor name robustly (exact -> case-insensitive -> contains)."""
    available = pd.Series(df_drug['inhibitor'].astype(str).unique())

    exact = available[available == inhibitor_query]
    if len(exact) > 0:
        return exact.iloc[0]

    exact_ci = available[available.str.lower() == str(inhibitor_query).lower()]
    if len(exact_ci) > 0:
        return exact_ci.iloc[0]

    contains = available[available.str.lower().str.contains(str(inhibitor_query).lower(), regex=False)]
    if len(contains) > 0:
        print(f"Inhibitor '{inhibitor_query}' not found exactly. Using closest match: {contains.iloc[0]}")
        if len(contains) > 1:
            print('Other matches:', ', '.join(contains.head(10).tolist()))
        return contains.iloc[0]

    raise ValueError(f"Inhibitor '{inhibitor_query}' not found in drug table.")


def ScanGenesForInhibitor(
    inhibitor,
    df_expr,
    df_drug,
    subset_col=None,
    subset_values=None,
    min_n=15,
    corr_method: Literal['pearson', 'spearman', 'both'] = 'both',
):
    """
    Scan all genes for association with one inhibitor response.

    Notes:
    - Uses AUC where lower means higher sensitivity.
    - Also reports sens_score = -AUC so positive correlation implies sensitization.
    """
    drug_long = BuildDrugLong(df_drug)
    inhibitor_resolved = _resolve_inhibitor_name(drug_long, inhibitor)
    drug_one = drug_long[drug_long['inhibitor'] == inhibitor_resolved].copy()

    if subset_col is not None and subset_values is not None:
        if isinstance(subset_values, str):
            subset_values = [subset_values]
        allowed = df_cl[df_cl[subset_col].isin(subset_values)]['ModelID'].astype(str).tolist()
        drug_one = drug_one[drug_one['ModelID'].isin(allowed)].copy()

    if drug_one.empty:
        raise ValueError(f"No rows left for inhibitor '{inhibitor_resolved}' after subset filtering.")

    # Prepare expression matrix with standardized ModelID.
    expr_cols = list(df_expr.columns)
    id_col = None
    for c in ('ModelID', 'Depmap Id', 'Unnamed: 0'):
        if c in expr_cols:
            id_col = c
            break
    if id_col is None:
        raise ValueError("No ID column found in expression dataframe. Expected one of: ModelID, Depmap Id, Unnamed: 0")

    gene_cols = [c for c in df_expr.columns if c != id_col]
    expr_mat = df_expr[[id_col] + gene_cols].copy().rename(columns={id_col: 'ModelID'})
    expr_mat['ModelID'] = expr_mat['ModelID'].astype(str)

    merged = pd.merge(drug_one[['ModelID', 'auc']], expr_mat, on='ModelID', how='inner')
    merged['sens_score'] = -pd.to_numeric(merged['auc'], errors='coerce')

    def corr_with_p(x, y, method):
        pair = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(pair) < 3:
            return np.nan, np.nan, 0

        # Avoid scipy ConstantInputWarning for zero-variance vectors.
        if pair['x'].nunique() <= 1 or pair['y'].nunique() <= 1:
            return np.nan, np.nan, len(pair)

        if method == 'pearson':
            r, p = pearsonr(pair['x'], pair['y'])
        else:
            r, p = spearmanr(pair['x'], pair['y'])
        return float(r), float(p), len(pair)

    results = []
    for col in gene_cols:
        gene_label = str(col).split(' (')[0]
        x = pd.to_numeric(merged[col], errors='coerce')
        y_auc = pd.to_numeric(merged['auc'], errors='coerce')
        y_sens = pd.to_numeric(merged['sens_score'], errors='coerce')

        row = {'inhibitor': inhibitor_resolved, 'gene': gene_label}
        n = int(pd.DataFrame({'x': x, 'y': y_auc}).dropna().shape[0])
        row['n'] = n
        if n < min_n:
            continue

        if corr_method in ('pearson', 'both'):
            r_auc, p_auc, _ = corr_with_p(x, y_auc, 'pearson')
            r_sens, p_sens, _ = corr_with_p(x, y_sens, 'pearson')
            row['r_pearson_expr_auc'] = r_auc
            row['p_pearson_expr_auc'] = p_auc
            row['r_pearson_expr_sens'] = r_sens
            row['p_pearson_expr_sens'] = p_sens

        if corr_method in ('spearman', 'both'):
            r_auc, p_auc, _ = corr_with_p(x, y_auc, 'spearman')
            r_sens, p_sens, _ = corr_with_p(x, y_sens, 'spearman')
            row['r_spearman_expr_auc'] = r_auc
            row['p_spearman_expr_auc'] = p_auc
            row['r_spearman_expr_sens'] = r_sens
            row['p_spearman_expr_sens'] = p_sens

        results.append(row)

    res = pd.DataFrame(results)
    if res.empty:
        return res

    for p_col in ('p_spearman_expr_sens', 'p_pearson_expr_sens'):
        if p_col in res.columns:
            res[f'fdr_{p_col}'] = _bh_fdr(res[p_col])

    strength_metric_col = None
    strength_p_col = None
    strength_fdr_col = None
    if 'r_spearman_expr_sens' in res.columns:
        res['strength'] = res['r_spearman_expr_sens'].astype(float)
        strength_metric_col = 'r_spearman_expr_sens'
        strength_p_col = 'p_spearman_expr_sens'
        strength_fdr_col = 'fdr_p_spearman_expr_sens'
    elif 'r_pearson_expr_sens' in res.columns:
        res['strength'] = res['r_pearson_expr_sens'].astype(float)
        strength_metric_col = 'r_pearson_expr_sens'
        strength_p_col = 'p_pearson_expr_sens'
        strength_fdr_col = 'fdr_p_pearson_expr_sens'
    elif 'r_spearman_expr_auc' in res.columns:
        res['strength'] = -res['r_spearman_expr_auc'].astype(float)
        strength_metric_col = 'r_spearman_expr_auc'
        strength_p_col = 'p_spearman_expr_auc'
    else:
        res['strength'] = -res['r_pearson_expr_auc'].astype(float)
        strength_metric_col = 'r_pearson_expr_auc'
        strength_p_col = 'p_pearson_expr_auc'

    res['strength_metric_col'] = strength_metric_col
    res['strength_metric_value'] = pd.to_numeric(res[strength_metric_col], errors='coerce') if strength_metric_col in res.columns else np.nan
    res['strength_p'] = pd.to_numeric(res[strength_p_col], errors='coerce') if (strength_p_col is not None and strength_p_col in res.columns) else np.nan
    res['strength_fdr'] = pd.to_numeric(res[strength_fdr_col], errors='coerce') if (strength_fdr_col is not None and strength_fdr_col in res.columns) else np.nan

    return res.sort_values(['strength', 'n'], ascending=[False, False]).reset_index(drop=True)


def RunInhibitorGeneCorrelationScan(
    inhibitor,
    subset_col='DepmapModelType',
    subset_values=None,
    min_n=15,
    corr_method='both',
    plot_top_gene=True,
):
    """Run inhibitor->gene scan using preloaded Section 2 data when available."""
    global df_expr_s10, df_drug_s10

    if df_expr_s10 is not None and df_drug_s10 is not None:
        print('Using preloaded Section 2 data for inhibitor->gene scan.')
        df_expr = df_expr_s10
        df_drug = df_drug_s10
    else:
        raise RuntimeError('Section 2 preloaded data not available. Run Section 2 first.')

    df_gene_scan = ScanGenesForInhibitor(
        inhibitor=inhibitor,
        df_expr=df_expr,
        df_drug=df_drug,
        subset_col=subset_col,
        subset_values=subset_values,
        min_n=min_n,
        corr_method=corr_method,
    )

    subset_suffix = 'all' if subset_values is None else '_'.join([str(v).replace(' ', '_') for v in (subset_values if isinstance(subset_values, list) else [subset_values])])
    out_file = os.path.join(out, f"{str(inhibitor).replace(' ', '_')}_gene_correlations_{subset_suffix}.csv")
    df_gene_scan.to_csv(out_file, index=False)
    print(f'Wrote gene correlation scan to: {out_file}')

    top_cols = [c for c in ['gene', 'n', 'strength', 'strength_metric_col', 'strength_metric_value', 'strength_p', 'strength_fdr'] if c in df_gene_scan.columns]
    print(f'Top 5 genes for inhibitor {inhibitor}:')
    print(df_gene_scan[top_cols].head(5).to_string(index=False))

    if plot_top_gene:
        if not df_gene_scan.empty:
            top_gene = str(df_gene_scan.iloc[0]['gene'])
            top_inhibitor = str(df_gene_scan.iloc[0]['inhibitor'])
            print(f"Auto-plotting top gene for {top_inhibitor}: {top_gene}")
            PlotExpressionVsDrug(
                gene=top_gene,
                inhibitor=top_inhibitor,
                df_expr=df_expr,
                df_drug=df_drug,
                subset_col=subset_col,
                subset_values=subset_values,
                show_regression=True,
                label_n=True,
            )
        else:
            print(f'No genes passed filters for {inhibitor}; no top-gene plot generated.')

    return df_gene_scan

#%% ===========================================================================
# 4 Waterfall plot showing dependencies of input genes
# =============================================================================

# Graph_n_write takes three arguments"
    # a name for a gene set (genes in gene set will be determined by GetGeneSet())
    # a column in the dataframe for the description of the cell lines (e.g. OncotreeLineage or OncotreeCode)
    # a value in the given column to consider a hit for inclusion purposes (e.g. BRCA or TLL)
Graph_n_write('SG', 'DepmapModelType', 'TALL')


#%% ===========================================================================
# 5 Function: Draw boxplot of all cancer dependencies for one gene
# =============================================================================

import seaborn as sns

gene = 'KIF11'
highlighted_cancer_types = ['TALL', 'BALL', 'AML']
df_cl_col='DepmapModelType' # E.g. OncotreeSubtype, OncotreeLineage, OncotreePrimaryDisease, DepmapModelType
fig_width=12
fig_height=4
min_samples=6
horizontal=False

PlotOneGene(gene, highlighted_cancer_types, df_cl_col=df_cl_col, min_samples=min_samples, horizontal=horizontal, fig_width=fig_width, fig_height=fig_height)


#%% ===========================================================================
# 7 Function: Scan all genes and find those were a specific cancer type has the lowest score - then plot them
# =============================================================================


target_cancer = 'TALL'
df_cl_col     = 'DepmapModelType'
min_samples   = 6
fig_height    = 6
fig_width     = 12
max_score     = -1.0

genes = FindGenesWhereCancerTypeHasLowestMedian(target_cancer, df_cl_col=df_cl_col, min_samples=min_samples, max_score=max_score,)

for gene in genes:
    PlotOneGene(gene, highlighted_cancer_types=[target_cancer], min_samples=min_samples, fig_width=fig_width, fig_height=fig_height)



#%% ===========================================================================
# 9 Heatmaps for specific genes and specific (or all) cancers
# =============================================================================


gene_set    = [
    "TOMM20",   # TOM20
    "BCL2",     # BCL-2
    "MCL1",     # MCL-1
    "BCL2L1",   # BCL-XL
    "PMAIP1",   # NOXA
    "BBC3",     # PUMA
    "FIS1",
    "OPA1",
    "DNM1L",    # DRP1
    "PINK1",
    "PRKN",     # PARKIN
    "HERPUD1",
    "DDIT4",
    "STARD4",
    "ALAS1",
    "TOB2",
    "NFE2L1",
    "CAVIN1",
    "GPX4",
    "ATF4",
]

#Ksenia up
# gene_set = ["B4GALT5", "FANCB", "SARAF", "R3HCC1L", "TMCC3", "TOB2", "ZNF845", "SREBF1", "HERPUD1", "EPN3", "MAPK4", "ZNF721", "ZNG1F;ZNG1C;ZNG1E", "GMPR", "UQCRH", "CNBP", "FAM110C", "EFCAB7", "UBE2W", "GPNMB", "DDIT4", "MCF2L2", "ZSCAN22", 'COX7A2L', 'NME4', 'DNAJB8']
#Ksenia dn
gene_set = ["TCEA2", "GPSM3", "RINL", "QPCTL", "RAB5IF", "MRPL35", "ELF4", "RDH10", "IFIT3", "PPP1R13B", "HPX", "MLXIP", "NTAQ1", "G6PC3"]

df_cl_col   = 'DepmapModelType'
cancers     = ['TALL', 'BALL', 'AML']
cancers     = 'All'
min_samples = 6
fig_width   = 30
fig_height  = 5
annot_vals  = False
title       ='Cancer dependency (DepMap)'


HeatmapSpecificGenes(gene_set=gene_set, df_cl_col=df_cl_col, cancers=cancers, min_samples=min_samples, fig_width=fig_width, fig_height=fig_height, annotate_values=annot_vals, title=title)

#%% =============================================================================
# Scatter plot of cell lines 
# =============================================================================

# -----------------------------
# DepMap dependency plot config
# -----------------------------

gene                = 'KIF11'

# How to subset cancers
model_type          = 'TALL'            # e.g. "TALL", "AML", None
oncotree_subtype    = None              # e.g. "Acute Myeloid Leukemia"
oncotree_lineage    = None              # e.g. "Lymphoid"
query               = None              # optional pandas query

# Labeling behavior
highlight_lines     = ['MOLM13', 'MOLM14', 'MV411', 'OCIAML2', 'THP1', 'SHI1']  # set() if none
highlight_lines     = []
label_mode          = "all"       # "highlight", "all", "none"
top_n_labels        = 0

# Plot aesthetics
sort_ascending      = True
fig_width           = 5
fig_height          = 4
dpi                 = 200
point_size          = 30
draw_zero_line      = True
title               = None

df_dep_plot = Scatter_CellLines(df_dm=df_dm, df_cl=df_cl, gene=gene, model_type=model_type, oncotree_subtype=oncotree_subtype, oncotree_lineage=oncotree_lineage, query=query, sort_ascending=sort_ascending, highlight=highlight_lines, label=label_mode, top_n_labels=top_n_labels, figsize=(fig_width, fig_height), dpi=dpi, point_size=point_size, yline0=draw_zero_line, title=title,)


#%% =============================================================================
# 10 Expression vs drug sensitivity scan (DepMap/CCLE)
# =============================================================================

gene_of_interest = 'BCL2'
subset_col       = 'DepmapModelType'   # e.g. 'DepmapModelType', 'OncotreeLineage', 'OncotreeSubtype'
subset_values    = None            # list or set to None to include all models
min_n            = 15
corr_method      = 'both'              # 'pearson', 'spearman', or 'both'
split_high_low   = True

# Optional: specified inhibitor scatter plot
plot_single_inhibitor = True
inhibitor_for_plot    = 'Venetoclax'

# Optional PRISM culture filter for 24Q2 data: None, 'PR500A', 'PR500B', or ['PR500A','PR500B']
prism_culture_filter = None
df_drug_scan = RunExpressionDrugSensitivityScan(
    gene_of_interest=gene_of_interest,
    subset_col=subset_col,
    subset_values=subset_values,
    min_n=min_n,
    corr_method=corr_method,
    split_high_low=split_high_low,
    prism_culture_filter=prism_culture_filter,
    plot_single_inhibitor=plot_single_inhibitor,
    inhibitor_for_plot=inhibitor_for_plot,
)


#%% =============================================================================
# 11 Genes correlated with selected inhibitors
# =============================================================================

subset_col_inhibitor_scan    = 'DepmapModelType'
subset_values_inhibitor_scan = ['AML']
min_n_inhibitor_scan         = 2
corr_method_inhibitor_scan   = 'both'

inhibitor_list = ['PIBRENTASVIR']

gene_scans_by_inhibitor = {}
for inhibitor_name in inhibitor_list:
    print(f'Running {inhibitor_name}')
    gene_scans_by_inhibitor[inhibitor_name] = RunInhibitorGeneCorrelationScan(
        inhibitor=inhibitor_name,
        subset_col=subset_col_inhibitor_scan,
        subset_values=subset_values_inhibitor_scan,
        min_n=min_n_inhibitor_scan,
        corr_method=corr_method_inhibitor_scan,
        plot_top_gene=True,
    )

