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
from scipy.stats import mannwhitneyu
import numpy as np
from KTC_functions import KTC_GetGeneSet
from tqdm import tqdm
from adjustText import adjust_text



in_dir = '/Volumes/kachrist/shares/cmgg_pnlab/Kasper/Data/Interesting_Lists'

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
        desired_order = ['AML', 'BLL', 'TLL']
        
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
                ax.scatter(i, row[gene], s=point_size * 3)

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

gene = 'MTOR'
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


gene_set    = KTC_GetGeneSet('SG')
df_cl_col   = 'DepmapModelType'
cancers     = ['TALL', 'BALL', 'AML']
min_samples = 6
fig_width   = 3
fig_height  = 4
annot_vals  = False
title       ='Cancer dependency (DepMap)'


HeatmapSpecificGenes(gene_set=gene_set, df_cl_col=df_cl_col, cancers=cancers, min_samples=min_samples, fig_width=fig_width, fig_height=fig_height, annotate_values=annot_vals, title=title)

#%% =============================================================================
# Scatter plot of cell lines 
# =============================================================================

# -----------------------------
# DepMap dependency plot config
# -----------------------------

gene                = "MEN1"

# How to subset cancers
model_type          = 'TALL'            # e.g. "TALL", "AML", None
oncotree_subtype    = None              # e.g. "Acute Myeloid Leukemia"
oncotree_lineage    = None              # e.g. "Lymphoid"
query               = None              # optional pandas query

# Labeling behavior
highlight_lines     = []  # set() if none
label_mode          = "all"       # "highlight", "all", "none"
top_n_labels        = 0

# Plot aesthetics
sort_ascending      = True
fig_width           = 8
fig_height          = 5
dpi                 = 200
point_size          = 30
draw_zero_line      = True
title               = "dependency (DepMap)"

df_dep_plot = Scatter_CellLines(df_dm=df_dm, df_cl=df_cl, gene=gene, model_type=model_type, oncotree_subtype=oncotree_subtype, oncotree_lineage=oncotree_lineage, query=query, sort_ascending=sort_ascending, highlight=highlight_lines, label=label_mode, top_n_labels=top_n_labels, figsize=(fig_width, fig_height), dpi=dpi, point_size=point_size, yline0=draw_zero_line, title=title,)




