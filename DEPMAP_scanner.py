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

# in_dm = '/Users/kasperthorhaugechristensen/Downloads/dependency_all.csv' # older data set
in_dm = '/Volumes/cmgg_pnlab/Kasper/Data/Interesting_Lists/CRISPRGeneEffect.csv'
in_cl = '/Volumes/cmgg_pnlab/Kasper/Data/Interesting_Lists/Depmap_CellLine_annotation.csv'
out   = '/Users/kasperthorhaugechristensen/Desktop/Dumpbox'

df_dm = pd.read_csv(in_dm, low_memory=False)
df_cl = pd.read_csv(in_cl)

df_dm.rename(columns={df_dm.columns[0]: 'Depmap Id'}, inplace=True)
df_dm.columns = df_dm.columns.str.replace(r'\s\(\d+\)', '', regex=True)


#%% ===========================================================================
# 3 Function: Plot all dependencies for one cancer
# =============================================================================

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


    # print("Highlight scores:", highlight_scores)
    # print("Background scores:", background_scores)
    # print("Number of highlight scores:", len(highlight_scores))
    # print("Number of background scores:", len(background_scores))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    out_file = os.path.join(out, 'DEPMAP_%s_filtered.csv' %(filter_content))
    df_means.to_csv(out_file, index=False)

#%% ===========================================================================
# 4 Main functionality
# =============================================================================

# Graph_n_write takes three arguments"
    # a name for a gene set (genes in gene set will be determined by GetGeneSet())
    # a column in the dataframe for the description of the cell lines (e.g. OncotreeLineage or OncotreeCode)
    # a value in the given column to consider a hit for inclusion purposes (e.g. BRCA or TLL)
Graph_n_write('Kevin', 'OncotreeCode', 'TLL')


#%% ===========================================================================
# 5 Function: Draw boxplot of all cancer dependencies for one gene
# =============================================================================

import seaborn as sns

def PlotOneGene(gene, highlighted_cancer_types):
    # Step 1: Select only the columns for DepmapID and the gene of interest from df_dm
    df_dm_gene = df_dm[['Depmap Id', gene]].rename(columns={'Depmap Id': 'ModelID'})
    
    # Step 2: Merge the gene dependency data with the clinical mapping data
    df_merged = pd.merge(df_dm_gene, df_cl[['ModelID', 'OncotreeSubtype']], on='ModelID')
    
    # Step 3: Calculate medians by OncotreeSubtype and sort them
    median_sorted = df_merged.groupby('OncotreeSubtype')[gene].median().sort_values().index
    
    # Define cancer types to recolor and their corresponding colors
    color_palette = ['#d62728' if subtype in highlighted_cancer_types else '#1f77b4' for subtype in median_sorted]
    
    # Step 4: Create boxplots with custom colors for selected cancer types
    plt.figure(figsize=(30, 8))
    sns.boxplot(x='OncotreeSubtype', y=gene, data=df_merged, order=median_sorted, palette=color_palette)
    plt.xticks(rotation=90)
    plt.title(f'Dependency Factor of {gene} Across Cancer Types')
    plt.xlabel('OncotreeSubtype')
    plt.ylabel(f'{gene} Dependency Factor')
    plt.show()


#%% ===========================================================================
# 6 Function: Scan all genes and find the lowest (most dependent) scores for a cancer - Then plot it
# =============================================================================


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



#%% ===========================================================================
# 7 Function: Scan all genes and find those were a specific cancer type has the lowest score - then plot them
# =============================================================================

def FindGenesWhereCancerTypeHasLowestMedian(target_cancer_type):
    # Merge df_dm with df_cl to include OncotreeSubtype for each ModelID
    df_dm_merged = pd.merge(
        df_dm.rename(columns={'Depmap Id': 'ModelID'}), 
        df_cl[['ModelID', 'OncotreeSubtype']], 
        on='ModelID'
    )
    
    # Ensure that all gene expression columns are numeric, convert non-numeric values to NaN
    gene_columns = df_dm_merged.columns.drop(['ModelID', 'OncotreeSubtype'])
    df_dm_merged[gene_columns] = df_dm_merged[gene_columns].apply(pd.to_numeric, errors='coerce')
    
    # Initialize a list to store genes where target_cancer_type has the lowest median
    genes_with_lowest_in_target = []
    
    # Get unique cancer types to handle the case where target_cancer_type might not exist
    unique_cancer_types = df_dm_merged['OncotreeSubtype'].unique()
    
    # Check if the target cancer type exists in the dataset
    if target_cancer_type not in unique_cancer_types:
        print(f"Warning: {target_cancer_type} is not found in the dataset.")
        return genes_with_lowest_in_target  # Return an empty list if the target cancer type doesn't exist
    
    # Iterate over each gene column to check if the target cancer type has the lowest score
    for gene in gene_columns:
        # Group by OncotreeSubtype and calculate the median for the current gene
        medians_by_cancer = df_dm_merged.groupby('OncotreeSubtype')[gene].median()
        
        # Check if the target cancer type has the lowest median score
        if medians_by_cancer[target_cancer_type] == medians_by_cancer.min():
            genes_with_lowest_in_target.append(gene)
    
    return genes_with_lowest_in_target

# Function to plot each gene in the list
def PlotGenesWithLowestMedian(target_cancer_type, genes_to_plot):
    for gene in genes_to_plot:
        PlotOneGene(gene, [target_cancer_type])



#%% ===========================================================================
# 8 Boxplot analysis proper
# =============================================================================

target_cancer_type = 'Acute Myeloid Leukemia'
target_cancer_type = 'T-Lymphoblastic Leukemia/Lymphoma'
highlighted_cancer_types = [target_cancer_type] #Add more if more should be highlighted

#%% Plot for a single gene
gene = 'LEF1'  # Used for plotting a single gene
PlotOneGene(gene, highlighted_cancer_types)

#%% Find the top n lowest scores (regardless of the score of other cancers) for a specific cancer and plot them
top_n_genes = FindTopNGenesWithLowestMedianForCancerType(target_cancer_type, n=5)
for _gene in top_n_genes:
    PlotOneGene(_gene, highlighted_cancer_types)

#%% Find all genes where the specified cancer type has the lowest median score
genes_with_lowest_in_target = FindGenesWhereCancerTypeHasLowestMedian(target_cancer_type)
# Plot each of these genes
PlotGenesWithLowestMedian(target_cancer_type, genes_with_lowest_in_target)
print(f'Genes for which {target_cancer_type} has the lowest score:')
for _gene in genes_with_lowest_in_target:
    print(f'\t{_gene}')

#%% ===========================================================================
# 9 Heatmaps for specific genes and specific (or all) cancers
# =============================================================================

def HeatmapSpecificGenes(gene_set, cancers='All', min_samples=10, fig_width=8, fig_height=6, annotate_values=True):
    dict_input = {}
    if cancers=='All':
        qualifying_cancers = df_cl['OncotreeCode'].value_counts()[df_cl['OncotreeCode'].value_counts()>min_samples].index.tolist()
        for cancer in qualifying_cancers:
            # if cancer not in dict_input:
            dict_input[cancer] = ['OncotreeCode', cancer]
    else:
        for cancer in cancers:
            dict_input[cancer] = ['OncotreeCode', cancer]

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
    sns.heatmap(mean_expression_data, cmap='coolwarm_r', annot=annotate_values)
    plt.title('Cancer dependency (DepMap)')
    # plt.xlabel('Cancer Types')
    # plt.ylabel('Genes')
    plt.show()

HeatmapSpecificGenes(KTC_GetGeneSet('WTAP'), cancers=['AML', 'BLL', 'TLL'], fig_width=4, fig_height=6, annotate_values=True)
HeatmapSpecificGenes(KTC_GetGeneSet(['DHFR', 'NAMPT', 'IDO1', 'NAPRT']), cancers=['AML', 'BLL', 'TLL'], fig_width=4, fig_height=6, annotate_values=True)
# HeatmapSpecificGenes(KTC_GetGeneSet('Kevin'), cancers=['TLL', 'BLL', 'AML'], fig_width=4, fig_height=5)
# HeatmapSpecificGenes(KTC_GetGeneSet('Kevin'), cancers='All', fig_width=40, fig_height=8, min_samples=10)
# HeatmapSpecificGenes(GetGeneSet('PRC2'), cancers='All', min_samples=10, fig_width=40, fig_height=6)

#%% ===========================================================================
# 10 Heatmaps - finding the top absolute differential dependencies between one cancer and a set (or all) others
# =============================================================================
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dict_input = {
    'AML' : ['OncotreeCode', 'AML'],
    'B-ALL' : ['OncotreeCode', 'BLL'],
    'T-ALL' : ['OncotreeCode', 'TLL']
    }

from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compare_cancers(target_cancer, comparison_cancers=None, min_samples=10, top_n=20, use_absolute_difference=False, fig_width=30, fig_height=6):
    """
    Generalized function to compare one specific cancer to the mean of a subset of cancers or all cancers.
    
    Parameters:
    - df_cl: Clinical dataframe.
    - df_dm: Data matrix dataframe.
    - target_cancer: The cancer to compare (e.g., 'T-ALL').
    - comparison_cancers: List of cancers to compare against. If None, use all cancers with more than min_samples.
    - min_samples: Minimum samples required for a cancer to be included.
    - top_n: Number of top genes to display in the plot.
    - use_absolute_difference: If True, sort genes by absolute difference. If false, sort by most negative (dependent)
    - fig_width: Width of the figure.
    - fig_height: Height of the figure.
    """
    # If comparison_cancers is None, include all cancers with at least min_samples
    if comparison_cancers is None:
        comparison_cancers = (
            df_cl['OncotreeCode']
            .value_counts()
            [df_cl['OncotreeCode'].value_counts() > min_samples]
            .index.tolist()
        )
        if target_cancer in comparison_cancers:
            comparison_cancers.remove(target_cancer)

    # Create a dictionary of cancers for filtering
    dict_input = {cancer: ['OncotreeCode', cancer] for cancer in comparison_cancers + [target_cancer]}

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

        # Append results to mean_expression_data with cancer type as index
        mean_expression_data[cancer] = mean_values

    # Remove genes with NaN values in any cancer type
    mean_expression_data_clean = mean_expression_data.dropna()

    if mean_expression_data_clean.empty:
        print("Error: No valid gene expression data available for plotting. Check your data.")
        return

    # Calculate differences for target cancer vs the mean of the comparison cancers
    differences = {}
    for gene in tqdm(mean_expression_data_clean.index, desc="Calculating differences"):
        target_value = mean_expression_data_clean.loc[gene, target_cancer]
        mean_comparison = mean_expression_data_clean.loc[gene, comparison_cancers].mean()
        if use_absolute_difference:
            differences[gene] = abs(target_value - mean_comparison)  # Absolute difference
        else:
            differences[gene] = target_value - mean_comparison  # Negative difference

    # Sort genes based on the chosen difference type and select top genes
    sorted_genes = sorted(differences.items(), key=lambda x: x[1], reverse=use_absolute_difference)

    # Extract only the gene names from the sorted list
    top_genes = [gene for gene, _ in sorted_genes][:top_n]

    # Subset data for the top genes
    top_genes_data = mean_expression_data_clean.loc[top_genes]

    if top_genes_data.empty:
        print(f"Error: No valid data for the top {top_n} genes.")
        return

    # Add a column for the calculated differences
    top_genes_data[f'{target_cancer} vs Comparison Mean'] = [
        differences[gene] for gene in top_genes
    ]

    # Plot the heatmap
    plt.figure(figsize=(fig_width, fig_height), dpi=100)
    sns.heatmap(
        top_genes_data,
        cmap='coolwarm_r',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Expression'}
    )
    if len(comparison_cancers) >= 5:
        comparison_cancers = 'others'
    title = f'Top {top_n} Genes: {target_cancer} vs {comparison_cancers} ( Largest {"absolute" if use_absolute_difference else "negative"} difference)'
    plt.title(title, fontsize=14)
    plt.xlabel('Cancer Types', fontsize=12)
    plt.ylabel('Genes', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.show()

# Usage
compare_cancers(target_cancer='AML', comparison_cancers=None, min_samples=10, top_n=20, use_absolute_difference=False, fig_width=30, fig_height=6)
compare_cancers(target_cancer='TLL', comparison_cancers=['BLL', 'AML'], min_samples=10, top_n=20, use_absolute_difference=True, fig_width=8, fig_height=6)

#%% ===========================================================================
# Section 11: Same as above, but ranked by relative differences instead of absolute differences
# =============================================================================
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Boolean to toggle between absolute difference or most negative difference
top_n = 20  # After sorting, how many genes are plotted
expression_threshold = 0.03  # Threshold for absolute expression value

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
    
    # Append results to mean_expression_data with cancer type as index
    mean_expression_data[cancer] = mean_values

# Remove genes with NaN values in any cancer type
mean_expression_data_clean = mean_expression_data.dropna()

# Filter genes based on absolute expression threshold for all cancer types (T-ALL, AML, B-ALL)
filtered_mean_expression_data = mean_expression_data_clean[
    (mean_expression_data_clean['T-ALL'].abs() >= expression_threshold) &
    (mean_expression_data_clean['AML'].abs() >= expression_threshold) &
    (mean_expression_data_clean['B-ALL'].abs() >= expression_threshold)
]

# Calculate relative differences for T-ALL vs AML and B-ALL
differences = {}

for gene in tqdm(filtered_mean_expression_data.index, desc="Calculating relative differences"):
    tall_value = filtered_mean_expression_data.loc[gene, 'T-ALL']
    mean_aml_b_all = filtered_mean_expression_data.loc[gene, ['AML', 'B-ALL']].mean()
    
    # Calculate relative difference (percentage difference)
    if mean_aml_b_all != 0:
        relative_difference = ((tall_value - mean_aml_b_all) / mean_aml_b_all) + 1
    else:
        relative_difference = 0  # If the mean of AML and B-ALL is 0, avoid division by zero
    
    differences[gene] = relative_difference

# Filter genes based on the threshold for absolute relative difference
filtered_differences = {gene: diff for gene, diff in differences.items() if abs(diff) >= expression_threshold}

# Sort genes based on the relative difference and select top genes
sorted_genes = sorted(filtered_differences.items(), key=lambda x: x[1], reverse=True)  # False to find most negative

# Extract only the gene names from the sorted list
top_genes = [gene for gene, _ in sorted_genes][:top_n]

# Subset data for the top genes
top_genes_data = filtered_mean_expression_data.loc[top_genes]

# Add a column for the calculated relative differences
top_genes_data['relative T-ALL vs Mean(AML & B-ALL)'] = [
    filtered_differences[gene] for gene in top_genes
]

# Create a mask where the difference column is True (hidden for coloring)
mask = top_genes_data.columns == 'relative T-ALL vs Mean(AML & B-ALL)'
mask = [False, False, False, True]  # Set 'True' for the last column to mask its color

# Plot the heatmap with the new difference column
plt.figure(figsize=(5, 6), dpi=100)  # Adjust width for the additional column

# Create a custom color map for the gene expression columns
cmap = sns.diverging_palette(220, 20, as_cmap=True)  # Diverging color palette
cmap_reversed = cmap.reversed()  # Reverse the color map

sns.heatmap(
    top_genes_data,
    cmap=cmap_reversed,  # Apply color map for expression data
    annot=True,
    fmt='.2f',  # Control formatting for annotations
    linewidths=0.5,
    cbar_kws={'label': 'Expression'},
    mask=mask,  # Mask the difference column for coloring
    linecolor='white',  # White line between cells
    annot_kws={"size": 10, 'weight': 'normal'}  # Make the annotations bold and readable
)

# Overlay annotations for the 'T-ALL vs Mean(AML & B-ALL)' column
for i, gene in enumerate(top_genes):
    # Positioning the text to be centered in the last column
    plt.text(
        x=len(top_genes_data.columns)-0.5,  # Last column
        y=i + 0.5,  # Centered in the row
        s=f'{top_genes_data.loc[gene, "relative T-ALL vs Mean(AML & B-ALL)"]:.2f}',
        ha='center', va='center', fontsize=10, color='black', weight='normal'
    )

# Plot styling
title =f'Top {top_n} Genes: T-ALL vs AML & B-ALL (largest fold difference)'
plt.title(title)
plt.xlabel(f'Cancer Types and Difference (min cutoff: {expression_threshold})')
plt.ylabel('Genes')
plt.show()


