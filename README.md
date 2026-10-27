# DEPMAP_Scanner.py:
Note: This script requires KTC_functions.py (also on github) to be in the same directory as this file.

## New script: TALL dependency discovery

Use [TALL_dependency_discovery.py](TALL_dependency_discovery.py) to find genes that are selectively essential in TALL while avoiding pan-essential hits.

IDE-first usage (load variables once):
- Open [TALL_dependency_discovery.py](TALL_dependency_discovery.py)
- Execute setup cells once
- Edit the IDE SETTINGS block near the top once
- Run the final "Run analysis" cell whenever you want to refresh results
- Data is cached in memory between runs; use run_ide(force_reload=True) only when input CSV paths change

The script combines:
- Mean and median shift in TALL vs all non-TALL models
- Essential fraction shift using a configurable dependency threshold
- One-sided Mann-Whitney U test with BH/FDR correction
- Global essentiality penalty/filter to remove broad housekeeping dependencies

Output files:
- TALL_dependency_gene_stats.csv: full metrics table for all genes
- TALL_dependency_candidates.csv: filtered high-confidence candidates

## New script: Per-gene relative visualization

Use [TALL_candidate_relative_plots.py](TALL_candidate_relative_plots.py) to inspect candidate genes one-by-one across cancers using per-gene relative effect plots.

This is useful when heatmaps look skewed by shared color scaling.

Example:

        python TALL_candidate_relative_plots.py \
            --dependency-csv /Users/kachrist/Desktop/Interesting_Lists/CRISPRGeneEffect_2025Q3.csv \
            --model-csv /Users/kachrist/Desktop/Interesting_Lists/DepMap_CellLine_annotation_2025Q3.csv \
            --gene-file sweep_runs/strict_and_bcl11b_run_gene_sets.txt \
            --group-column DepmapModelType \
            --target-group TALL \
            --highlight-groups TALL BALL AML \
            --outdir sweep_runs/relative_gene_plots_strict_bcl

Outputs:
- gene_plot_summary.csv (target rank and effect per gene)
- gene_stats/<gene>_cross_cancer_stats.csv (per-cancer table)
- gene_plots/<gene>_cross_cancer_relative.png (relative + absolute panels)

This script has three main purposes:
1. Scan a subset of cell lines and output it's dependencies for all genes
2. Scan all cancer types and identify genes with lowest (most dependent) CRISPR scores.
3. Generate heatmaps to highlight differential dependencies of cancers

## Purpose 1: Example graph (with accompanying .csv):
Using the following command in Section 4:<br>

    Graph_n_write('splicing factors', 'OncotreeLineage', 'Lymphoid')

- The accompanying DEPMAP data is limited to those cell lines marked as 'Lymphoid' in the 'OncetreeLineage' column
- The mean CRISPR score per gene for the qualifying cell lines is calculated and ranked
- The genes in the 'splicing factors' list is visualized and the significance of a potentially skewed distribution is calculated (Mann-Whitney U)

<img width="450" alt="image" src="https://github.com/user-attachments/assets/eae04b06-74b7-4728-9de4-c58aab3cb1f2">
<br>
<img width="320" alt="image" src="https://github.com/user-attachments/assets/c33c371a-20b0-4624-ba19-29f563d49f4c">

## Purpose 2:
Sections 5 through 7 contain functions that can be launched in section 8.
- Section 5: Draw a series of boxplots showing each cancer's dependency on a specific gene
- Section 6: Scan all genes for the lowest median scores for a specific cancer - then create a plot for the n genes with the lowest median scores (regardless of the scores of other cancers)
- Section 7: Scan all genes and identify the ones for which a specific cancer has the lowest median score
  
<img width="450" alt="image" src="https://github.com/user-attachments/assets/c043075e-1324-4379-8623-d5a1bb872728">

## Purpose 3:
Section 9 draws heatmaps of dependencies for several genes and several cancers

### Example #1:

    HeatmapSpecificGenes(KTC_GetGeneSet('m6a_re_wr_er'), cancers=['TLL', 'BLL', 'AML'], fig_width=6, fig_height=6)

Will produce:

<img width="450" alt="image" src="https://github.com/user-attachments/assets/25fb5907-d414-49e9-b965-2ae38a3ac0be">

### Example #2:

    HeatmapSpecificGenes(KTC_GetGeneSet('PRC2'), cancers='All', min_samples=10, fig_width=40, fig_height=6)

Will produce:

<img width="450" alt="image" src="https://github.com/user-attachments/assets/9036c1b2-e7e9-4a96-bd10-ddfb7eca086f">

Section 10 can find relative differences (absolute and relative) between a cancers and others in a set.

### Example #3:

    compare_cancers(target_cancer='AML', comparison_cancers=None, min_samples=10, top_n=20, use_absolute_difference=False, fig_width=30, fig_height=6)

Will produce:

<img width="450" alt="image" src="https://github.com/user-attachments/assets/da2284e2-185a-49a2-b9bd-4213a3e017cc">

### Example #4: 

    compare_cancers(target_cancer='TLL', comparison_cancers=['BLL', 'AML'], min_samples=10, top_n=20, use_absolute_difference=True, fig_width=8, fig_height=6)

Will produce:

<img width="450" alt="image" src="https://github.com/user-attachments/assets/890cef1d-3466-4cf1-ab47-b53e96dcc349">

Section 11 produces a graph on relative differences (BETA)

### Example #5:
<img width="450" alt="image" src="https://github.com/user-attachments/assets/9d5f9aa1-929e-493f-9cef-efa6578bfd47">






