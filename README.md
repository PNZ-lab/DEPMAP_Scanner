# DEPMAP_Scanner.py:
Note: This script requires KTC_functions.py (also on github) to be in the same directory as this file.

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

    HeatmapSpecificGenes(GetGeneSet('m6a_re_wr_er'), cancers=['TLL', 'BLL', 'AML'], fig_width=6, fig_height=6)

Will produce:

<img width="450" alt="image" src="https://github.com/user-attachments/assets/25fb5907-d414-49e9-b965-2ae38a3ac0be">

### Example #2:

    HeatmapSpecificGenes(GetGeneSet('PRC2'), cancers='All', min_samples=10, fig_width=40, fig_height=6)

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






