# DEPMAP_Scanner.py:
This script has two main purposes:
1. Scan a subset of cell lines and output it's dependencies for all genes
2. Scan all cancer types and identify genes with low (most dependent) CRISPR scores.

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
- Section 6: Scan all genes for the lowest median scores for a specific cancer - then create a plot for the n genes with the lowest median scores
- Section 7: Scan all genes and identify the ones for which a specific cancer has the lowest median score
  
<img width="450" alt="image" src="https://github.com/user-attachments/assets/c043075e-1324-4379-8623-d5a1bb872728">
