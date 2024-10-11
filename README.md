# DEPMAP_Scanner.py:
This purpose of this script is to explore DEPMAP data for subsets of cell lines and optionally, the dependency of sets of genes.

## Example graph (with accompanying .csv):
Using the following command in Section 4 :<br>

    Graph_n_write('splicing factors', 'OncotreeLineage', 'Lymphoid')

- The accompanying DEPMAP data is limited to those cell lines marked as 'Lymphoid' in the 'OncetreeLineage' column
- The mean CRISPR score per gene for the qualifying cell lines is calculated and ranked
- The genes in the 'splicing factors' list is visualized and the significance of a potential skewed distribution is calculated (Mann-Whitney U)

<img width="450" alt="image" src="https://github.com/user-attachments/assets/eae04b06-74b7-4728-9de4-c58aab3cb1f2">
<br>
<img width="320" alt="image" src="https://github.com/user-attachments/assets/c33c371a-20b0-4624-ba19-29f563d49f4c">
