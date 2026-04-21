import os
import pandas as pd

# directory with cleaned TPM file
base_dir = "/Users/kachrist/Downloads/aml_target_gdc"

# -----------------------------
# 1) Load patient-level clinical
# -----------------------------
path_clin_patient = os.path.join(base_dir, "data_clinical_patient.txt")
df_clin_patient = pd.read_csv(
    path_clin_patient,
    sep="\t",
    comment="#",
    low_memory=False
)
df_clin_patient.set_index("PATIENT_ID", inplace=True)

# -----------------------------
# 2) Load sample-level clinical
# -----------------------------
path_clin_sample = os.path.join(base_dir, "data_clinical_sample.txt")
df_clin_sample = pd.read_csv(
    path_clin_sample,
    sep="\t",
    comment="#",
    low_memory=False
)
df_clin_sample.set_index("SAMPLE_ID", inplace=True)

# ---------------------------------------------
# 3) Load the *mapped* expression matrix (TPM)
# ---------------------------------------------
path_expr = os.path.join(base_dir, "data_mrna_seq_tpm_mapped.tsv")
df_expr_rna = pd.read_csv(path_expr, sep="\t", index_col=0)

# Ensure index is string (safety)
df_expr_rna.index = df_expr_rna.index.astype(str)

# ------------------------------
# 4) Merge clinical data
# ------------------------------
df_clin_merged_aml = (
    df_clin_sample.reset_index()
    .merge(df_clin_patient.reset_index(),
           on="PATIENT_ID",
           suffixes=("_SAMPLE", "_PATIENT"))
    .set_index("SAMPLE_ID")
)

# ------------------------------
# 5) Sanity checks
# ------------------------------
expr_samples = set(df_expr_rna.columns)
clin_samples = set(df_clin_merged_aml.index)

print("Expression samples:", len(expr_samples))
print("Clinical samples:", len(clin_samples))
print("Intersection:", len(expr_samples & clin_samples))

# Optional: reduce expression matrix to only samples with clinical data
common = list(expr_samples & clin_samples)
df_expr_rna = df_expr_rna[common]

print("Final expression shape:", df_expr_rna.shape)
print("Final clinical shape:", df_clin_merged_aml.shape)


#%%
def get_gene_expr(gene, expr_df, clin_df):
    """
    Return a DataFrame with one column 'expr' plus clinical columns,
    indexed by SAMPLE_ID.
    """
    if gene not in expr_df.index:
        raise ValueError(f"Gene {gene} not found in expression matrix.")
    
    expr = expr_df.loc[gene]
    # expr is a Series with index = SAMPLE_ID
    df = pd.DataFrame({"expr": expr})
    # join clinical by SAMPLE_ID
    df = df.join(clin_df, how="inner")
    return df

def gene_correlation(genes, expr_df):
    """
    Small correlation matrix for a list of genes.
    """
    genes_present = [g for g in genes if g in expr_df.index]
    mat = expr_df.loc[genes_present].T   # samples × genes
    return mat.corr()

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd

def km_plot_for_gene(gene,
                     expr_df,
                     clin_merged_df,
                     time_col="OS_MONTHS",
                     status_col="OS_STATUS",
                     split="median",
                     title=None):
    # --- build KM table ---
    if gene not in expr_df.index:
        raise ValueError(f"Gene {gene} not found in expression matrix.")

    # expression vector: index = SAMPLE_ID
    expr = expr_df.loc[gene]

    # join to clinical by SAMPLE_ID
    df = pd.DataFrame({"expr": expr}).join(clin_merged_df, how="inner").copy()

    # survival time
    df["time"] = df[time_col]

    # parse status like '0:LIVING' / '1:DECEASED' -> numeric 0/1
    status_raw = df[status_col].astype(str)
    df["status"] = (
        status_raw.str.extract(r"^(\d)", expand=False)
                  .astype(float)
    )

    # group by expression
    if split == "median":
        cutoff = df["expr"].median()
        df["group"] = (df["expr"] >= cutoff).map({True: "high", False: "low"})
    else:
        raise NotImplementedError("Only median split is implemented for now.")

    df_km = df[["time", "status", "group", "expr"]].dropna(subset=["time", "status"])

    # --- plot KM ---
    plt.figure(dpi=200)

    kmf = KaplanMeierFitter()

    # one curve per group
    for group_name, df_sub in df_km.groupby("group"):
        kmf.fit(
            durations=df_sub["time"],
            event_observed=df_sub["status"],
            label=str(group_name)
        )
        kmf.plot(ci_show=True)

    # logrank p if 2 groups
    if df_km["group"].nunique() == 2:
        groups = list(df_km["group"].unique())
        g1, g2 = groups[0], groups[1]
        df_g1 = df_km[df_km["group"] == g1]
        df_g2 = df_km[df_km["group"] == g2]

        res = logrank_test(
            df_g1["time"], df_g2["time"],
            event_observed_A=df_g1["status"],
            event_observed_B=df_g2["status"]
        )
        p = res.p_value

        if title is None:
            title = f"{gene} high vs low, logrank p = {p:.3e}"

    if title is not None:
        plt.title(title)

    plt.xlabel("Time (months)")
    plt.ylabel("Overall survival probability")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df_km

# df_expr_rna is the symbol-indexed TPM matrix loaded earlier
df_exp_matrix = df_expr_rna.copy()
df_exp_matrix = df_exp_matrix.apply(pd.to_numeric, errors='coerce')
df_exp_matrix = df_exp_matrix.dropna(how="all")   # remove empty rows
df_exp_matrix.index  # gene symbols
df_exp_matrix.columns  # sample IDs

import numpy as np
from scipy.stats import pearsonr

def _safe_pearsonr(a: np.ndarray, b: np.ndarray):
    """Pearson r with NaN/constant guards; returns (r, p) or (np.nan, np.nan)."""
    m = np.isfinite(a) & np.isfinite(b)
    ax, bx = a[m], b[m]
    if ax.size < 3:
        return np.nan, np.nan
    if np.std(ax) == 0 or np.std(bx) == 0:
        return np.nan, np.nan
    return pearsonr(ax, bx)


import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.stats import mannwhitneyu

def WaterfallPlot(sorted_genes,
                  target_gene,
                  gene_set=None,
                  label='',
                  show_breakpoint=True,
                  print_corr_genes=False):
    """
    sorted_genes: list of (gene, (r, p)) sorted descending by r.
    gene_set: list of genes to highlight (can be empty or None).
    """
    genes, r_p_values = zip(*sorted_genes)
    r_values = [r for r, p in r_p_values]
    ranks = np.arange(1, len(genes) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    plt.scatter(ranks, r_values, color='black', s=2, zorder=2,
                label='gene expression correlations')

    # Highlight a gene set, if provided
    highlight_genes = gene_set or []
    if len(highlight_genes) != 0:
        highlight_indices = [i for i, gene in enumerate(genes)
                             if gene in highlight_genes]
        highlight_ranks = np.array(ranks)[highlight_indices]
        highlight_r_values = np.array(r_values)[highlight_indices]

        background_r_values = [r for r in r_values
                               if r not in highlight_r_values]

        if len(highlight_r_values) > 0 and len(background_r_values) > 0:
            stat, p_value = mannwhitneyu(highlight_r_values,
                                         background_r_values,
                                         alternative='two-sided')
        else:
            stat, p_value = np.nan, np.nan

        for rank, r_value in zip(highlight_ranks, highlight_r_values):
            plt.vlines(x=rank,
                       ymin=min(r_values),
                       ymax=max(r_values),
                       color='red',
                       linewidth=0.2,
                       zorder=1,
                       label=f'{label} (U={stat:.0f}, p={p_value:.3g})')

    plt.xlabel('Rank of gene–gene correlation', fontsize=16)
    plt.ylabel("Pearson's R", fontsize=16)
    plt.title(f'{target_gene} : Waterfall plot of gene correlations',
              fontsize=22)

    # Knee points
    kn_positive = KneeLocator(ranks, r_values,
                              curve='convex', direction='decreasing')
    kn_negative = KneeLocator(ranks, r_values,
                              curve='concave', direction='decreasing')

    if kn_positive.knee is not None:
        if show_breakpoint:
            plt.axvline(x=kn_positive.knee, color='blue', linestyle='--',
                        label='upper breakpoint')
        genes_above_elbow = [genes[i] for i in range(kn_positive.knee)]
    else:
        genes_above_elbow = []

    if kn_negative.knee is not None:
        if show_breakpoint:
            plt.axvline(x=kn_negative.knee, color='blue', linestyle='--')
        genes_below_elbow = [genes[i] for i in range(kn_negative.knee,
                                                     len(genes))]
    else:
        genes_below_elbow = []

    # unique legend entries
    handles, labels_ = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10)

    plt.tight_layout()
    plt.show()

    if print_corr_genes:
        print("Genes above upper elbow:")
        for g in genes_above_elbow:
            print(g)
        print("\nGenes below lower elbow:")
        for g in genes_below_elbow:
            print(g)

    return genes_above_elbow, genes_below_elbow
from tqdm import tqdm

def scan_gene_correlations(
    gene,
    df_expr,
    gene_set=None,
    label='',
    top_n=50,
    out_dir=None,
    project='TARGET_AML',
    write_csv=True
):
    """
    Pearson correlations of 'gene' vs all others in df_expr.

    Returns:
        top_hits, bottom_hits : lists of (gene, (r, p)) of length ~top_n/2
    """
    # 1) Build numeric matrix with genes as index
    df_num = df_expr.copy()
    df_num = df_num.apply(pd.to_numeric, errors='coerce')
    df_num = df_num.dropna(how='all')

    if gene not in df_num.index:
        raise ValueError(f"Gene '{gene}' not found in expression matrix.")

    gene_values = df_num.loc[gene].to_numpy(dtype=float)

    r_dict = {}
    all_genes = df_num.index.tolist()

    # 2) Compare against every other gene
    for other_gene in tqdm(all_genes,
                           desc=f'Pearson scan for {gene}',
                           unit='gene'):
        if other_gene == gene:
            continue
        other_vals = df_num.loc[other_gene].to_numpy(dtype=float)
        r_value, pearson_p = _safe_pearsonr(gene_values, other_vals)
        r_dict[other_gene] = (r_value, pearson_p)

    # 3) Sort by r (descending), NaNs at end
    def sort_key(item):
        r = item[1][0]
        return -1e9 if np.isnan(r) else r

    sorted_genes = sorted(r_dict.items(), key=sort_key, reverse=True)

    # 4) WaterfallPlot + elbow gene sets
    genes_above_elbow, genes_below_elbow = WaterfallPlot(
        sorted_genes,
        target_gene=gene,
        gene_set=gene_set,
        label=label,
        show_breakpoint=True,
        print_corr_genes=False
    )

    # 5) CSV output with elbow categories
    if write_csv and out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        for g_name, (r_value, p_value) in sorted_genes:
            if g_name in genes_above_elbow:
                category = 'above_1st_elbow'
            elif g_name in genes_below_elbow:
                category = 'below_2nd_elbow'
            else:
                category = 'neither'
            rows.append([g_name, r_value, p_value, category])

        df_result = pd.DataFrame(
            rows,
            columns=['Gene', 'Pearson_r', 'p_value', 'Category']
        )
        out_path = os.path.join(out_dir,
                                f'{project}_{gene}_correlation_data.csv')
        df_result.to_csv(out_path, index=False)
        print(f'File created: {out_path}')

    # 6) Return top/bottom N like your Polonen function
    n = int(round(top_n / 2))
    top_hits = sorted_genes[:n]
    bottom_hits = sorted_genes[-n:]

    return top_hits, bottom_hits, sorted_genes

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import numpy as np

def Grapher_TARGET(gene1, gene2, df_expr, title_prefix="TARGET-AML"):
    """
    Scatterplot + Pearson r line for two genes using df_expr (rows=genes, cols=samples).
    """
    if gene1 not in df_expr.index:
        raise ValueError(f"{gene1} not in expression matrix.")
    if gene2 not in df_expr.index:
        raise ValueError(f"{gene2} not in expression matrix.")

    x = df_expr.loc[gene1].values.astype(float)
    y = df_expr.loc[gene2].values.astype(float)

    # Compute Pearson
    r, p = pearsonr(x, y)

    # Regression line
    X = x.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    Y_pred = model.predict(X)

    # Plot
    plt.figure(figsize=(7, 6), dpi=200)
    plt.scatter(x, y, color='black', alpha=0.2, s=20)
    plt.plot(x, Y_pred, color='red')

    plt.xlabel(f"{gene1} expression", fontsize=14)
    plt.ylabel(f"{gene2} expression", fontsize=14)
    plt.title(f"{title_prefix}: {gene1} vs {gene2}\n"
              f"Pearson r = {r:.2f}, p = {p:.2e}", fontsize=16)
    plt.tight_layout()
    plt.show()

    return r, p

def scan_km_for_genes(expr_df,
                      clin_df,
                      genes=None,
                      time_col="OS_MONTHS",
                      status_col="OS_STATUS",
                      split="median",
                      min_group_size=10,
                      min_events=5,
                      out_dir=None,
                      project="TARGET_AML",
                      out_prefix="KM_scan"):

    # 1) Align samples across expression and clinical data
    common_samples = expr_df.columns.intersection(clin_df.index)
    if len(common_samples) == 0:
        raise ValueError("No overlapping SAMPLE_IDs between expr_df and clin_df.")

    expr = expr_df[common_samples].copy()
    clin = clin_df.loc[common_samples].copy()

    # survival time
    clin["time"] = pd.to_numeric(clin[time_col], errors="coerce")

    # parse status "0:LIVING" / "1:DECEASED" -> numeric 0/1
    status_raw = clin[status_col].astype(str)
    clin["status"] = (
        status_raw.str.extract(r"^(\d)", expand=False)
                  .astype(float)
    )

    # global mask of valid survival entries
    surv_ok = np.isfinite(clin["time"].values) & np.isfinite(clin["status"].values)

    # 2) Gene list
    if genes is None:
        genes = expr.index.tolist()
    else:
        # keep only genes that actually exist
        genes = [g for g in genes if g in expr.index]

    rows = []

    # 3) Loop over genes
    for gene in tqdm(genes, desc="KM scan over genes", unit="gene"):
        values = pd.to_numeric(expr.loc[gene], errors="coerce").values

        # combine valid survival + valid expression
        mask = surv_ok & np.isfinite(values)
        if mask.sum() < 2 * min_group_size:
            continue

        vals = values[mask]
        times = clin["time"].values[mask]
        status = clin["status"].values[mask]

        # group by expression
        if split == "median":
            cutoff = np.median(vals)
            group_high = vals >= cutoff
        else:
            raise NotImplementedError("Only median split is implemented for now.")

        n_high = int(group_high.sum())
        n_low = int((~group_high).sum())

        if n_high < min_group_size or n_low < min_group_size:
            continue

        # durations & events per group
        t_high = times[group_high]
        t_low = times[~group_high]
        e_high = status[group_high]
        e_low = status[~group_high]

        events_high = float(e_high.sum())
        events_low = float(e_low.sum())
        total_events = events_high + events_low

        if total_events < min_events:
            continue

        # logrank test
        try:
            res = logrank_test(
                t_high, t_low,
                event_observed_A=e_high,
                event_observed_B=e_low
            )
            p = float(res.p_value)
            stat = float(res.test_statistic)
        except Exception:
            # occasionally lifelines can choke on edge cases
            continue

        # directional info: is high expression associated with worse outcome?
        event_rate_high = events_high / n_high if n_high > 0 else np.nan
        event_rate_low = events_low / n_low if n_low > 0 else np.nan

        if np.isfinite(event_rate_high) and np.isfinite(event_rate_low):
            if event_rate_high > event_rate_low:
                direction = "high_worse"
            elif event_rate_high < event_rate_low:
                direction = "high_better"
            else:
                direction = "no_diff"
        else:
            direction = "NA"

        rows.append({
            "gene": gene,
            "p_value": p,
            "logrank_statistic": stat,
            "n_high": n_high,
            "n_low": n_low,
            "events_high": events_high,
            "events_low": events_low,
            "event_rate_high": event_rate_high,
            "event_rate_low": event_rate_low,
            "direction": direction,
            "cutoff_median_expr": cutoff
        })

    if not rows:
        raise RuntimeError("No genes produced valid KM statistics.")

    df_results = pd.DataFrame(rows)
    df_results = df_results.sort_values("p_value").reset_index(drop=True)

    # 4) Optional: add FDR (if statsmodels is installed)
    try:
        from statsmodels.stats.multitest import multipletests
        _, qvals, _, _ = multipletests(df_results["p_value"].values,
                                       method="fdr_bh")
        df_results["q_value"] = qvals
    except Exception:
        # statsmodels not installed or something else went wrong
        pass

    # 5) Write CSV
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                                f"{project}_{out_prefix}.csv")
        df_results.to_csv(out_path, index=False)
        print(f"KM scan results written to: {out_path}")

    return df_results

#%% Plot a single gene-gene correlation

gene_x = 'EZH2'
gene_y = 'METTL3'

Grapher_TARGET(
    gene_x,
    gene_y,
    df_expr_rna
    )


#%% Plot a single KM

gene = 'HNRNPA1'

df_km = km_plot_for_gene(
    gene,
    expr_df=df_expr_rna,         # TPM matrix with symbols as index
    clin_merged_df=df_clin_merged_aml,
    time_col="OS_MONTHS",        # <- use months
    status_col="OS_STATUS"       # <- parse 0/1 from this
)

from KTC_functions import KTC_multi_enrichment_plot
from KTC_functions import KTC_make_list

gene_list = KTC_make_list('''
                          GALE
COMTD1
PLA2G2C
MT1E
CAP1P1
FAM50A
TAGLN2
HSD17B10
SRGAP2
MYOF
SRGAP2D
TUBB6
PSMA7
NAPSB
ANKRD18DP
KCNC3
MIR4802
YIF1B
CRIP1
MARCO
C1QTNF2
SRGAP2B
ANXA2
SAMHD1
PRKCD
ARHGEF10L
MILR1
PTPN6
UNC93B1
S100A4
CTSV
CTSZ
ADAP2
FIBP
BID
RAB31
UQCRC1
CTSB
SLC37A2
PLD2
MRPL18
DPYSL2
FGD2
PSMD4
SCPEP1
COPZ2
LRP1
PECAM1
TACC2
NACC2
CIMAP1C
POLR2E
KCNMB1
TYMP
MTMR11
ASL
AURKAIP1
RCAN1
ITGA7
ST14
ADCK5
SORT1
PKM
NOD2
MYO7A
DNM1P35
SMCO4
CD63
PLCB3
UBXN10
PGP
LMAN2
RNH1
CAMKV
PFDN1P1
MRAS
MIR378A
MS4A4A
SLC8A1
WARS1
ANXA2P2
RET
EHD4
KCNE1
PLEC
TPI1
RAB11FIP5
ETFB
CUEDC1
GLIPR2
RTL8A
P2RY6
RPL5P5
LINC01127
LILRB4
ACOT7
TRPM4
P4HB
''')

KTC_multi_enrichment_plot(gene_list)

df_clin_merged_aml["OS_STATUS"].astype(str).value_counts()

for gene in gene_list:
    df_km = km_plot_for_gene(
        gene,
        expr_df=df_expr_rna,         # TPM matrix with symbols as index
        clin_merged_df=df_clin_merged_aml,
        time_col="OS_MONTHS",        # <- use months
        status_col="OS_STATUS"       # <- parse 0/1 from this
    )


#%% Scan for gene-gene correlations
out_dir = "/Users/kachrist/Downloads/aml_target_gdc/correlation_results"

target_gene = 'SMIM24'

top_hits, bottom_hits, all_hits = scan_gene_correlations(
    gene=target_gene,
    df_expr=df_exp_matrix,
    out_dir=out_dir,
    project="TARGET_AML",
    write_csv=True
)

for gene, (r, p) in top_hits[:10]:
    print(f"Plotting {gene} (r={r:.2f})")
    Grapher_TARGET(target_gene, gene, df_exp_matrix)

#%% Scan for KM plot significances

df_km_scan = scan_km_for_genes(
    expr_df=df_exp_matrix,
    clin_df=df_clin_merged_aml,
    time_col="OS_MONTHS",
    status_col="OS_STATUS",
    min_group_size=10,
    min_events=5,
    out_dir="/Users/kachrist/Downloads/aml_target_gdc/km_results",
    project="TARGET_AML",
    out_prefix="OS_median_split"
)

df_km_scan.head()

print(df_clin_merged_aml)
