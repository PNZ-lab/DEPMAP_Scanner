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

#%% 
# gene_sets = {
#     'None' : [],
#     'YTHDC2' : ['YTHDC2'],
#     'HNRNPC' : ['HNRNPC'],
#     'm6a readers' : ['HNRNPC', 'YTHDF1', 'YTHDF2', 'YTHDF3', 'YTHDC1', 'YTHDC2', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'HNRNPA2B1'],
#     'splicing factors' : ['SFRS7', 'CUGBP1', 'DAZAP1', 'CUGBP2', 'FMR1', 'A2BP1', 'RBFOX2', 'HNRNPA0', 'HNRNPA1', 'HNRNPA2B1', 'HNRNPC', 'HNRNPC', 'HNRNPC', 'HNRNPD', 'HNRNPD', 'HNRPDL', 'PCBP1', 'PCBP2', 'HNRNPF', 'HNRNPH1', 'HNRNPH2', 'HNRNPH3', 'PTBP1', 'HNRNPK', 'HNRNPK', 'HNRNPL', 'HNRPLL', 'HNRNPM', 'FUS', 'HNRNPU', 'TRA2A', 'TRA2B', 'ELAVL2', 'ELAVL4', 'ELAVL1', 'KHSRP', 'MBNL1', 'NOVA1', 'NOVA2', 'PTBP2', 'SFPQ', 'RBM25', 'RBM4', 'KHDRBS1', 'SF3B1', 'SFRS2', 'SF1', 'SFRS1', 'KHDRBS2', 'KHDRBS3', 'SFRS3', 'SFRS9', 'SFRS13A', 'SFRS5', 'SFRS11', 'SFRS6', 'SFRS4', 'TARDBP', 'TIA1', 'TIAL1', 'YBX1', 'ZRANB2', 'ELAVL3', 'RBM5', 'SYNCRIP', 'HNRNPA3', 'QKI', 'RBMX', 'SRRM1', 'ESRP1', 'ESRP2'], # From SpliceAid
#     'EpiFactors' : ['A1CF', 'ACINU', 'ACTB', 'ACTL6A', 'ACTL6B', 'ACTR3B', 'ACTR5', 'ACTR6', 'ACTR8', 'ADNP', 'AEBP2', 'AICDA', 'AIRE', 'ALKBH1', 'ALKBH1', 'ALKBH4', 'ALKBH5', 'ANKRD32', 'ANP32A', 'ANP32B', 'ANP32E', 'APBB1', 'APEX1', 'APOBEC1', 'APOBEC2', 'APOBEC3A', 'APOBEC3B', 'APOBEC3C', 'APOBEC3D', 'APOBEC3F', 'APOBEC3G', 'APOBEC3H', 'ARID1A', 'ARID1B', 'ARID2', 'ARID4A', 'ARID4B', 'ARNTL', 'ARRB1', 'ASF1A', 'ASF1B', 'ASH1L', 'ASH2L', 'ASXL1', 'ASXL2', 'ASXL3', 'ATAD2', 'ATAD2B', 'ATF2', 'ATF7IP', 'ATM', 'ATN1', 'ATR', 'ATRX', 'ATXN7', 'ATXN7L3', 'AURKA', 'AURKB', 'AURKC', 'BABAM1', 'BAHD1', 'BANP', 'BAP1', 'BARD1', 'BAZ1A', 'BAZ1B', 'BAZ2A', 'BAZ2B', 'BCOR', 'BCORL1', 'BMI1', 'BPTF', 'BRCA1', 'BRCA2', 'BRCC3', 'BRD1', 'BRD2', 'BRD3', 'BRD4', 'BRD7', 'BRD8', 'BRD9', 'BRDT', 'BRE', 'BRMS1', 'BRMS1L', 'BRPF1', 'BRPF3', 'BRWD1', 'BRWD3', 'BUB1', 'C11orf30', 'C14orf169', 'C17orf49', 'CARM1', 'CBLL1', 'CBX1', 'CBX2', 'CBX3', 'CBX4', 'CBX5', 'CBX6', 'CBX7', 'CBX8', 'CCDC101', 'CDC6', 'CDC73', 'CDK1', 'CDK17', 'CDK2', 'CDK3', 'CDK5', 'CDK7', 'CDK9', 'CDY1', 'CDY1B', 'CDY2A', 'CDY2B', 'CDYL', 'CDYL2', 'CECR2', 'CELF1', 'CELF2', 'CELF3', 'CELF4', 'CELF5', 'CELF6', 'CENPC', 'CHAF1A', 'CHAF1B', 'CHD1', 'CHD1L', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9', 'CHEK1', 'CHRAC1', 'CHTOP', 'CHUK', 'CIR1', 'CIT', 'CLNS1A', 'CLOCK', 'CRB2', 'CREBBP', 'CSNK2A1', 'CSRP2BP', 'CTBP1', 'CTBP2', 'CTCF', 'CTCFL', 'CTR9', 'CUL1', 'CUL2', 'CUL3', 'CUL4A', 'CUL4B', 'CUL5', 'CXXC1', 'DAPK3', 'DAXX', 'DDB1', 'DDB2', 'DDX17', 'DDX21', 'DDX5', 'DDX50', 'DEK', 'DHX9', 'DMAP1', 'DNAJC1', 'DNAJC2', 'DND1', 'DNMT1', 'DNMT3A', 'DNMT3B', 'DNMT3L', 'DNTTIP2', 'DOT1L', 'DPF1', 'DPF2', 'DPF3', 'DPPA3', 'DPY30', 'DR1', 'DTX3L', 'DZIP3', 'E2F6', 'EED', 'EEF1AKMT3', 'EEF1AKMT4', 'EEF1AKNMT', 'EHMT1', 'EHMT2', 'EID1', 'EID2', 'EID2B', 'EIF4A3', 'ELP2', 'ELP3', 'ELP4', 'ELP5', 'ELP6', 'ENY2', 'EP300', 'EP400', 'EPC1', 'EPC2', 'ERBB4', 'ERCC6', 'EXOSC1', 'EXOSC2', 'EXOSC3', 'EXOSC4', 'EXOSC5', 'EXOSC6', 'EXOSC7', 'EXOSC8', 'EXOSC9', 'EYA1', 'EYA2', 'EYA3', 'EYA4', 'EZH1', 'EZH2', 'FAM175A', 'FAM175B', 'FBL', 'FBRS', 'FBRSL1', 'FOXA1', 'FOXO1', 'FOXP1', 'FOXP2', 'FOXP3', 'FOXP4', 'FTO', 'GADD45A', 'GADD45B', 'GADD45G', 'GATAD1', 'GATAD2A', 'GATAD2B', 'GFI1', 'GFI1B', 'GLYR1', 'GSE1', 'GSG2', 'GTF2I', 'GTF3C4', 'HAT1', 'HCFC1', 'HCFC2', 'HDAC1', 'HDAC10', 'HDAC11', 'HDAC2', 'HDAC3', 'HDAC4', 'HDAC5', 'HDAC6', 'HDAC7', 'HDAC8', 'HDAC9', 'HDGF', 'HDGFL2', 'HELLS', 'HIF1AN', 'HINFP', 'HIRA', 'HIRIP3', 'HJURP', 'HLCS', 'HLTF', 'HMG20A', 'HMG20B', 'HMGB1', 'HMGN1', 'HMGN2', 'HMGN3', 'HMGN4', 'HMGN5', 'HNRNPU', 'HNRPL', 'HNRPM', 'HP1BP3', 'HR', 'HSFX3', 'HSPA1A', 'HSPA1A', 'HSPA1B', 'HSPA1B', 'HUWE1', 'IKBKAP', 'IKZF1', 'IKZF3', 'ING1', 'ING2', 'ING3', 'ING4', 'ING5', 'INO80', 'INO80B', 'INO80C', 'INO80D', 'INO80E', 'JADE1', 'JADE2', 'JADE3', 'JAK2', 'JARID2', 'JDP2', 'JMJD1C', 'JMJD6', 'KANSL1', 'KANSL2', 'KANSL3', 'KAT2A', 'KAT2B', 'KAT5', 'KAT6A', 'KAT6B', 'KAT7', 'KAT8', 'KDM1A', 'KDM1B', 'KDM2A', 'KDM2B', 'KDM3A', 'KDM3B', 'KDM4A', 'KDM4B', 'KDM4C', 'KDM4D', 'KDM4E', 'KDM5A', 'KDM5B', 'KDM5C', 'KDM5D', 'KDM6A', 'KDM6B', 'KDM7A', 'KDM8', 'KEAP1', 'KHDRBS1', 'KLF18', 'KMT2A', 'KMT2B', 'KMT2C', 'KMT2D', 'KMT2E', 'L3MBTL1', 'L3MBTL2', 'L3MBTL3', 'L3MBTL4', 'LAS1L', 'LBR', 'LEO1', 'LRWD1', 'MAGOH', 'MAP3K7', 'MAPKAPK3', 'MASTL', 'MAX', 'MAZ', 'MBD1', 'MBD2', 'MBD3', 'MBD4', 'MBD5', 'MBD6', 'MBIP', 'MBNL1', 'MBNL3', 'MBTD1', 'MCRS1', 'MDC1', 'MEAF6', 'MECP2', 'MEN1', 'METTL11B', 'METTL14', 'METTL16', 'METTL21A', 'METTL3', 'METTL4', 'MGA', 'MGEA5', 'MINA', 'MIS18A', 'MIS18BP1', 'MLLT1', 'MLLT10', 'MLLT6', 'MORF4L1', 'MORF4L2', 'MOV10', 'MPHOSPH8', 'MPND', 'MRGBP', 'MSH6', 'MSL1', 'MSL2', 'MSL3', 'MST1', 'MTA1', 'MTA2', 'MTA3', 'MTF2', 'MUM1', 'MYBBP1A', 'MYO1C', 'MYSM1', 'NAA60', 'NAP1L1', 'NAP1L2', 'NAP1L4', 'NASP', 'NAT10', 'NAT10', 'NBN', 'NCL', 'NCOA1', 'NCOA2', 'NCOA3', 'NCOA6', 'NCOR1', 'NCOR2', 'NEK6', 'NEK9', 'NFRKB', 'NFYB', 'NFYC', 'NIPBL', 'NOC2L', 'NPAS2', 'NPM1', 'NPM2', 'NSD1', 'NSL1', 'NSRP1', 'NSUN2', 'NSUN6', 'NTMT1', 'NUP98', 'OGT', 'OIP5', 'PADI1', 'PADI2', 'PADI3', 'PADI4', 'PAF1', 'PAGR1', 'PAK2', 'PARG', 'PARP1', 'PARP2', 'PARP3', 'PAXIP1', 'PBK', 'PBRM1', 'PCGF1', 'PCGF2', 'PCGF3', 'PCGF5', 'PCGF6', 'PCNA', 'PDP1', 'PELP1', 'PHC1', 'PHC2', 'PHC3', 'PHF1', 'PHF10', 'PHF12', 'PHF13', 'PHF14', 'PHF19', 'PHF2', 'PHF20', 'PHF20L1', 'PHF21A', 'PHF8', 'PHIP', 'PIWIL4', 'PKM', 'PKN1', 'POGZ', 'POLE3', 'PPARGC1A', 'PPM1G', 'PPP2CA', 'PPP4C', 'PPP4R2', 'PQBP1', 'PRDM1', 'PRDM11', 'PRDM12', 'PRDM13', 'PRDM14', 'PRDM16', 'PRDM2', 'PRDM4', 'PRDM5', 'PRDM6', 'PRDM7', 'PRDM8', 'PRDM9', 'PRKAA1', 'PRKAA2', 'PRKAB1', 'PRKAB2', 'PRKAG1', 'PRKAG2', 'PRKAG3', 'PRKCA', 'PRKCB', 'PRKCD', 'PRKDC', 'PRMT1', 'PRMT2', 'PRMT5', 'PRMT6', 'PRMT7', 'PRMT8', 'PRMT9', 'PRPF31', 'PRR14', 'PSIP1', 'PTBP1', 'PTBP1', 'PUF60', 'RAD51', 'RAD54B', 'RAD54L', 'RAD54L2', 'RAG1', 'RAG2', 'RAI1', 'RARA', 'RB1', 'RBBP4', 'RBBP5', 'RBBP7', 'RBFOX1', 'RBM11', 'RBM15', 'RBM15B', 'RBM17', 'RBM24', 'RBM25', 'RBM4', 'RBM5', 'RBM7', 'RBM8A', 'RBMY1A1', 'RBX1', 'RCC1', 'RCOR1', 'RCOR3', 'REST', 'RFOX1', 'RING1', 'RLIM', 'RMI1', 'RNF168', 'RNF2', 'RNF20', 'RNF40', 'RNF8', 'RNPS1', 'RPS6KA3', 'RPS6KA4', 'RPS6KA5', 'RPUSD3', 'RRP8', 'RSF1', 'RSRC1', 'RUVBL1', 'RUVBL2', 'RYBP', 'SAFB', 'SAP130', 'SAP18', 'SAP25', 'SAP30', 'SAP30L', 'SATB1', 'SATB2', 'SCMH1', 'SCML2', 'SCML4', 'SENP1', 'SENP3', 'SET', 'SETD1A', 'SETD1B', 'SETD2', 'SETD3', 'SETD5', 'SETD6', 'SETD7', 'SETD8', 'SETDB1', 'SETDB2', 'SETMAR', 'SF3B1', 'SF3B3', 'SFMBT1', 'SFMBT2', 'SFPQ', 'SFSWAP', 'SHPRH', 'SIN3A', 'SIN3B', 'SIRT1', 'SIRT2', 'SIRT6', 'SIRT7', 'SKP1', 'SLU7', 'SMARCA1', 'SMARCA2', 'SMARCA4', 'SMARCA5', 'SMARCAD1', 'SMARCAL1', 'SMARCB1', 'SMARCC1', 'SMARCC2', 'SMARCD1', 'SMARCD2', 'SMARCD3', 'SMARCE1', 'SMEK1', 'SMEK2', 'SMYD1', 'SMYD2', 'SMYD3', 'SMYD4', 'SNAI2', 'SP1', 'SP100', 'SP140', 'SPEN', 'SPOP', 'SRCAP', 'SRRM4', 'SRSF1', 'SRSF10', 'SRSF12', 'SRSF3', 'SRSF6', 'SS18L1', 'SS18L2', 'SSRP1', 'STK4', 'SUDS3', 'SUPT16H', 'SUPT3H', 'SUPT6H', 'SUPT7L', 'SUV39H1', 'SUV39H2', 'SUV420H1', 'SUV420H2', 'SUZ12', 'SYNCRIP', 'TADA1', 'TADA2A', 'TADA2B', 'TADA3', 'TAF1', 'TAF10', 'TAF12', 'TAF1L', 'TAF2', 'TAF3', 'TAF4', 'TAF5', 'TAF5L', 'TAF6', 'TAF6L', 'TAF7', 'TAF8', 'TAF9', 'TAF9B', 'TBL1XR1', 'TDG', 'TDRD3', 'TDRD7', 'TDRKH', 'TET1', 'TET2', 'TET3', 'TEX10', 'TFDP1', 'TFPT', 'THRAP3', 'TLE1', 'TLE2', 'TLE4', 'TLK1', 'TLK2', 'TNP1', 'TNP2', 'TONSL', 'TOP2A', 'TOP2B', 'TP53', 'TP53BP1', 'TRA2B', 'TRIM16', 'TRIM24', 'TRIM27', 'TRIM28', 'TRIM33', 'TRRAP', 'TRUB2', 'TSSK6', 'TTK', 'TYW5', 'U2AF2', 'UBE2A', 'UBE2B', 'UBE2D1', 'UBE2D3', 'UBE2E1', 'UBE2H', 'UBE2N', 'UBE2T', 'UBN1', 'UBR2', 'UBR5', 'UBR7', 'UCHL5', 'UHRF1', 'UHRF2', 'UIMC1', 'USP11', 'USP12', 'USP15', 'USP16', 'USP17L2', 'USP21', 'USP22', 'USP3', 'USP36', 'USP44', 'USP46', 'USP49', 'USP7', 'UTY', 'VDR', 'VIRMA', 'VPS72', 'VRK1', 'WAC', 'WDR5', 'WDR77', 'WDR82', 'WHSC1', 'WHSC1L1', 'WSB2', 'WTAP', 'YAF2', 'YEATS2', 'YEATS4', 'YTHDC1', 'YWHAB', 'YWHAE', 'YWHAZ', 'YY1', 'ZBTB16', 'ZBTB33', 'ZBTB7A', 'ZBTB7C', 'ZC3H13', 'ZCWPW1', 'ZFP57', 'ZGPAT', 'ZHX1', 'ZMYM2', 'ZMYM3', 'ZMYND11', 'ZMYND8', 'ZNF217', 'ZNF516', 'ZNF532', 'ZNF541', 'ZNF592', 'ZNF687', 'ZNF711', 'ZNHIT1', 'ZRANB3', 'ZZZ3'],
#     'm6a_writers' : ['METTL3', 'METTL14', 'METTL16', 'KIAA1429','RBM15', 'WTAP'],
#     'm6a_erasers' : ['FTO', 'ALKBH5'],
#     'm6a_readers' : ['YTHS', 'EIF3', 'HNRNPC', 'HNRNPA2B1', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3'],
#     'm6a_re_wr_er' : ['METTL3', 'METTL14', 'METTL16', 'KIAA1429','RBM15', 'WTAP', 'FTO', 'ALKBH5', 'YTHS', 'EIF3', 'HNRNPC', 'HNRNPA2B1', 'YTHDF1', 'YTHDF2', 'YTHDC1', 'YTHDC2', 'TYSND1', 'SND1', 'PRRC2A', 'LRPPRC', 'FMR1','FMR1NB', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3'],
#     'PRC2' : ['EZH1', 'EZH2', 'EED', 'SUZ12', 'RBBP4', 'RBBP7', 'JARID2', 'PCGF1', 'PCGF2', 'RING1', 'BMI1'],
#     'Laura' : ['NAMPT', 'NAPRT', 'IDO', 'DHFR', 'NNMAT1', 'NNMAT2', 'NNMAT3', 'QPRT', 'MAT2A'],
#     'Kevin' :['PRPF8', 'SRRM1', 'SRRM2', 'ACIN1', 'RNPS1', 'CLK1', 'CLK2', 'CLK3', 'CLK4']
# 	}

# This function helps define gene sets to be highlighted.
# A set of genes is selected based on the string used as input.
# First, the string is tested in the custom set of gene sets in gene_sets dictionary above.
# If that fails, it will use the string to search for a public gene set on Msigdb
# If that fails, it will default to interpreting the input string as a set with a single gene name (the input string)
# from gseapy import Msigdb
# def GetGeneSet(name_gene_set):
#     try:
#         # First, try to find the gene set in the `gene_sets` dictionary
#         gene_set = gene_sets[name_gene_set]
#         print(f'\n -- NOTE -- Using custom set of genes from gene_sets dictionary: {name_gene_set}')
#     except KeyError:
#         try:
#             # If not found, try to fetch it from msigdb
#             gmt = Msigdb().get_gmt(category='h.all', dbver="2024.1.Hs")
#             gene_set = gmt[name_gene_set]
#             print(f'\n -- NOTE -- Using gene set from msigdb: {name_gene_set}')
#         except KeyError:
#             # If still not found, use string as a gene name for set with single gene
#             print(f'\n -- NOTE -- Gene set: {name_gene_set} not found in gene_sets or using Msigdb. Defaulting to interpreting {name_gene_set} as a set with a single gene')
#             gene_set = [name_gene_set.upper()]
#     return gene_set

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

HeatmapSpecificGenes(KTC_GetGeneSet('m6a_re_wr_er'), cancers=['AML', 'BLL', 'TLL'], fig_width=4, fig_height=6, annotate_values=True)
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

