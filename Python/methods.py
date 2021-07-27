import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import re
import pandas as pd 
import glob
import scipy
import scipy.io
import math
from collections import Counter
import scanpy as sc
import h5py
import gseapy as gp
import time
from matplotlib import rcParams
import harmonypy as hm
import scrublet as scr


        ####################
        #
        # NEW FUNCTIONS ANALYSIS 2
        #
        ####################


def adata_preproc(files_list, samples_names, batch_corr = False, max_genes = 3000, 
                    min_genes = 250, min_cells=10,
                    perc_mito = 0.15, plots = False, samples_file =  'samples.csv',
                    verbose = False, dir_for_figs = '', convert_ids = False,
                    scrublet = False, file_h5 = False, flex_filt = False, del_genes = False):
    samples = samples_names
    raw_adata = {}
    samples_info = pd.read_csv(samples_file, sep = '\t')
    if verbose:
        print("reading samples ...")
    if convert_ids:
        converted_once = False
        annot = sc.queries.biomart_annotations("hsapiens",["ensembl_gene_id", "start_position", "end_position", "chromosome_name", "hgnc_symbol"]).set_index("ensembl_gene_id")
    for sample in samples:
        if file_h5:
            raw_adata[sample] = sc.read_10x_h5(files_list[sample])
        else:
            raw_adata[sample] = sc.read_10x_mtx(files_list[sample], cache = True)
        raw_adata[sample].obs.index = ["cell_" + str(sample) + '_' + str(i) for i in range(len(raw_adata[sample].obs.index))]
        if convert_ids:
            print(raw_adata[sample])
            raw_adata[sample].var['old_index'] = raw_adata[sample].var.index
            if converted_once:
                raw_adata[sample].var.index = new_genes_index
            else:
                new_index = [annot.loc[x, 'hgnc_symbol'] if (x in annot.index.to_list()) else x for x in raw_adata[sample].var['old_index']]
                new_genes_index = [raw_adata[sample].var.index.to_list()[i] if (not isinstance(new_index[i], str)) else new_index[i] for i in range(len(new_index))]
                raw_adata[sample].var.index = new_genes_index
                converted_once = True
            raw_adata[sample].var_names_make_unique()
            print(raw_adata[sample].var.index[:20])
        if verbose:
            print(raw_adata[sample])
        if del_genes:
            print('after...')
            raw_adata[sample] = raw_adata[sample][:,~(raw_adata[sample].var.index.isin(del_genes))]
            print(raw_adata[sample])
        raw_adata[sample].var_names_make_unique()
    if scrublet != False:
        if verbose:
            print('pre-scrublet filtering ...')
        for sample in samples:
            print(sample)
            sc.pp.filter_cells(raw_adata[sample], min_genes=min_genes)
            if verbose:
                print('after filt genes')
                print(raw_adata[sample])
            mito_genes = raw_adata[sample].var_names.str.startswith('MT-')
            raw_adata[sample].obs['percent_mito'] = np.sum(
                raw_adata[sample][:, mito_genes].X, axis=1).A1 / np.sum(raw_adata[sample].X, axis=1).A1
            raw_adata[sample].obs['n_counts'] = raw_adata[sample].X.sum(axis=1).A1
            if plots:
                sc.pl.violin(raw_adata[sample], ['n_genes'], jitter=0.2)
                sc.pl.violin(raw_adata[sample], ['n_counts'], jitter=0.2)
                sc.pl.violin(raw_adata[sample], ['percent_mito'], jitter=0.2)
                sc.pl.scatter(raw_adata[sample], x='n_counts', y='percent_mito')
                sc.pl.scatter(raw_adata[sample], x='n_counts', y='n_genes')
                sc.pl.highest_expr_genes(raw_adata[sample], n_top=15)
            raw_adata[sample] = raw_adata[sample][raw_adata[sample].obs.n_genes <= max_genes]
            if verbose:
                print('after filt max')
                print(raw_adata[sample])
            if flex_filt:
                print(raw_adata[sample].obs.shape[0])
                print((1/math.sqrt(raw_adata[sample].obs.shape[0])) * perc_mito)
                if ((1/math.sqrt(raw_adata[sample].obs.shape[0])) * perc_mito) < 0.90:
                    new_perc_mito = raw_adata[sample].obs.percent_mito.quantile((1/math.sqrt(raw_adata[sample].obs.shape[0])) * perc_mito)
                else:
                    new_perc_mito = raw_adata[sample].obs.percent_mito.quantile(0.90)
                print("New perc_mito: " + str(new_perc_mito))
            else:
                new_perc_mito = perc_mito
            raw_adata[sample] = raw_adata[sample][raw_adata[sample].obs['percent_mito'] < new_perc_mito]
            if verbose:
                print('after filt mito')
                print(raw_adata[sample])
        if verbose:
            print('scrublet ...')
        for sample in samples:
            if verbose:
                print(sample)
            scrub = scr.Scrublet(raw_adata[sample].X, expected_doublet_rate=0.06)
            doublet_scores, predicted_doublets = scrub.scrub_doublets(min_counts=2, 
                                                                min_cells=3, 
                                                                min_gene_variability_pctl=85, 
                                                                n_prin_comps=30)
            scrub.call_doublets(threshold=scrublet)
            if plots:
                scrub.plot_histogram()
                #scrub.set_embedding('UMAP', scr.get_umap(scrub.manifold_obs_, 10, min_dist=0.3))
                #scrub.plot_embedding('UMAP', order_points=True)
            if verbose:
                print('before filtering ..')
                print(raw_adata[sample])
            raw_adata[sample].obs['doublet_score'] = scrub.doublet_scores_obs_
            raw_adata[sample] = raw_adata[sample][scrub.predicted_doublets_ == False]
            if verbose: 
                print('after filtering...')
                print(raw_adata[sample])
    if verbose:
        print("concatenation...")
    for sample in samples:
        if not raw_adata[sample].var.index.is_unique:
            idx = pd.Series(raw_adata[sample].var.index.to_list())
            idx.loc[raw_adata[sample].var.index.duplicated()] = raw_adata[sample].var[raw_adata[sample].var.index.duplicated()].index + '-1'
            raw_adata[sample].var.index = idx
    adata = list(raw_adata.values())[0].concatenate(list(raw_adata.values())[1:])
    samples_dict = { i : samples[i] for i in range(0, len(samples) ) }
    adata.obs['batch'] = adata.obs['batch'].astype(int).replace(samples_dict)
    adata.obs['batch'] = pd.Categorical(adata.obs.batch)
    adata.obs['sample_id'] = adata.obs['batch']
    for i, column in enumerate(samples_info.columns):
        if (i > 0):
            adata.obs[column] = [ samples_info[samples_info.Sample_id == sample_id][column].values[0] for sample_id in list(adata.obs.sample_id)]
            adata.obs[column] = pd.Categorical(adata.obs[column])
    if verbose:
        print("filtering...")
    if scrublet == False:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        if plots:
            sc.pl.highest_expr_genes(adata, n_top=15, save = dir_for_figs + "highest_expr1.png")
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['percent_mito'] = np.sum(
            adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
        adata.obs['n_counts'] = adata.X.sum(axis=1).A1
        if plots:
            sc.pl.violin(adata, ['n_genes'],
                jitter=0.2, save= dir_for_figs + "quality_metrics_aft_filt1_genes.png")
            sc.pl.violin(adata, ['n_counts'],
                jitter=0.2, save= dir_for_figs + "quality_metrics_aft_filt1_counts.png")
            sc.pl.violin(adata, ['percent_mito'],
                jitter=0.2, save= dir_for_figs + "quality_metrics_aft_filt1_pers_mito.png")
            sc.pl.scatter(adata, x='n_counts', y='percent_mito', save= dir_for_figs + "count_vs_mito1.png")
            sc.pl.scatter(adata, x='n_counts', y='n_genes', save= dir_for_figs + "count_vs_gene1.png")
    if verbose:
        print(adata)
        print(adata.obs["sample_id"].value_counts())
    if scrublet == False:
        if plots:
            plt.hist(adata.obs.n_counts, bins = 50)
            plt.show()
        adata = adata[adata.obs.n_genes <= max_genes,:]
        if plots:
            plt.hist(adata.obs.n_counts, bins = 500)
            plt.show()
            for sample in samples:
                print(sample)
                plt.hist(adata[adata.obs.sample_id == sample].obs.n_counts, bins = 500)
                plt.show()
        adata = adata[adata.obs['percent_mito'] < perc_mito, :]
    if verbose:
        print(adata)
        print(adata.obs["sample_id"].value_counts())
        print("mean_ncounts")
        print(adata.obs['n_counts'].mean())
        for sample in samples:
            print(sample)
            print(adata[adata.obs.sample_id == sample].obs['n_counts'].mean())
        print("mean ngenes")
        print(adata.obs['n_genes'].mean())
        for sample in samples:
            print(sample)
            print(adata[adata.obs.sample_id == sample].obs['n_genes'].mean())
    if plots:
        sc.pl.violin(adata, ['n_genes'],
            jitter=0.2, save= dir_for_figs + "quality_metrics_aft_filt2_genes.png")
        sc.pl.violin(adata, ['n_counts'],
            jitter=0.2, save= dir_for_figs + "quality_metrics_aft_filt2_counts.png")
        sc.pl.violin(adata, ['percent_mito'],
            jitter=0.2, save= dir_for_figs + "quality_metrics_aft_filt2_pers_mito.png")
        for sample in samples:
            print(sample)
            sc.pl.violin(adata[adata.obs.sample_id == sample], ['n_genes'], jitter=0.2)
            sc.pl.violin(adata[adata.obs.sample_id == sample], ['n_counts'], jitter=0.2)
            sc.pl.violin(adata[adata.obs.sample_id == sample], ['percent_mito'], jitter=0.2)
        sc.pl.scatter(adata, x='n_counts', y='percent_mito', save= dir_for_figs + "count_vs_mito2.png")
        sc.pl.scatter(adata, x='n_counts', y='n_genes', save= dir_for_figs + "count_vs_gene2.png")
        sc.pl.highest_expr_genes(adata, n_top=15, save = dir_for_figs + "highest_expr2.png")
    if verbose:
        print("Normalisation ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    if verbose:
        print("Highly var genes")
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.highly_variable_genes(adata, min_mean=0.005, max_mean=5, min_disp=-0.25)
    if plots:
        sc.pl.highly_variable_genes(adata, save=dir_for_figs + "var_genes.png")
    adata = adata[:, adata.var.highly_variable]
    if verbose:
        print(adata)
    if verbose:
        print("regression ...")
    sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])
    sc.pp.scale(adata, max_value=10)
    if plots:
        print("without batch correction:")
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pl.pca(adata, color=['batch', 'sample_id', 'phenotype'], save = dir_for_figs + "PCA_uncorr.png")
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        sc.tl.leiden(adata)
        sc.pl.umap(adata, color=['batch', 'sample_id', 'phenotype', 'leiden'], save = dir_for_figs + "UMAP_uncorr.png")
    if batch_corr:
        if verbose:
            print("Batch correction per sample...")
        ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, "sample_id")
        #adata.obsm['X_pca'] = ho.Z_corr.T
        adata.obsm['X_pca'] = ho.Z_cos.T
        adata.uns['PCA_corr'] = ho.Z_corr.T
        adata.uns['PCA_cos'] = ho.Z_cos.T
        if plots:
            print("with batch correction:")
            #sc.tl.pca(adata, svd_solver='arpack')
            print('cos')
            sc.pl.pca(adata, color=['batch', 'sample_id', 'phenotype'], save = dir_for_figs + "PCA_corr.png")
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
            sc.tl.umap(adata)
            sc.tl.leiden(adata)
            sc.pl.umap(adata, color=['batch', 'sample_id', 'phenotype', 'leiden'], save = dir_for_figs + "UMAP_corr.png")
            print('corr')
            adata.obsm['X_pca'] = ho.Z_corr.T
            sc.pl.pca(adata, color=['batch', 'sample_id', 'phenotype'], save = dir_for_figs + "PCA_corr.png")
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
            sc.tl.umap(adata)
            sc.tl.leiden(adata)
            sc.pl.umap(adata, color=['batch', 'sample_id', 'phenotype', 'leiden'], save = dir_for_figs + "UMAP_corr.png")
    return adata

def plots_markers(adata, markers_dict = "default", save_plots = False, dim_type = 'UMAP'):
    if markers_dict == "default":
        markers_dict = {
            'CM': ['leiden', 'sample_id', 'TTN', 'MYH7', 'MYH6', 'TNNT2'],
            'VEC': ['leiden', 'sample_id', 'VWF', 'ANO2', 'PECAM1'],
            'EEC': ['leiden', 'MYRIP', 'LEPR', 'EMCN', 'HMCN1', 'PECAM1'],
            'FB': ['leiden', 'sample_id', 'DCN', 'LUM', 'FBLN1', 'COL1A2'],
            'MESO' : ['WT1', 'UPK3B', 'HAS1', 'MSLN'],
            'MC': ['leiden', 'sample_id', 'CD163', 'CCL4', 'MRC1', 'SLC9A9'],
            'SMC': ['leiden', 'ACTA2', 'MYH11'],
            'PER': ['leiden', "RGS5", "PDGFRB", "ABCC9"],
            'AD': ['leiden', 'PLIN4', 'PLIN1', 'LIPE', 'ADIPOQ', 'CIDEA', 'LPL'],
            'N' : ['SYT1', 'SNAP25', 'GRIN1'],
            'SC' : ['PLP1', 'MPZ', 'PMP22'],
            'L' : ['CD79A', 'CD79B', 'CD3E', 'CD247'],
            'Others': ['leiden', 'RHOA', 'JCAD', 'PHACTR1', 'KLF2']
        }
    for markers in markers_dict:
        print(markers)
        if save_plots:
            save_file = save_plots + markers + '.png'
        else:
            save_file = False
        if dim_type == 'UMAP':
            sc.pl.umap(adata, color=markers_dict[markers], save= save_file)
        else:
            sc.pl.tsne(adata, color=markers_dict[markers], save= save_file)

def plot_heatmap_markers(adata, dir_for_figs, groupby = 'leiden', use_raw = False):
    markers_all = ["TTN", "MYH6", "MYH7", "TNNT2", 
               "VWF", "ANO2", "PECAM1", 
               "DCN", "C7", "LUM", "FBLN1", "COL1A2",
               "CD163", "SLC9A9", "MRC1", 
               "MYH11", "ACTA2",
               "RGS5", "PDGFRB", "ABCC9",
               "PLIN4", "PLIN1",
               'SYT1', 'SNAP25', 'GRIN1',
               'PLP1', 'MPZ', 'PMP22',
               'CD79A', 'CD79B', 'CD3E', 'CD247',
               'WT1', 'UPK3B', 'HAS1', 'MSLN',
               'MYRIP', 'LEPR', 'NPR3', 'NFATC1'
              ]
    markers_groups = [(0,3), (4,6), (7,11), (12,14), (15,16), (17,19), (20, 21), (22, 24), 
                  (25, 27), (28, 31), (32, 35), (36, 39)]
    markers_groups_names = ["CM", "VEC" , "FB", "MP", "SMC", "PER", "AD", 'N', 'SC', 'L', 'MESO', 'EEC']
    sc.tl.dendrogram(adata, groupby = groupby)
    sc.pl.dotplot(adata, markers_all, groupby = groupby, use_raw = use_raw, dendrogram= True,
                    var_group_positions = markers_groups,
                    var_group_labels = markers_groups_names, save = dir_for_figs + "dot_unscaled.png")
    sc.pl.dotplot(adata, markers_all, groupby = groupby, use_raw = use_raw, dendrogram= True,
                    standard_scale = "var", var_group_positions = markers_groups, 
                    var_group_labels = markers_groups_names, save = dir_for_figs + "dot_scaled.png")
    sc.pl.matrixplot(adata, markers_all, groupby = groupby, use_raw = use_raw, dendrogram= True,
                    standard_scale = "var", var_group_positions = markers_groups, 
                    var_group_labels = markers_groups_names,
                    cmap = "Blues", save = dir_for_figs + "heatmap.png")


def run_umap_leiden(adata, plot = False):
    n_pcs = 50
    if adata.shape[0] < 50:
        n_pcs = adata.shape[0] - 1
    sc.tl.pca(adata, svd_solver='arpack', n_comps = n_pcs)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs= n_pcs)
    sc.tl.umap(adata)
    adata.obs["old_leiden"] = adata.obs["leiden"].copy()
    sc.tl.leiden(adata)
    if plot:
        sc.pl.umap(adata, color=['batch', 'leiden'])
    return adata

######
##
## Preprocessing functions:
##
######

def partition_adata_obs(adata, categories_name):
    categories = list(set(adata.obs[categories_name].to_list()))
    subadata = {}
    for category in categories:
        subadata[category] = adata[adata.obs[categories_name] == category]
        sc.pl.umap(subadata[category], color=['leiden', 'cell_types'])
    return subadata

def rerun_umap_leiden(subadata_1, scale = False):
    if scale:
        sc.pp.scale(subadata_1, max_value=10)
    n_pcs = 50
    if subadata_1.shape[0] < 50:
        n_pcs = subadata_1.shape[0] - 1
    sc.tl.pca(subadata_1, svd_solver='arpack', n_comps = n_pcs)
    sc.pp.neighbors(subadata_1, n_neighbors=10, n_pcs= n_pcs)
    sc.tl.umap(subadata_1)
    subadata_1.obs["old_leiden"] = subadata_1.obs["leiden"].copy()
    sc.tl.leiden(subadata_1)
    sc.pl.umap(subadata_1, color=['leiden', 'cell_types'])
    return subadata_1

def rerun_all_umap_leiden(subadata, scale = False):
    for category in subadata:
        print('\n\n' + category + '\n')
        subadata[category] = rerun_umap_leiden(subadata[category], scale)
    return subadata

def filter_markers(markers, adata):
    new_markers = {}
    for cell_type in markers:
        new_markers[cell_type] = []
        for marker in markers[cell_type]:
            marker = marker.upper()
            if marker in set(adata.var.index.to_list()):
                new_markers[cell_type].append(marker)
    return new_markers

def umap_fig(adata, markers, save_file):
    sc.pl.umap(adata, color=markers, save =save_file)
    
def multiple_umap_leiden(adata, markers, file_beg, add_markers = [], ext = '.pdf'):
    for subtype in markers:
        umap_fig(adata, ["leiden"] + add_markers + markers[subtype], file_beg + subtype + ext)

def display_markers_subtypes(subadata, markers, file_name, ext = '.pdf'):
    for category in subadata:
        multiple_umap_leiden(subadata[category], markers[category], category + "/" + file_name, ext = '.pdf')

def rank_plot_all(subadata, n_genes, folder, file_name, ext = '.pdf'):
    for category in subadata:
        dirName = folder + "/" + category
        try:
            os.makedirs(dirName)    
            print("Directory " , dirName ,  " Created ")
        except FileExistsError:
            print("Directory " , dirName ,  " already exists")  
        sc.settings.figdir = dirName + '/'
        print(category)
        sc.tl.rank_genes_groups(subadata[category], 'leiden', method='t-test', n_genes = 1000)
        sc.pl.rank_genes_groups_dotplot(subadata[category], groupby = 'leiden', n_genes = n_genes, save = file_name + ext)


def enrichment_all(subadata, gene_sets, folder, cutoff = 0.05, verbose = False):
    subadata_enr = {}
    for category in subadata:
        dirName = folder + "/" + category
        try:
            os.makedirs(dirName)    
            print("Directory " , dirName ,  " Created ")
        except FileExistsError:
            print("Directory " , dirName ,  " already exists")  
        sc.settings.figdir = dirName + '/'
        print(category + "7\n\n")
        sc.tl.rank_genes_groups(subadata[category], 'leiden', method='t-test', n_genes = 1000)
        enr = {}
        groups = set(subadata[category].obs['leiden'].to_list())
        i = 0
        while i < len(groups):
            #enr[group] = {}
            group = list(groups)[i]
            num_sign = (subadata[category].uns['rank_genes_groups']['pvals'][group] < cutoff).sum()
            print("i: {}, group: {}, num_genes: {} \n".format(i, group, num_sign))
            list_genes = pd.DataFrame(subadata[category].uns['rank_genes_groups']['names'])[group].head(num_sign)
            list_genes = [x for x in list_genes if not x.startswith("MT-")]
            #for gene_set in gene_sets:
            time.sleep(0.5)
            try:
                enr[group] = gp.enrichr(gene_list = list_genes,
                                                        description = group, # + gene_set, 
                                                        gene_sets = gene_sets,
                                                        outdir = dirName + '/' + group + '/',  
                                                        cutoff = 0.5,
                                                        verbose = verbose, format='png')
                i += 1
            except:
                time.sleep(3)
                print("connection error \n")
        subadata_enr[category] = enr
    return subadata_enr

def enrichment_once(adata, group, gene_sets, folder, cutoff = 0.05, verbose = False, n_genes = 1000, method = 'wilcoxon', to_remove = False, n_max = False):
    dirName = folder + "/"
    try:
        os.makedirs(dirName)    
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")  
    sc.settings.figdir = dirName + '/'
    sc.tl.rank_genes_groups(adata, group, method= method, n_genes = n_genes)
    enr = {}
    groups = set(adata.obs[group].to_list())
    i = 0
    while i < len(groups):
        #enr[group] = {}
        group = list(groups)[i]
        num_sign = (adata.uns['rank_genes_groups']['pvals'][group] < cutoff).sum()
        #print("i: {}, group: {}, num_genes: {} \n".format(i, group, num_sign))
        list_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names'])[group].head(num_sign)
        list_genes = [x for x in list_genes if not x.startswith("MT-")]
        list_genes = [x for x in list_genes if not x.startswith("RPL")]
        list_genes = [x for x in list_genes if not x.startswith("RPS")]
        if not (to_remove == False):
            list_genes = [x for x in list_genes if not x in to_remove]
        if n_max :
            list_genes = list_genes[:n_max]
        #for gene_set in gene_sets:
        print("i: {}, group: {}, num_genes: {} \n".format(i, group, len(list_genes)))
        time.sleep(0.5)
        try:
            enr[group] = gp.enrichr(gene_list = list_genes,
                                                    description = group, # + gene_set, 
                                                    gene_sets = gene_sets,
                                                    outdir = dirName + '/' + group + '/',  
                                                    cutoff = 0.5,
                                                    verbose = verbose, format='png')
            i += 1
        except:
            time.sleep(3)
            print("connection error \n")
    return enr