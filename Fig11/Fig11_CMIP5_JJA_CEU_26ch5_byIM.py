#!/usr/bin/env python

# must be in ternary environment:
# conda activate ternary_plot

from pathlib import Path
import csv
import math
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import ternary

def shannon_entropy(p):
    """Computes the Shannon Entropy at a distribution in the simplex."""
    s = 0.
    for i in range(len(p)):
        try:
            s += p[i] * math.log(p[i])
        except ValueError:
            continue
    return -1.*s

def main():

    dir = "/**/CMIP_subselection/Data/"
    filename =dir+"alpha-beta-scan_SM_JJA_CMIP5_both_sum_ch5_sqrt.csv"

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        data = []
        for d in reader:
            d['alpha'] = np.round(float(d['alpha']), 3)
            d['beta'] = np.round(float(d['beta']), 3)
            d['models_str'] = ', '.join(sorted([d['member0'],d['member1'],d['member2'],d['member3'],d['member4']])) # change here
            data.append(d)

    alphas = list(sorted(list(set([d['alpha'] for d in data]))))
    betas = list(sorted(list(set([d['beta'] for d in data]))))
    member_combo_data = [[np.nan for _ in betas] for _ in alphas]
    models_labels = [['' for _ in betas] for _ in alphas]
    models_costvals = [[0 for _ in betas] for _ in alphas]

    models_to_number = {}
    for d in data:
        ia, ib = alphas.index(d['alpha']),  betas.index(d['beta'])
        assert ia+ib <= 100
        models_str = d['models_str']
        models_to_number.setdefault(models_str, len(models_to_number))
        member_combo_data[ia][ib] = models_to_number[models_str]
        models_labels[ia][ib] = models_str
        models_costvals[ia][ib] = float(d['min_val'])

    ds_test = xr.Dataset(dict(idx = (['alpha','beta'],member_combo_data), models=(['alpha','beta'],models_labels),costvals=(['alpha','beta'],models_costvals)), coords=dict(alpha=alphas,beta=betas))

    for k,v in models_to_number.items():
        print(v, k)

    data = {}
    for a in ds_test.alpha.data:
        for b in ds_test.beta.data:
            c = round(1-a-b,4)
            if c < 0:
                continue
            ia, ib = alphas.index(a),  betas.index(b)
            data[(ia, 100-ia-ib, ib)] = member_combo_data[ia][ib] #change here

    from matplotlib.colors import ListedColormap,LinearSegmentedColormap
    from matplotlib import colors

# JJA CMIP5 SM ch5 sqrt
    cmap = ListedColormap([
    "firebrick", #CSIRO-Mk3-6-0-r10i1p1, GISS-E2-H-r1i1p3, IPSL-CM5B-LR-r1i1p1, MIROC-ESM-r1i1p1, bcc-csm1-1-m-r1i1p1
    "crimson", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2G-r1i1p1, HadGEM2-ES-r4i1p1, IPSL-CM5B-LR-r1i1p1, MIROC-ESM-r1i1p1
    "lightcoral", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2G-r1i1p1, GISS-E2-H-r1i1p3, IPSL-CM5B-LR-r1i1p1, MIROC-ESM-r1i1p1
    "orangered", #CSIRO-Mk3-6-0-r10i1p1, GISS-E2-H-r1i1p3, IPSL-CM5B-LR-r1i1p1, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "orange", #CNRM-CM5-r2i1p1, CSIRO-Mk3-6-0-r10i1p1, HadGEM2-ES-r4i1p1, IPSL-CM5B-LR-r1i1p1, MIROC-ESM-r1i1p1
    "gold", #CSIRO-Mk3-6-0-r10i1p1, GISS-E2-H-r1i1p3, MIROC5-r3i1p1, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "yellow", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2G-r1i1p1, GISS-E2-H-r1i1p3, MIROC-ESM-r1i1p1, bcc-csm1-1-m-r1i1p1
    "sienna", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2M-r1i1p1, GISS-E2-H-r1i1p3, MIROC-ESM-r1i1p1, bcc-csm1-1-m-r1i1p1
    "peru", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2M-r1i1p1, GISS-E2-H-r1i1p3, MIROC-ESM-r1i1p1, NorESM1-ME-r1i1p1
    "olivedrab", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2M-r1i1p1, GISS-E2-H-r1i1p3, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "yellowgreen", #GFDL-CM3-r1i1p1, GISS-E2-H-r1i1p3, MIROC5-r3i1p1, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "darkgreen", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2G-r1i1p1, GISS-E2-H-r1i1p3, MIROC-ESM-r1i1p1
    "mediumseagreen", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, GISS-E2-H-r1i1p3, MIROC-ESM-r1i1p1
    "darkturquoise", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2G-r1i1p1, GISS-E2-H-r1i1p3, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "aquamarine", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, MIROC-ESM-r1i1p1, NorESM1-ME-r1i1p1
    "navy", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, GISS-E2-H-r1i1p3, NorESM1-ME-r1i1p1
    "mediumblue", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2M-r1i1p1, GISS-E2-H-r1i1p3, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "dodgerblue", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "deepskyblue", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, MIROC-ESM-r1i1p1, bcc-csm1-1-m-r1i1p1
    "lightskyblue", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "lightcyan", #GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, MIROC5-r3i1p1, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "mediumslateblue", #GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "indigo", #CESM1-CAM5-r1i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, MIROC5-r3i1p1, NorESM1-ME-r1i1p1
    "darkorchid", #ACCESS1-0-r1i1p1, CESM1-CAM5-r1i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, NorESM1-ME-r1i1p1
    "violet", #CESM1-CAM5-r1i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, NorESM1-ME-r1i1p1
    "mediumvioletred", #GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1, NorESM1-ME-r1i1p1
    "deeppink", #CNRM-CM5-r2i1p1, CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2G-r1i1p1, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "pink", #CNRM-CM5-r2i1p1, CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "lavenderblush", #CNRM-CM5-r2i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2G-r1i1p1, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "ivory", #GFDL-CM3-r1i1p1, GFDL-ESM2G-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "silver", #GFDL-CM3-r1i1p1, GFDL-ESM2G-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, NorESM1-ME-r1i1p1
    "tab:gray", # GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, NorESM1-M-r1i1p1, NorESM1-ME-r1i1p1
    "k"]) #CESM1-CAM5-r1i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, NorESM1-M-r1i1p1, NorESM1-ME-r1i1p1

    cmap_r = cmap.reversed()
    cb_kwargs = {"ticks" : np.arange(0,v+1)}

##################################################
    fig = plt.figure(figsize=(15,6)) #8,6, 14,10
    ax = fig.add_subplot(111)

    tax = ternary.TernaryAxesSubplot(ax=ax, scale=100)
    tax.boundary(linewidth=1)
    tax.gridlines(color="w", multiple=10, linewidth=0.5)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()

    tax.ticks(axis='b', linewidth=1, multiple=10, tick_formats="%i%%", offset=.018, clockwise=True,fontsize=12)
    tax.ticks(axis='r', linewidth=1, multiple=10, tick_formats="%i%%", offset=.026, clockwise=True,fontsize=12)
    tax.ticks(axis='l', linewidth=1, multiple=10, tick_formats="%i%%", offset=.028, clockwise=True,fontsize=12)

    tax.left_axis_label(r"Performance ([1-$\alpha$-$\beta$] $\times$ 100%)", offset=.15,fontsize=14)
    tax.right_axis_label(r"Independence ($\alpha$ $\times$ 100%)", offset=.15,fontsize=14)
    tax.bottom_axis_label(r"Spread  ($\beta$ $\times$ 100%)",offset=0.05,fontsize=14)
##################################################
    ternary.heatmap(data, scale=1, ax=ax, style="hexagonal", cmap=cmap_r, colorbar=False)
    models_list = [k for k,v in sorted(models_to_number.items(), key=lambda it: it[1])]
    vmin = min(data.values())
    vmax = max(data.values())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cb = plt.colorbar(sm, ax=ax)
    cb.set_ticks([(len(models_list)-1)*(i-0.5)/len(models_list) for i in range(1,len(models_list)+1)])
    models_list.reverse()
    cb.ax.set_yticklabels(models_list)
##################################################
    tax._redraw_labels()
    fig.tight_layout()
    fig.savefig(str("Fig11_CMIP5_JJA_CEU_26ch5_byIM.png"),bbox_inches='tight',dpi=300)

# JJA CMIP5 SM ch5 sqrt (rec)
    cmap = ListedColormap([
    "w", #CSIRO-Mk3-6-0-r10i1p1, GISS-E2-H-r1i1p3, IPSL-CM5B-LR-r1i1p1, MIROC-ESM-r1i1p1, bcc-csm1-1-m-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2G-r1i1p1, HadGEM2-ES-r4i1p1, IPSL-CM5B-LR-r1i1p1, MIROC-ESM-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2G-r1i1p1, GISS-E2-H-r1i1p3, IPSL-CM5B-LR-r1i1p1, MIROC-ESM-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GISS-E2-H-r1i1p3, IPSL-CM5B-LR-r1i1p1, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "w", #CNRM-CM5-r2i1p1, CSIRO-Mk3-6-0-r10i1p1, HadGEM2-ES-r4i1p1, IPSL-CM5B-LR-r1i1p1, MIROC-ESM-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GISS-E2-H-r1i1p3, MIROC5-r3i1p1, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2G-r1i1p1, GISS-E2-H-r1i1p3, MIROC-ESM-r1i1p1, bcc-csm1-1-m-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2M-r1i1p1, GISS-E2-H-r1i1p3, MIROC-ESM-r1i1p1, bcc-csm1-1-m-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2M-r1i1p1, GISS-E2-H-r1i1p3, MIROC-ESM-r1i1p1, NorESM1-ME-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2M-r1i1p1, GISS-E2-H-r1i1p3, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "w", #GFDL-CM3-r1i1p1, GISS-E2-H-r1i1p3, MIROC5-r3i1p1, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2G-r1i1p1, GISS-E2-H-r1i1p3, MIROC-ESM-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, GISS-E2-H-r1i1p3, MIROC-ESM-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2G-r1i1p1, GISS-E2-H-r1i1p3, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, MIROC-ESM-r1i1p1, NorESM1-ME-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, GISS-E2-H-r1i1p3, NorESM1-ME-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2M-r1i1p1, GISS-E2-H-r1i1p3, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "w", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, MIROC-ESM-r1i1p1, bcc-csm1-1-m-r1i1p1
    "lightskyblue", #CSIRO-Mk3-6-0-r10i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "lightcyan", #GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, MIROC5-r3i1p1, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "mediumslateblue", #GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, NorESM1-ME-r1i1p1, bcc-csm1-1-m-r1i1p1
    "indigo", #CESM1-CAM5-r1i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, MIROC5-r3i1p1, NorESM1-ME-r1i1p1
    "darkorchid", #ACCESS1-0-r1i1p1, CESM1-CAM5-r1i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, NorESM1-ME-r1i1p1
    "violet", #CESM1-CAM5-r1i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, NorESM1-ME-r1i1p1
    "w", #GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1, NorESM1-ME-r1i1p1
    "w", #CNRM-CM5-r2i1p1, CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2G-r1i1p1, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "w", #CNRM-CM5-r2i1p1, CSIRO-Mk3-6-0-r10i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "w", #CNRM-CM5-r2i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2G-r1i1p1, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "w", #GFDL-CM3-r1i1p1, GFDL-ESM2G-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, MIROC-ESM-r1i1p1
    "w", #GFDL-CM3-r1i1p1, GFDL-ESM2G-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, NorESM1-ME-r1i1p1
    "w", # GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, HadGEM2-ES-r4i1p1, NorESM1-M-r1i1p1, NorESM1-ME-r1i1p1
    "w"]) #CESM1-CAM5-r1i1p1, GFDL-CM3-r1i1p1, GFDL-ESM2M-r1i1p1, NorESM1-M-r1i1p1, NorESM1-ME-r1i1p1

    cmap_r = cmap.reversed()
    cb_kwargs = {"ticks" : np.arange(0,v+1)}

##################################################
    fig = plt.figure(figsize=(15,6)) #8,6, 14,10
    ax = fig.add_subplot(111)

    tax = ternary.TernaryAxesSubplot(ax=ax, scale=100)
    tax.boundary(linewidth=1)
    tax.gridlines(color="w", multiple=10, linewidth=0.5)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()

    tax.ticks(axis='b', linewidth=1, multiple=10, tick_formats="%i%%", offset=.018, clockwise=True,fontsize=12)
    tax.ticks(axis='r', linewidth=1, multiple=10, tick_formats="%i%%", offset=.026, clockwise=True,fontsize=12)
    tax.ticks(axis='l', linewidth=1, multiple=10, tick_formats="%i%%", offset=.028, clockwise=True,fontsize=12)

    tax.left_axis_label(r"Performance ([1-$\alpha$-$\beta$] $\times$ 100%)", offset=.15,fontsize=14)
    tax.right_axis_label(r"Independence ($\alpha$ $\times$ 100%)", offset=.15,fontsize=14)
    tax.bottom_axis_label(r"Spread  ($\beta$ $\times$ 100%)",offset=0.05,fontsize=14)
##################################################
    ternary.heatmap(data, scale=1, ax=ax, style="hexagonal", cmap=cmap_r, colorbar=False)
    models_list = [k for k,v in sorted(models_to_number.items(), key=lambda it: it[1])]
    vmin = min(data.values())
    vmax = max(data.values())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cb = plt.colorbar(sm, ax=ax)
    cb.set_ticks([(len(models_list)-1)*(i-0.5)/len(models_list) for i in range(1,len(models_list)+1)])
    models_list.reverse()
    cb.ax.set_yticklabels(models_list)
##################################################
    tax._redraw_labels()
    fig.tight_layout()
    fig.savefig(str("Fig11_CMIP5_JJA_CEU_26ch5_byIM_recommendations.png"),bbox_inches='tight',dpi=300)

if __name__ == "__main__":
    main()
