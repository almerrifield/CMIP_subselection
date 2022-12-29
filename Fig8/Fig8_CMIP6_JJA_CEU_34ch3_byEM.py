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
    filename =dir+"alpha-beta-scan_EM_JJA_CMIP6_both_sum_ch3_sqrt.csv"

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        data = []
        for d in reader:
            d['alpha'] = np.round(float(d['alpha']), 3)
            d['beta'] = np.round(float(d['beta']), 3)
            d['models_str'] = ', '.join(sorted([d['member0'],d['member1'],d['member2']]))
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

## Colormaps
##################################

# JJA CMIP6 EM ch3 sqrt
    cmap = ListedColormap([
    "firebrick", # FGOALS-g3-r0i0p0f0, KIOST-ESM-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "crimson", #CESM2-r0i0p0f0, KIOST-ESM-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "lightcoral", #CanESM5-r0i0p0f0, FGOALS-g3-r0i0p0f0, MIROC-ES2L-r0i0p0f0
    "orangered", #CanESM5-r0i0p0f0, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "orange", #CESM2-r0i0p0f0, CNRM-CM6-1-HR-r1i1p1f2, MIROC-ES2L-r0i0p0f0
    "gold", #CESM2-WACCM-r0i0p0f0, CNRM-CM6-1-HR-r1i1p1f2, MIROC-ES2L-r0i0p0f0
    "yellow", #E3SM-1-1-r1i1p1f1, FGOALS-g3-r0i0p0f0, MIROC-ES2L-r0i0p0f0
    "sienna", #CESM2-r0i0p0f0, IPSL-CM6A-LR-r0i0p0f0, MIROC-ES2L-r0i0p0f0
    "peru", #CESM2-r0i0p0f0, GFDL-CM4-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "olivedrab", #CESM2-WACCM-r0i0p0f0, IPSL-CM6A-LR-r0i0p0f0, MIROC-ES2L-r0i0p0f0
    "yellowgreen", #E3SM-1-1-r1i1p1f1, MIROC-ES2L-r0i0p0f0, TaiESM1-r1i1p1f1
    "darkgreen", #AWI-CM-1-1-MR-r1i1p1f1, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "mediumseagreen", #E3SM-1-1-r1i1p1f1, MIROC-ES2L-r0i0p0f0, UKESM1-0-LL-r0i0p0f0
    "limegreen", #GFDL-CM4-r1i1p1f1, MIROC-ES2L-r0i0p0f0, TaiESM1-r1i1p1f1
    "springgreen", #FGOALS-g3-r0i0p0f0, MIROC-ES2L-r0i0p0f0, UKESM1-0-LL-r0i0p0f0
    "palegreen", #CMCC-ESM2-r1i1p1f1, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "darkturquoise", #CMCC-ESM2-r1i1p1f1, GFDL-CM4-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "aquamarine", #AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "navy", #CanESM5-r0i0p0f0, MIROC-ES2L-r0i0p0f0, UKESM1-0-LL-r0i0p0f0
    "mediumblue", #CMCC-ESM2-r1i1p1f1, GFDL-ESM4-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "dodgerblue", #AWI-CM-1-1-MR-r1i1p1f1, MIROC-ES2L-r0i0p0f0, TaiESM1-r1i1p1f1
    "deepskyblue", #CMCC-ESM2-r1i1p1f1, CanESM5-r0i0p0f0, MIROC-ES2L-r0i0p0f0
    "lightskyblue", #AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, CanESM5-r0i0p0f0
    "lightcyan", #CMCC-ESM2-r1i1p1f1, CanESM5-r0i0p0f0, GFDL-ESM4-r1i1p1f1
    "mediumslateblue", #GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r0i0p0f0, TaiESM1-r1i1p1f1
    "indigo", #AWI-CM-1-1-MR-r1i1p1f1, GFDL-ESM4-r1i1p1f1, UKESM1-0-LL-r0i0p0f0
    "darkorchid", #AWI-CM-1-1-MR-r1i1p1f1, GFDL-CM4-r1i1p1f1, HadGEM3-GC31-MM-r0i0p0f0
    "violet", #E3SM-1-1-r1i1p1f1, FGOALS-f3-L-r1i1p1f1, UKESM1-0-LL-r0i0p0f0
    "plum", #AWI-CM-1-1-MR-r1i1p1f1, HadGEM3-GC31-MM-r0i0p0f0, TaiESM1-r1i1p1f1
    "mediumvioletred", #E3SM-1-1-r1i1p1f1, FGOALS-g3-r0i0p0f0, UKESM1-0-LL-r0i0p0f0
    "deeppink", #CMCC-ESM2-r1i1p1f1, E3SM-1-1-r1i1p1f1, FGOALS-g3-r0i0p0f0
    "pink", #CanESM5-r0i0p0f0, GFDL-ESM4-r1i1p1f1, UKESM1-0-LL-r0i0p0f0
    "lavenderblush", #CMCC-ESM2-r1i1p1f1, E3SM-1-1-r1i1p1f1, GFDL-ESM4-r1i1p1f1
    "ivory", #CMCC-ESM2-r1i1p1f1, GFDL-ESM4-r1i1p1f1, TaiESM1-r1i1p1f1
    "silver", #AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, TaiESM1-r1i1p1f1
    "tab:gray", #AWI-CM-1-1-MR-r1i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r0i0p0f0
    "k"]) #AWI-CM-1-1-MR-r1i1p1f1, HadGEM3-GC31-MM-r0i0p0f0, MPI-ESM1-2-HR-r0i0p0f0

## Plots
    cmap_r = cmap.reversed()
    cb_kwargs = {"ticks" : np.arange(0,v+1)}
##################################################
    fig = plt.figure(figsize=(13,6)) #8,6, 14,10
    ax = fig.add_subplot(111)

    tax = ternary.TernaryAxesSubplot(ax=ax, scale=100)
    # Remove default Matplotlib Axes
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
    cb.ax.set_yticklabels(models_list,fontsize=10)
##################################################
    tax._redraw_labels()
    fig.tight_layout()
    fig.savefig(str("Fig8_CMIP6_JJA_CEU_34ch3_byEM.png"),bbox_inches='tight',dpi=300)

## Plots
# JJA CMIP6 EM ch3 sqrt (recommendation)
    cmap = ListedColormap([
    "w", # FGOALS-g3-r0i0p0f0, KIOST-ESM-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "w", #CESM2-r0i0p0f0, KIOST-ESM-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "w", #CanESM5-r0i0p0f0, FGOALS-g3-r0i0p0f0, MIROC-ES2L-r0i0p0f0
    "w", #CanESM5-r0i0p0f0, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "w", #CESM2-r0i0p0f0, CNRM-CM6-1-HR-r1i1p1f2, MIROC-ES2L-r0i0p0f0
    "w", #CESM2-WACCM-r0i0p0f0, CNRM-CM6-1-HR-r1i1p1f2, MIROC-ES2L-r0i0p0f0
    "w", #E3SM-1-1-r1i1p1f1, FGOALS-g3-r0i0p0f0, MIROC-ES2L-r0i0p0f0
    "sienna", #CESM2-r0i0p0f0, IPSL-CM6A-LR-r0i0p0f0, MIROC-ES2L-r0i0p0f0
    "peru", #CESM2-r0i0p0f0, GFDL-CM4-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "w", #CESM2-WACCM-r0i0p0f0, IPSL-CM6A-LR-r0i0p0f0, MIROC-ES2L-r0i0p0f0
    "w", #E3SM-1-1-r1i1p1f1, MIROC-ES2L-r0i0p0f0, TaiESM1-r1i1p1f1
    "w", #AWI-CM-1-1-MR-r1i1p1f1, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "w", #E3SM-1-1-r1i1p1f1, MIROC-ES2L-r0i0p0f0, UKESM1-0-LL-r0i0p0f0
    "limegreen", #GFDL-CM4-r1i1p1f1, MIROC-ES2L-r0i0p0f0, TaiESM1-r1i1p1f1
    "w", #FGOALS-g3-r0i0p0f0, MIROC-ES2L-r0i0p0f0, UKESM1-0-LL-r0i0p0f0
    "w", #CMCC-ESM2-r1i1p1f1, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "darkturquoise", #CMCC-ESM2-r1i1p1f1, GFDL-CM4-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "w", #AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "navy", #CanESM5-r0i0p0f0, MIROC-ES2L-r0i0p0f0, UKESM1-0-LL-r0i0p0f0
    "mediumblue", #CMCC-ESM2-r1i1p1f1, GFDL-ESM4-r1i1p1f1, MIROC-ES2L-r0i0p0f0
    "w", #AWI-CM-1-1-MR-r1i1p1f1, MIROC-ES2L-r0i0p0f0, TaiESM1-r1i1p1f1
    "deepskyblue", #CMCC-ESM2-r1i1p1f1, CanESM5-r0i0p0f0, MIROC-ES2L-r0i0p0f0
    "lightskyblue", #AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, CanESM5-r0i0p0f0
    "lightcyan", #CMCC-ESM2-r1i1p1f1, CanESM5-r0i0p0f0, GFDL-ESM4-r1i1p1f1
    "mediumslateblue", #GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r0i0p0f0, TaiESM1-r1i1p1f1
    "indigo", #AWI-CM-1-1-MR-r1i1p1f1, GFDL-ESM4-r1i1p1f1, UKESM1-0-LL-r0i0p0f0
    "darkorchid", #AWI-CM-1-1-MR-r1i1p1f1, GFDL-CM4-r1i1p1f1, HadGEM3-GC31-MM-r0i0p0f0
    "w", #E3SM-1-1-r1i1p1f1, FGOALS-f3-L-r1i1p1f1, UKESM1-0-LL-r0i0p0f0
    "plum", #AWI-CM-1-1-MR-r1i1p1f1, HadGEM3-GC31-MM-r0i0p0f0, TaiESM1-r1i1p1f1
    "w", #E3SM-1-1-r1i1p1f1, FGOALS-g3-r0i0p0f0, UKESM1-0-LL-r0i0p0f0
    "w", #CMCC-ESM2-r1i1p1f1, E3SM-1-1-r1i1p1f1, FGOALS-g3-r0i0p0f0
    "pink", #CanESM5-r0i0p0f0, GFDL-ESM4-r1i1p1f1, UKESM1-0-LL-r0i0p0f0
    "w", #CMCC-ESM2-r1i1p1f1, E3SM-1-1-r1i1p1f1, GFDL-ESM4-r1i1p1f1
    "w", #CMCC-ESM2-r1i1p1f1, GFDL-ESM4-r1i1p1f1, TaiESM1-r1i1p1f1
    "w", #AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, TaiESM1-r1i1p1f1
    "tab:gray", #AWI-CM-1-1-MR-r1i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r0i0p0f0
    "w"]) #AWI-CM-1-1-MR-r1i1p1f1, HadGEM3-GC31-MM-r0i0p0f0, MPI-ESM1-2-HR-r0i0p0f0

## Plots
    cmap_r = cmap.reversed()
    cb_kwargs = {"ticks" : np.arange(0,v+1)}
##################################################
    fig = plt.figure(figsize=(13,6)) #8,6, 14,10
    ax = fig.add_subplot(111)

    tax = ternary.TernaryAxesSubplot(ax=ax, scale=100)
    # Remove default Matplotlib Axes
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
    cb.ax.set_yticklabels(models_list,fontsize=10)
##################################################
    tax._redraw_labels()
    fig.tight_layout()
    fig.savefig(str("Fig8_CMIP6_JJA_CEU_34ch3_byEM_recommendations.png"),bbox_inches='tight',dpi=300)

if __name__ == "__main__":
    main()
