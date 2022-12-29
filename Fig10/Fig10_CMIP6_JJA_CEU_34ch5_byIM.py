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
    filename =dir+"alpha-beta-scan_SM_JJA_CMIP6_both_sum_ch5_sqrt.csv"


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
            data[(ia, 100-ia-ib, ib)] = member_combo_data[ia][ib]

    from matplotlib.colors import ListedColormap,LinearSegmentedColormap
    from matplotlib import colors

##################################

## JJA CMIP6 SM ch5 sqrt
    cmap = ListedColormap([
    "firebrick", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, KIOST-ESM-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "indianred", # CanESM5-r16i1p1f1, FGOALS-g3-r2i1p1f1, KIOST-ESM-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "crimson", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, FGOALS-g3-r2i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "lightcoral", # CESM2-WACCM-r2i1p1f1, E3SM-1-1-r1i1p1f1, IPSL-CM6A-LR-r6i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "orangered", # CESM2-r11i1p1f1, FGOALS-g3-r2i1p1f1, IPSL-CM6A-LR-r6i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "coral", # CESM2-WACCM-r2i1p1f1, FGOALS-g3-r2i1p1f1, IPSL-CM6A-LR-r6i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "orange", # E3SM-1-1-r1i1p1f1, IPSL-CM6A-LR-r6i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, TaiESM1-r1i1p1f1
    "goldenrod",  # IPSL-CM6A-LR-r6i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "gold", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, TaiESM1-r1i1p1f1
    "yellow", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-CM4-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "tab:olive", # CanESM5-r16i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "tab:brown", # GFDL-CM4-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "sienna",  # HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "peru", # AWI-CM-1-1-MR-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "olivedrab", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "yellowgreen", # GFDL-CM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "darkgreen", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, FGOALS-g3-r2i1p1f1, MIROC-ES2L-r1i1p1f2, UKESM1-0-LL-r1i1p1f2
    "tab:green", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, UKESM1-0-LL-r1i1p1f2
    "mediumseagreen", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-CM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2
    "limegreen", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, MIROC6-r15i1p1f1, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "springgreen", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "palegreen", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, FGOALS-g3-r2i1p1f1, MIROC6-r15i1p1f1, UKESM1-0-LL-r1i1p1f2
    "darkturquoise", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, MIROC-ES2L-r1i1p1f2, UKESM1-0-LL-r1i1p1f2
    "aquamarine", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2
    "navy", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2
    "mediumblue", # AWI-CM-1-1-MR-r1i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2, TaiESM1-r1i1p1f1
    "blue", # AWI-CM-1-1-MR-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "dodgerblue", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, MIROC-ES2L-r1i1p1f2, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "deepskyblue", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "lightskyblue", # AWI-CM-1-1-MR-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, UKESM1-0-LL-r1i1p1f2
    "lightcyan", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "mediumslateblue", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3
    "blueviolet", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, TaiESM1-r1i1p1f1
    "indigo", # AWI-CM-1-1-MR-r1i1p1f1, GFDL-CM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "darkorchid", # AWI-CM-1-1-MR-r1i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "violet", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, FGOALS-g3-r2i1p1f1, MPI-ESM1-2-LR-r10i1p1f1, UKESM1-0-LL-r1i1p1f2
    "plum", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, FGOALS-g3-r2i1p1f1, UKESM1-0-LL-r1i1p1f2
    "mediumvioletred", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "deeppink", # CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-LL-r3i1p1f3, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "pink", # CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "lavenderblush", # AWI-CM-1-1-MR-r1i1p1f1, CanESM5-r16i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "ivory", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MPI-ESM1-2-HR-r1i1p1f1
    "silver", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MPI-ESM1-2-HR-r1i1p1f1, TaiESM1-r1i1p1f1
    "tab:gray", # AWI-CM-1-1-MR-r1i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MPI-ESM1-2-HR-r1i1p1f1, TaiESM1-r1i1p1f1
    "k"]) # AWI-CM-1-1-MR-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MPI-ESM1-2-HR-r1i1p1f1, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1

    cmap_r = cmap.reversed()
    cb_kwargs = {"ticks" : np.arange(0,v+1)}

##################################################
    fig = plt.figure(figsize=(15,6))
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
    fig.savefig(str("Fig10_CMIP6_JJA_CEU_34ch5_byIM.png"),bbox_inches='tight',dpi=300)

# JJA CMIP6 SM ch5 sqrt (recommendation)
    cmap = ListedColormap([
    "w", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, KIOST-ESM-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "w", # CanESM5-r16i1p1f1, FGOALS-g3-r2i1p1f1, KIOST-ESM-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "w", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, FGOALS-g3-r2i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "w", # CESM2-WACCM-r2i1p1f1, E3SM-1-1-r1i1p1f1, IPSL-CM6A-LR-r6i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "w", # CESM2-r11i1p1f1, FGOALS-g3-r2i1p1f1, IPSL-CM6A-LR-r6i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "w", # CESM2-WACCM-r2i1p1f1, FGOALS-g3-r2i1p1f1, IPSL-CM6A-LR-r6i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "w", # E3SM-1-1-r1i1p1f1, IPSL-CM6A-LR-r6i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, TaiESM1-r1i1p1f1
    "w",  # IPSL-CM6A-LR-r6i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "w", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, TaiESM1-r1i1p1f1
    "w", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-CM4-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "w", # CanESM5-r16i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "w", # GFDL-CM4-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "w",  # HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "w", # AWI-CM-1-1-MR-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "w", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1
    "w", # GFDL-CM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "w", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, FGOALS-g3-r2i1p1f1, MIROC-ES2L-r1i1p1f2, UKESM1-0-LL-r1i1p1f2
    "w", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MIROC6-r15i1p1f1, UKESM1-0-LL-r1i1p1f2
    "w", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-CM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2
    "w", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, MIROC6-r15i1p1f1, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "w", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, MIROC-ES2L-r1i1p1f2, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "w", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, FGOALS-g3-r2i1p1f1, MIROC6-r15i1p1f1, UKESM1-0-LL-r1i1p1f2
    "w", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, MIROC-ES2L-r1i1p1f2, UKESM1-0-LL-r1i1p1f2
    "w", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2
    "w", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2
    "mediumblue", # AWI-CM-1-1-MR-r1i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2, TaiESM1-r1i1p1f1
    "blue", # AWI-CM-1-1-MR-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MIROC-ES2L-r1i1p1f2, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "w", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, MIROC-ES2L-r1i1p1f2, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "deepskyblue", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "w", # AWI-CM-1-1-MR-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, UKESM1-0-LL-r1i1p1f2
    "w", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "mediumslateblue", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3
    "w", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, TaiESM1-r1i1p1f1
    "indigo", # AWI-CM-1-1-MR-r1i1p1f1, GFDL-CM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "darkorchid", # AWI-CM-1-1-MR-r1i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1
    "w", # CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, FGOALS-g3-r2i1p1f1, MPI-ESM1-2-LR-r10i1p1f1, UKESM1-0-LL-r1i1p1f2
    "w", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, FGOALS-g3-r2i1p1f1, UKESM1-0-LL-r1i1p1f2
    "w", # CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, E3SM-1-1-r1i1p1f1, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "w", # CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-LL-r3i1p1f3, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "w", # CanESM5-r16i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "w", # AWI-CM-1-1-MR-r1i1p1f1, CanESM5-r16i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MPI-ESM1-2-HR-r1i1p1f1, UKESM1-0-LL-r1i1p1f2
    "w", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, CanESM5-r16i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MPI-ESM1-2-HR-r1i1p1f1
    "w", # AWI-CM-1-1-MR-r1i1p1f1, CMCC-ESM2-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MPI-ESM1-2-HR-r1i1p1f1, TaiESM1-r1i1p1f1
    "w", # AWI-CM-1-1-MR-r1i1p1f1, GFDL-ESM4-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MPI-ESM1-2-HR-r1i1p1f1, TaiESM1-r1i1p1f1
    "w"]) # AWI-CM-1-1-MR-r1i1p1f1, HadGEM3-GC31-MM-r1i1p1f3, MPI-ESM1-2-HR-r1i1p1f1, MRI-ESM2-0-r1i1p1f1, TaiESM1-r1i1p1f1

    cmap_r = cmap.reversed()
    cb_kwargs = {"ticks" : np.arange(0,v+1)}

##################################################
    fig = plt.figure(figsize=(15,6))
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
    fig.savefig(str("Fig10_CMIP6_JJA_CEU_34ch5_byIM_recommendations.png"),bbox_inches='tight',dpi=300)

if __name__ == "__main__":
    main()
