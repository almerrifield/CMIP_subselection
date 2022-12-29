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
    filename =dir+"alpha-beta-scan_EM_JJA_CMIP6_both_sum_ch3_sqrt.comp.csv"

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = []
        for d in reader:
            g = {}
            g['alpha'] = np.round(float(d[0]), 3)
            g['beta'] = np.round(float(d[1]), 3)
            g['perf_comp'] = np.round(float(d[3]), 3)
            g['dist_comp'] = np.round(float(d[4]), 3)
            g['change_comp'] = np.round(float(d[5]), 3)
            data.append(g)

    alphas = list(sorted(list(set([d['alpha'] for d in data]))))
    betas = list(sorted(list(set([d['beta'] for d in data]))))
    models_perf_comp = [[0 for _ in betas] for _ in alphas]
    models_dist_comp = [[0 for _ in betas] for _ in alphas]
    models_change_comp = [[0 for _ in betas] for _ in alphas]

    for d in data:
        ia, ib = alphas.index(d['alpha']),  betas.index(d['beta'])
        assert ia+ib <= 100
        models_perf_comp[ia][ib] = float(d['perf_comp'])
        models_dist_comp[ia][ib] = float(d['dist_comp'])
        models_change_comp[ia][ib] = float(d['change_comp'])

    ds_test = xr.Dataset(dict(perf_comp=(['alpha','beta'],models_perf_comp),dist_comp=(['alpha','beta'],models_dist_comp),change_comp=(['alpha','beta'],models_change_comp)), coords=dict(alpha=alphas,beta=betas))


    data = {}
    for a in ds_test.alpha.data:
        for b in ds_test.beta.data:
            c = round(1-a-b,4)
            if c < 0:
                continue
            ia, ib = alphas.index(a),  betas.index(b)
            data[(ia, 100-ia-ib, ib)] = -models_change_comp[ia][ib] ## change here, make ind and spread negative

    fig = plt.figure(figsize=(8,6))
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
    tax.heatmap(data, cmap='YlGnBu_r', style="hexagonal", vmin=-8, vmax=8)
    tax._redraw_labels()
    fig.tight_layout()
    fig.savefig(str("SupFig14_CMIP6_JJA_CEU_34ch3_byEM_spread.png"),bbox_inches='tight',dpi=300)

if __name__ == "__main__":
    main()
