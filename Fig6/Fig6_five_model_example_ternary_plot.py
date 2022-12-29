###########################
# ternary plot
###########################
# need to enter the ternary environment
# qce
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

dir = '/**/CMIP_subselection/Data/'

filename=dir+"alpha-beta-scan_EM_JJA_CMIP6_both_sum_5example_v1.csv"

with open(filename, 'r') as f:
    reader = csv.DictReader(f)
    data = []
    for d in reader:
        d['alpha'] = np.round(float(d['alpha']), 3)
        d['beta'] = np.round(float(d['beta']), 3)
        d['models_str'] = ', '.join(sorted([d['member0'],d['member1']])) # choose 2 subselection
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


data = {}
for a in ds_test.alpha.data:
    for b in ds_test.beta.data:
        c = round(1-a-b,4)
        if c < 0:
            continue
        ia, ib = alphas.index(a),  betas.index(b)
        data[(ia, 100-ia-ib, ib)] = member_combo_data[ia][ib]


from matplotlib.colors import ListedColormap
from matplotlib import colors
cmap = ListedColormap([
"red",
"yellow",#3
"mediumseagreen",#6
"navy",#12
"darkorchid",#13
"deeppink"]) #15

cmap_r = cmap.reversed()

fig = plt.figure(figsize=(11,6))
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
#tax.heatmap(data, cmap=cmap_r, style="hexagonal")
##################################################
#check of colorbar labels
ternary.heatmap(data, scale=1, ax=ax, style="hexagonal", cmap=cmap_r, colorbar=False)
models_list = [k for k,v in sorted(models_to_number.items(), key=lambda it: it[1])]
vmin = min(data.values())
vmax = max(data.values())
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) #_r
sm._A = []
cb = plt.colorbar(sm, ax=ax)
cb.set_ticks([(len(models_list)-1)*(i-0.5)/len(models_list) for i in range(1,len(models_list)+1)])
models_list.reverse()
cb.ax.set_yticklabels(models_list,fontsize=10)
##################################################
tax._redraw_labels()
fig.tight_layout()
fig.savefig(str("Fig6_five_model_example_ternary.png"),bbox_inches='tight',dpi=300)
