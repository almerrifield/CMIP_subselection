# Figure 4: makes ECS indpendence constraint distribution plot

#################################
# packages
#################################

import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import regionmask
import cdo

import datetime
from scipy import signal, stats
from statsmodels.stats.weightstats import DescrStatsW
import xskillscore

CMIP5_ECS_all = {
'ACCESS1-0':3.83,
'ACCESS1-3':3.53,
"BNU-ESM":3.92,
'CCSM4':2.94,
'CESM1-CAM5':3.4, # Bacmeister
'CNRM-CM5':3.25,
'CSIRO-Mk3-6-0':4.08,
'CanESM2':3.69,
"EC-EARTH":3.34, # Wyser
"FGOALS-g2":3.38,
'GFDL-CM3':3.97,
'GFDL-ESM2G':2.39,
'GFDL-ESM2M':2.44,
'GISS-E2-H':2.31,
'GISS-E2-R':2.11,
'HadGEM2-ES':4.61,
'IPSL-CM5A-LR':4.13,
'IPSL-CM5A-MR':4.12,
'IPSL-CM5B-LR':2.60,
'MIROC-ESM':4.67,
'MIROC5':2.72,
'MPI-ESM-LR':3.63,
'MPI-ESM-MR':3.46,
'MRI-CGCM3':2.60,
'NorESM1-ME':2.99, # Seland
'NorESM1-M':2.80,
'bcc-csm1-1-m':2.86,
'bcc-csm1-1':2.83,
'inmcm4':2.08}

coords, values = zip(* list(CMIP5_ECS_all.items()))
ds_ECS_all = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

## Bootstrap
had_ind = [0,1,15]
cnrm_ind = [5,8]
bcc_ind = [2,27]
nor_ind = [24,25]
ipsl_ind = [16,17]
mpi_ind = [21,22]
gfdl_ind = [11,12]
giss_ind = [13,14]

ind_3 = np.random.randint(0,3,10000)
ind_2a = np.random.randint(0,2,10000)
ind_2b = np.random.randint(0,2,10000)
ind_2c = np.random.randint(0,2,10000)
ind_2d = np.random.randint(0,2,10000)
ind_2e = np.random.randint(0,2,10000)
ind_2f = np.random.randint(0,2,10000)
ind_2g = np.random.randint(0,2,10000)

ds_ECS_subset=[]
ind = np.arange(0,10000,1)
for ii in ind:
    ds_ECS_sub = ds_ECS_all.isel(model = [had_ind[ind_3[ii]], bcc_ind[ind_2a[ii]], 3, 4, cnrm_ind[ind_2b[ii]],
           6, 7, 9, 10, gfdl_ind[ind_2c[ii]], giss_ind[ind_2d[ii]],
           ipsl_ind[ind_2e[ii]], 18, 19, 20,
           mpi_ind[ind_2f[ii]], 23, nor_ind[ind_2g[ii]],
           26,28])
    #print(ds_ECS_sub.data)
    ds_ECS_subset.append(ds_ECS_sub.data)


def flatten(t):
    return [item for sublist in t for item in sublist]

ds_ECS_subset_flat = flatten(ds_ECS_subset)
ds_ECS_subset_reshape = np.reshape(ds_ECS_subset_flat,(10000,20))

lows_5 = []
low_likely_5 = []
mids_5 = []
high_likely_5 = []
highs_5 = []
ind = np.arange(0,10000)
for ii in ind:
    low_quan =  np.quantile(ds_ECS_subset_reshape[ii],0.05)
    lows_5.append(low_quan)
    low_likely_quan =  np.quantile(ds_ECS_subset_reshape[ii],0.25) #change here
    low_likely_5.append(low_likely_quan)
    mids_quan =  np.quantile(ds_ECS_subset_reshape[ii],0.5)
    mids_5.append(mids_quan)
    high_likely_quan =  np.quantile(ds_ECS_subset_reshape[ii],0.75) #change here
    high_likely_5.append(high_likely_quan)
    highs_quan =  np.quantile(ds_ECS_subset_reshape[ii],0.95)
    highs_5.append(highs_quan)


#####


CMIP6_ECS_all = {
'ACCESS-CM2':4.72,
'ACCESS-ESM1-5':3.87,
'AWI-CM-1-1-MR':3.16,
'CAS-ESM2-0':3.51,
'CESM2-WACCM':4.75,
'CESM2':5.16,
'CMCC-CM2-SR5':3.52,
'CMCC-ESM2':3.58, # Zelinka
'CNRM-CM6-1-HR':4.28,
'CNRM-CM6-1':4.83,
'CNRM-ESM2-1':4.76,
'CanESM5':5.62,
'E3SM-1-1':5.30, # Golaz
"EC-Earth3-Veg":4.31,
"EC-Earth3":4.26, # Zelinka
'FGOALS-f3-L':3.00,
'FGOALS-g3':2.88,
'GFDL-CM4':3.89, # Zelinka
'GFDL-ESM4':2.65, # Zelinka
'GISS-E2-1-G':2.72,
'HadGEM3-GC31-LL':5.55,
'HadGEM3-GC31-MM':5.42,
'INM-CM4-8':1.83,
'INM-CM5-0':1.92,
'IPSL-CM6A-LR':4.56,
"KACE-1-0-G":4.48,
'KIOST-ESM':3.36, # Pak
"MCM-UA-1-0":3.65,
'MIROC-ES2L':2.68,
'MIROC6':2.61,
'MPI-ESM1-2-HR':2.98,
'MPI-ESM1-2-LR':3.00,
'MRI-ESM2-0':3.15,
'NESM3':4.72,
'NorESM2-MM':2.50,
'TaiESM1':4.31,
'UKESM1-0-LL':5.34}

coords, values = zip(* list(CMIP6_ECS_all.items()))
ds6_ECS_all = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# Bootstrap
had_ind = [0,20,21,25,36]
cnrm_ind = [8,9,10,24]
cesm_ind = [4,5,6,7,34,35]
earth_ind = [13,14]
inm_ind = [22,23]
mpi_ind = [2,30,31,33]
gfdl_ind = [17,18]

ind_5 = np.random.randint(0,5,10000)
ind_4a = np.random.randint(0,4,10000)
ind_6 = np.random.randint(0,6,10000)
ind_2a = np.random.randint(0,2,10000)
ind_2b = np.random.randint(0,2,10000)
ind_4b = np.random.randint(0,4,10000)
ind_2c = np.random.randint(0,2,10000)

ds6_ECS_subset=[]
ind = np.arange(0,10000,1)
for ii in ind:
    ds6_ECS_sub = ds6_ECS_all.isel(model = [had_ind[ind_5[ii]], 1, mpi_ind[ind_4a[ii]], 3,
           cesm_ind[ind_6[ii]], cnrm_ind[ind_4b[ii]], 11, 12, earth_ind[ind_2a[ii]],15, 16, gfdl_ind[ind_2b[ii]],
           19, inm_ind[ind_2c[ii]], 26, 27,
           28, 29, 32])
    #print(ds6_ECS_sub.model)
    ds6_ECS_subset.append(ds6_ECS_sub.data)

def flatten(t):
    return [item for sublist in t for item in sublist]

ds6_ECS_subset_flat = flatten(ds6_ECS_subset)
ds6_ECS_subset_reshape = np.reshape(ds6_ECS_subset_flat,(10000,19))

lows = []
low_likely = []
mids = []
high_likely = []
highs = []
ind = np.arange(0,10000)
for ii in ind:
    low_quan =  np.quantile(ds6_ECS_subset_reshape[ii],0.05)
    lows.append(low_quan)
    low_likely_quan =  np.quantile(ds6_ECS_subset_reshape[ii],0.25) #change here
    low_likely.append(low_likely_quan)
    mids_quan =  np.quantile(ds6_ECS_subset_reshape[ii],0.5)
    mids.append(mids_quan)
    high_likely_quan =  np.quantile(ds6_ECS_subset_reshape[ii],0.75) #change here
    high_likely.append(high_likely_quan)
    highs_quan =  np.quantile(ds6_ECS_subset_reshape[ii],0.95)
    highs.append(highs_quan)

###########################
ds6_ECS_all_05 = np.quantile(ds6_ECS_all,0.05)
ds6_ECS_all_25 = np.quantile(ds6_ECS_all,0.25) #change here
ds6_ECS_all_50 = np.quantile(ds6_ECS_all,0.50)
ds6_ECS_all_75 = np.quantile(ds6_ECS_all,0.75) #change here
ds6_ECS_all_95 = np.quantile(ds6_ECS_all,0.95)

ds_ECS_all_05 = np.quantile(ds_ECS_all,0.05)
ds_ECS_all_25 = np.quantile(ds_ECS_all,0.25) #change here
ds_ECS_all_50 = np.quantile(ds_ECS_all,0.50)
ds_ECS_all_75 = np.quantile(ds_ECS_all,0.75) #change here
ds_ECS_all_95 = np.quantile(ds_ECS_all,0.95)


lows_50 = np.mean(lows)
lows_5_50 = np.mean(lows_5)

low_likely_50 = np.mean(low_likely)
low_likely_5_50 = np.mean(low_likely_5)

mids_50 = np.mean(mids)
mids_5_50 = np.mean(mids_5)

high_likely_50 = np.mean(high_likely)
high_likely_5_50 = np.mean(high_likely_5)

highs_50 = np.mean(highs)
highs_5_50 = np.mean(highs_5)


###################
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)
violin_parts = plt.violinplot([ds6_ECS_all],positions=[1],widths=0.1,showmeans = False, showextrema = False, showmedians = False)
for pc in violin_parts['bodies']:
    pc.set_facecolor('k')
    pc.set_edgecolor('k')
    pc.set_alpha(0.1)
plt.plot([0.975, 1.025],[ds6_ECS_all_05, ds6_ECS_all_05],color="tab:purple")
plt.plot([0.975, 1.025],[ds6_ECS_all_25, ds6_ECS_all_25],color="tab:red")
plt.plot([0.975, 1.025],[ds6_ECS_all_50, ds6_ECS_all_50],color="tab:green")
plt.plot([0.975, 1.025],[ds6_ECS_all_75, ds6_ECS_all_75],color="tab:orange")
plt.plot([0.975, 1.025],[ds6_ECS_all_95, ds6_ECS_all_95],color="tab:blue")

plt.plot([1, 1],[ds6_ECS_all_05,ds6_ECS_all_25],'k')
plt.plot([1, 1],[ds6_ECS_all_75,ds6_ECS_all_95],'k')
plt.plot([0.975, 0.975],[ds6_ECS_all_25,ds6_ECS_all_75],'k')
plt.plot([1.025, 1.025],[ds6_ECS_all_25,ds6_ECS_all_75],'k')

violin_parts = plt.violinplot([lows],positions=[1.15],widths=0.1,showmeans = False, showextrema = False, showmedians = False)
for pc in violin_parts['bodies']:
    pc.set_facecolor('tab:purple')
    pc.set_edgecolor('tab:purple')
    pc.set_alpha(0.2)
violin_parts = plt.violinplot([low_likely],positions=[1.15],widths=0.1,showmeans = False, showextrema = False, showmedians = False)
for pc in violin_parts['bodies']:
    pc.set_facecolor('tab:red')
    pc.set_edgecolor('tab:red')
    pc.set_alpha(0.2)
violin_parts = plt.violinplot([mids],positions=[1.15],widths=0.1,showmeans = False, showextrema = False, showmedians = False)
for pc in violin_parts['bodies']:
    pc.set_facecolor('tab:green')
    pc.set_edgecolor('tab:green')
    pc.set_alpha(0.2)
violin_parts = plt.violinplot([high_likely],positions=[1.15],widths=0.1,showmeans = False, showextrema = False, showmedians = False)
for pc in violin_parts['bodies']:
    pc.set_facecolor('tab:orange')
    pc.set_edgecolor('tab:orange')
    pc.set_alpha(0.2)
violin_parts = plt.violinplot([highs],positions=[1.15],widths=0.1,showmeans = False, showextrema = False, showmedians = False)
for pc in violin_parts['bodies']:
    pc.set_facecolor('tab:blue')
    pc.set_edgecolor('tab:blue')
    pc.set_alpha(0.2)

plt.plot([1.125, 1.175],[lows_50, lows_50],color="tab:purple",linestyle='solid')
plt.plot([1.125, 1.175],[low_likely_50, low_likely_50],color="tab:red",linestyle='solid')
plt.plot([1.125, 1.175],[mids_50, mids_50],color="tab:green",linestyle='solid')
plt.plot([1.125, 1.175],[high_likely_50, high_likely_50],color="tab:orange",linestyle='solid')
plt.plot([1.125, 1.175],[highs_50, highs_50],color="tab:blue",linestyle='solid')

plt.plot([1.025, 1.125],[ds6_ECS_all_95,highs_50],color="tab:blue",linestyle='dotted')
plt.plot([1.025, 1.125],[ds6_ECS_all_75,high_likely_50],color="tab:orange",linestyle='dotted')
plt.plot([1.025, 1.125],[ds6_ECS_all_50,mids_50],color="tab:green",linestyle='dotted')
plt.plot([1.025, 1.125],[ds6_ECS_all_25,low_likely_50],color="tab:red",linestyle='dotted')
plt.plot([1.025, 1.125],[ds6_ECS_all_05,lows_50],color="tab:purple",linestyle='dotted')

plt.plot([1.15, 1.15],[lows_50,low_likely_50],'k',linestyle='solid')
plt.plot([1.15, 1.15],[high_likely_50,highs_50],'k',linestyle='solid')
plt.plot([1.125, 1.125],[low_likely_50,high_likely_50],'k',linestyle='solid')
plt.plot([1.175, 1.175],[low_likely_50,high_likely_50],'k',linestyle='solid')

############
violin_parts = plt.violinplot([ds_ECS_all],positions=[1.4],widths=0.1,showmeans = False, showextrema = False, showmedians = False) #quantiles= [0.05,0.25,0.5,0.75,0.95]

for pc in violin_parts['bodies']:
    pc.set_facecolor('k')
    pc.set_edgecolor('k')
    pc.set_alpha(0.1)
plt.plot([1.375, 1.425],[ds_ECS_all_05, ds_ECS_all_05],color="tab:purple")
plt.plot([1.375, 1.425],[ds_ECS_all_25, ds_ECS_all_25],color="tab:red")
plt.plot([1.375, 1.425],[ds_ECS_all_50, ds_ECS_all_50],color="tab:green")
plt.plot([1.375, 1.425],[ds_ECS_all_75, ds_ECS_all_75],color="tab:orange")
plt.plot([1.375, 1.425],[ds_ECS_all_95, ds_ECS_all_95],color="tab:blue")

plt.plot([1.4, 1.4],[ds_ECS_all_05,ds_ECS_all_25],'k')
plt.plot([1.4, 1.4],[ds_ECS_all_75,ds_ECS_all_95],'k')
plt.plot([1.375, 1.375],[ds_ECS_all_25,ds_ECS_all_75],'k')
plt.plot([1.425, 1.425],[ds_ECS_all_25,ds_ECS_all_75],'k')

violin_parts = plt.violinplot([lows_5],positions=[1.55],widths=0.1,showmeans = False, showextrema = False, showmedians = False)
for pc in violin_parts['bodies']:
    pc.set_facecolor('tab:purple')
    pc.set_edgecolor('tab:purple')
    pc.set_alpha(0.2)
violin_parts = plt.violinplot([low_likely_5],positions=[1.55],widths=0.1,showmeans = False, showextrema = False, showmedians = False)
for pc in violin_parts['bodies']:
    pc.set_facecolor('tab:red')
    pc.set_edgecolor('tab:red')
    pc.set_alpha(0.2)
violin_parts = plt.violinplot([mids_5],positions=[1.55],widths=0.1,showmeans = False, showextrema = False, showmedians = False)
for pc in violin_parts['bodies']:
    pc.set_facecolor('tab:green')
    pc.set_edgecolor('tab:green')
    pc.set_alpha(0.2)
violin_parts = plt.violinplot([high_likely_5],positions=[1.55],widths=0.1,showmeans = False, showextrema = False, showmedians = False)
for pc in violin_parts['bodies']:
    pc.set_facecolor('tab:orange')
    pc.set_edgecolor('tab:orange')
    pc.set_alpha(0.2)
violin_parts = plt.violinplot([highs_5],positions=[1.55],widths=0.1,showmeans = False, showextrema = False, showmedians = False)
for pc in violin_parts['bodies']:
    pc.set_facecolor('tab:blue')
    pc.set_edgecolor('tab:blue')
    pc.set_alpha(0.2)

plt.plot([1.525, 1.575],[lows_5_50, lows_5_50],color="tab:purple",linestyle='solid')
plt.plot([1.525, 1.575],[low_likely_5_50, low_likely_5_50],color="tab:red",linestyle='solid')
plt.plot([1.525, 1.575],[mids_5_50, mids_5_50],color="tab:green",linestyle='solid')
plt.plot([1.525, 1.575],[high_likely_5_50, high_likely_5_50],color="tab:orange",linestyle='solid')
plt.plot([1.525, 1.575],[highs_5_50, highs_5_50],color="tab:blue",linestyle='solid')

plt.plot([1.425, 1.525],[ds_ECS_all_95,highs_5_50],color="tab:blue",linestyle='dotted')
plt.plot([1.425, 1.525],[ds_ECS_all_75,high_likely_5_50],color="tab:orange",linestyle='dotted')
plt.plot([1.425, 1.525],[ds_ECS_all_50,mids_5_50],color="tab:green",linestyle='dotted')
plt.plot([1.425, 1.525],[ds_ECS_all_25,low_likely_5_50],color="tab:red",linestyle='dotted')
plt.plot([1.425, 1.525],[ds_ECS_all_05,lows_5_50],color="tab:purple",linestyle='dotted')

plt.plot([1.55, 1.55],[lows_5_50,low_likely_5_50],'k',linestyle='solid')
plt.plot([1.55, 1.55],[high_likely_5_50,highs_5_50],'k',linestyle='solid')
plt.plot([1.525, 1.525],[low_likely_5_50,high_likely_5_50],'k',linestyle='solid')
plt.plot([1.575, 1.575],[low_likely_5_50,high_likely_5_50],'k',linestyle='solid')

plt.title('Independence Constraint on Equilibrium Climate Sensitivity',fontsize=11,fontweight='bold')
plt.ylim([1.5,6])
plt.ylabel('ECS (˚C; Gregory et al. 2004 Method)')
ax.set_xticks([1,1.15,1.4,1.55])
ax.set_xticklabels(['CMIP6 \n (37)','CMIP6 \n (one per family; 19)', 'CMIP5 \n (29)', 'CMIP5 \n (one per family; 20)'])
plt.savefig('Fig4_ECS_constraint_means.png',bbox_inches='tight',dpi=300)
###
