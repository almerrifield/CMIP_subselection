# Supplementary Figure 4: makes "fingerprint" hatched map plots for PR
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


dirT6 = '/**/CMIP_subselection/Data/'
dsPr6 = xr.open_dataset(dirT6 + 'pr_mon_CMIP6_SSP585_g025_v2_ann.nc',use_cftime = True)
dsPr6 = dsPr6*86400
dsPr6 = dsPr6.sortby(dsPr6.member)

dsPr6_clim = dsPr6.sel(year=slice(1905, 2005)).mean('year')
dsPr6_clim = dsPr6_clim.drop_sel(member=['BCC-CSM2-MR-r1i1p1f1','CIESM-r1i1p1f1','FIO-ESM-2-0-r1i1p1f1','FIO-ESM-2-0-r2i1p1f1', 'FIO-ESM-2-0-r3i1p1f1','EC-Earth3-Veg-r6i1p1f1','NorESM2-LM-r1i1p1f1'])

dsPr6_clim_std = dsPr6_clim.std('member')

dsPr6_clim_miroc_std = dsPr6_clim.sel(member=['MIROC6-r10i1p1f1', 'MIROC6-r11i1p1f1',
'MIROC6-r12i1p1f1', 'MIROC6-r13i1p1f1', 'MIROC6-r14i1p1f1',
'MIROC6-r15i1p1f1', 'MIROC6-r16i1p1f1', 'MIROC6-r17i1p1f1',
'MIROC6-r18i1p1f1', 'MIROC6-r19i1p1f1', 'MIROC6-r1i1p1f1',
'MIROC6-r20i1p1f1', 'MIROC6-r21i1p1f1', 'MIROC6-r22i1p1f1',
'MIROC6-r23i1p1f1', 'MIROC6-r24i1p1f1', 'MIROC6-r25i1p1f1',
'MIROC6-r26i1p1f1', 'MIROC6-r27i1p1f1', 'MIROC6-r28i1p1f1',
'MIROC6-r29i1p1f1', 'MIROC6-r2i1p1f1', 'MIROC6-r30i1p1f1',
'MIROC6-r31i1p1f1', 'MIROC6-r32i1p1f1', 'MIROC6-r33i1p1f1',
'MIROC6-r34i1p1f1', 'MIROC6-r35i1p1f1', 'MIROC6-r36i1p1f1',
'MIROC6-r37i1p1f1', 'MIROC6-r38i1p1f1', 'MIROC6-r39i1p1f1',
'MIROC6-r3i1p1f1', 'MIROC6-r40i1p1f1', 'MIROC6-r41i1p1f1',
'MIROC6-r42i1p1f1', 'MIROC6-r43i1p1f1', 'MIROC6-r44i1p1f1',
'MIROC6-r45i1p1f1', 'MIROC6-r46i1p1f1', 'MIROC6-r47i1p1f1',
'MIROC6-r48i1p1f1', 'MIROC6-r49i1p1f1', 'MIROC6-r4i1p1f1',
'MIROC6-r50i1p1f1', 'MIROC6-r5i1p1f1', 'MIROC6-r6i1p1f1',
'MIROC6-r7i1p1f1', 'MIROC6-r8i1p1f1', 'MIROC6-r9i1p1f1']).std('member')

dsPr6_clim_can_std = dsPr6_clim.sel(member=['CanESM5-r10i1p1f1', 'CanESM5-r11i1p1f1',
'CanESM5-r12i1p1f1','CanESM5-r13i1p1f1', 'CanESM5-r14i1p1f1',
'CanESM5-r15i1p1f1', 'CanESM5-r16i1p1f1', 'CanESM5-r17i1p1f1',
'CanESM5-r18i1p1f1', 'CanESM5-r19i1p1f1', 'CanESM5-r1i1p1f1',
'CanESM5-r20i1p1f1','CanESM5-r21i1p1f1', 'CanESM5-r22i1p1f1',
'CanESM5-r23i1p1f1', 'CanESM5-r24i1p1f1', 'CanESM5-r25i1p1f1',
'CanESM5-r2i1p1f1', 'CanESM5-r3i1p1f1', 'CanESM5-r4i1p1f1',
'CanESM5-r5i1p1f1', 'CanESM5-r6i1p1f1', 'CanESM5-r7i1p1f1',
'CanESM5-r8i1p1f1', 'CanESM5-r9i1p1f1']).std('member')

dsPr6_clim_access_std = dsPr6_clim.sel(member=['ACCESS-ESM1-5-r10i1p1f1',
'ACCESS-ESM1-5-r1i1p1f1', 'ACCESS-ESM1-5-r2i1p1f1',
'ACCESS-ESM1-5-r3i1p1f1', 'ACCESS-ESM1-5-r4i1p1f1',
'ACCESS-ESM1-5-r5i1p1f1', 'ACCESS-ESM1-5-r6i1p1f1',
'ACCESS-ESM1-5-r7i1p1f1', 'ACCESS-ESM1-5-r8i1p1f1',
'ACCESS-ESM1-5-r9i1p1f1']).std('member')

dsPr6_clim_cesm_std = dsPr6_clim.sel(member=['CESM2-r10i1p1f1', 'CESM2-r11i1p1f1',
'CESM2-r1i1p1f1', 'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1']).std('member')

dsPr6_clim_cnrm_std = dsPr6_clim.sel(member=['CNRM-CM6-1-r1i1p1f2',
'CNRM-CM6-1-r2i1p1f2', 'CNRM-CM6-1-r3i1p1f2',
'CNRM-CM6-1-r4i1p1f2', 'CNRM-CM6-1-r5i1p1f2',
'CNRM-CM6-1-r6i1p1f2']).std('member')

dsPr6_clim_earth_std = dsPr6_clim.sel(member=['EC-Earth3-r11i1p1f1', 'EC-Earth3-r13i1p1f1',
'EC-Earth3-r15i1p1f1', 'EC-Earth3-r1i1p1f1', 'EC-Earth3-r3i1p1f1',
'EC-Earth3-r4i1p1f1', 'EC-Earth3-r6i1p1f1', 'EC-Earth3-r9i1p1f1']).std('member')

dsPr6_clim_giss_std = dsPr6_clim.sel(member=['GISS-E2-1-G-r1i1p3f1',
'GISS-E2-1-G-r2i1p3f1','GISS-E2-1-G-r3i1p3f1', 'GISS-E2-1-G-r4i1p3f1',
'GISS-E2-1-G-r5i1p3f1']).std('member')

dsPr6_clim_ipsl_std = dsPr6_clim.sel(member=['IPSL-CM6A-LR-r14i1p1f1',
'IPSL-CM6A-LR-r1i1p1f1', 'IPSL-CM6A-LR-r2i1p1f1',
'IPSL-CM6A-LR-r3i1p1f1', 'IPSL-CM6A-LR-r4i1p1f1',
'IPSL-CM6A-LR-r6i1p1f1']).std('member')

dsPr6_clim_mpi_std = dsPr6_clim.sel(member=['MPI-ESM1-2-LR-r10i1p1f1', 'MPI-ESM1-2-LR-r1i1p1f1',
'MPI-ESM1-2-LR-r2i1p1f1', 'MPI-ESM1-2-LR-r3i1p1f1',
'MPI-ESM1-2-LR-r4i1p1f1', 'MPI-ESM1-2-LR-r5i1p1f1',
'MPI-ESM1-2-LR-r6i1p1f1', 'MPI-ESM1-2-LR-r7i1p1f1',
'MPI-ESM1-2-LR-r8i1p1f1', 'MPI-ESM1-2-LR-r9i1p1f1']).std('member')

dsPr6_clim_ukesm_std = dsPr6_clim.sel(member=['UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2','UKESM1-0-LL-r8i1p1f2']).std('member')


dsPr6_clim_miroc_e_std = dsPr6_clim.sel(member=['MIROC-ES2L-r10i1p1f2',
'MIROC-ES2L-r1i1p1f2', 'MIROC-ES2L-r2i1p1f2',
'MIROC-ES2L-r3i1p1f2', 'MIROC-ES2L-r4i1p1f2',
'MIROC-ES2L-r5i1p1f2', 'MIROC-ES2L-r6i1p1f2',
'MIROC-ES2L-r7i1p1f2', 'MIROC-ES2L-r8i1p1f2',
'MIROC-ES2L-r9i1p1f2']).std('member')

dsPr6_clim_cnrm_e_std = dsPr6_clim.sel(member=['CNRM-ESM2-1-r1i1p1f2',
'CNRM-ESM2-1-r2i1p1f2', 'CNRM-ESM2-1-r3i1p1f2',
'CNRM-ESM2-1-r4i1p1f2', 'CNRM-ESM2-1-r5i1p1f2']).std('member')

dsPr6_clim_iv_std = xr.concat([dsPr6_clim_miroc_std,dsPr6_clim_can_std,dsPr6_clim_access_std,
dsPr6_clim_cesm_std,dsPr6_clim_cnrm_std,dsPr6_clim_cnrm_e_std,dsPr6_clim_earth_std,
dsPr6_clim_giss_std,dsPr6_clim_ipsl_std,dsPr6_clim_mpi_std,dsPr6_clim_miroc_e_std,
dsPr6_clim_ukesm_std],dim='member')

dsPr6_clim_iv_std_mean = dsPr6_clim_iv_std.median('member')

dsPr6_clim_one_std = dsPr6_clim.sel(member=['ACCESS-CM2-r1i1p1f1',
'ACCESS-ESM1-5-r1i1p1f1', 'AWI-CM-1-1-MR-r1i1p1f1','CAS-ESM2-0-r1i1p1f1',
'CESM2-WACCM-r1i1p1f1', 'CESM2-r1i1p1f1', 'CMCC-CM2-SR5-r1i1p1f1', 'CMCC-ESM2-r1i1p1f1',
'CNRM-CM6-1-HR-r1i1p1f2', 'CNRM-CM6-1-r1i1p1f2','CNRM-ESM2-1-r1i1p1f2','CanESM5-r1i1p1f1',
'E3SM-1-1-r1i1p1f1','EC-Earth3-Veg-r1i1p1f1',
'EC-Earth3-r1i1p1f1','FGOALS-f3-L-r1i1p1f1', 'FGOALS-g3-r1i1p1f1',
'GFDL-CM4-r1i1p1f1','GFDL-ESM4-r1i1p1f1', 'GISS-E2-1-G-r1i1p3f1',
'HadGEM3-GC31-LL-r1i1p1f3', 'HadGEM3-GC31-MM-r1i1p1f3',
'INM-CM4-8-r1i1p1f1','INM-CM5-0-r1i1p1f1','IPSL-CM6A-LR-r1i1p1f1',
'KACE-1-0-G-r1i1p1f1','KIOST-ESM-r1i1p1f1', 'MIROC-ES2L-r1i1p1f2',
'MIROC6-r1i1p1f1', 'MPI-ESM1-2-HR-r1i1p1f1','MPI-ESM1-2-LR-r1i1p1f1',
'MRI-ESM2-0-r1i1p1f1','NESM3-r1i1p1f1','NorESM2-MM-r1i1p1f1',
'TaiESM1-r1i1p1f1', 'UKESM1-0-LL-r1i1p1f2']).std('member') #'MCM-UA-1-0-r1i1p1f2'


a = np.percentile(dsPr6_clim_one_std.pr,15)
n = np.percentile(dsPr6_clim_iv_std_mean.pr,85)

mask_obs = dsPr6_clim_one_std.where(dsPr6_clim_one_std>a)/dsPr6_clim_one_std.where(dsPr6_clim_one_std>a)

mask_obs_1 = mask_obs.sel(lon=np.arange(1.25,181.25,2.5))
mask_obs_2 = mask_obs.sel(lon=np.arange(181.25,360,2.5))
mask_obs_2['lon'] = mask_obs_2.lon - 360
mask_obs_flip = xr.concat([mask_obs_2,mask_obs_1],dim='lon')

mask_obs_Pr6 = np.nan_to_num(mask_obs_flip.pr)

mask_obs13 = dsPr6_clim_iv_std_mean.where(dsPr6_clim_iv_std_mean<n)/dsPr6_clim_iv_std_mean.where(dsPr6_clim_iv_std_mean<n)

mask_obs13_1 = mask_obs13.sel(lon=np.arange(1.25,181.25,2.5))
mask_obs13_2 = mask_obs13.sel(lon=np.arange(181.25,360,2.5))
mask_obs13_2['lon'] = mask_obs13_2.lon - 360
mask_obs13_flip = xr.concat([mask_obs13_2,mask_obs13_1],dim='lon')

mask_obs13_Pr6 = np.nan_to_num(mask_obs13_flip.pr)

dsPr6_clim_mask = mask_obs * mask_obs13 * dsPr6_clim.squeeze(drop=True)

dirT5 = '/**/CMIP_subselection/Data/'
dsPr5 = xr.open_dataset(dirT5 + 'pr_mon_CMIP5_rcp85_g025_v2_ann.nc',use_cftime = True)
dsPr5 = dsPr5*86400
dsPr5 = dsPr5.sortby(dsPr5.member)

dsPr5_clim = dsPr5.sel(year=slice(1905, 2005)).mean('year')

dsPr5_clim = dsPr5_clim.sel(member=['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1', 'BNU-ESM-r1i1p1',
'CCSM4-r1i1p1', 'CCSM4-r2i1p1', 'CCSM4-r3i1p1', 'CCSM4-r4i1p1',
'CCSM4-r5i1p1', 'CCSM4-r6i1p1', 'CESM1-CAM5-r1i1p1',
'CESM1-CAM5-r2i1p1', 'CESM1-CAM5-r3i1p1', 'CNRM-CM5-r10i1p1',
'CNRM-CM5-r1i1p1', 'CNRM-CM5-r2i1p1', 'CNRM-CM5-r4i1p1',
'CNRM-CM5-r6i1p1', 'CSIRO-Mk3-6-0-r10i1p1', 'CSIRO-Mk3-6-0-r1i1p1',
'CSIRO-Mk3-6-0-r2i1p1', 'CSIRO-Mk3-6-0-r3i1p1', 'CSIRO-Mk3-6-0-r4i1p1',
'CSIRO-Mk3-6-0-r5i1p1', 'CSIRO-Mk3-6-0-r6i1p1', 'CSIRO-Mk3-6-0-r7i1p1',
'CSIRO-Mk3-6-0-r8i1p1', 'CSIRO-Mk3-6-0-r9i1p1', 'CanESM2-r1i1p1',
'CanESM2-r2i1p1', 'CanESM2-r3i1p1', 'CanESM2-r4i1p1', 'CanESM2-r5i1p1',
'EC-EARTH-r12i1p1', 'EC-EARTH-r1i1p1', 'EC-EARTH-r2i1p1',
'EC-EARTH-r8i1p1', 'EC-EARTH-r9i1p1', 'FGOALS-g2-r1i1p1',
'GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1', 'GFDL-ESM2M-r1i1p1',
'GISS-E2-H-r1i1p1', 'GISS-E2-H-r1i1p2', 'GISS-E2-H-r1i1p3',
'GISS-E2-H-r2i1p1', 'GISS-E2-H-r2i1p3', 'GISS-E2-R-r1i1p1',
'GISS-E2-R-r1i1p2', 'GISS-E2-R-r1i1p3', 'GISS-E2-R-r2i1p1',
'GISS-E2-R-r2i1p3', 'HadGEM2-ES-r1i1p1', 'HadGEM2-ES-r2i1p1',
'HadGEM2-ES-r3i1p1', 'HadGEM2-ES-r4i1p1', 'IPSL-CM5A-LR-r1i1p1',
'IPSL-CM5A-LR-r2i1p1', 'IPSL-CM5A-LR-r3i1p1', 'IPSL-CM5A-LR-r4i1p1',
'IPSL-CM5A-MR-r1i1p1', 'IPSL-CM5B-LR-r1i1p1', 'MIROC-ESM-r1i1p1',
'MIROC5-r1i1p1', 'MIROC5-r2i1p1', 'MIROC5-r3i1p1', 'MPI-ESM-LR-r1i1p1',
'MPI-ESM-LR-r2i1p1', 'MPI-ESM-LR-r3i1p1', 'MPI-ESM-MR-r1i1p1',
'MRI-CGCM3-r1i1p1', 'NorESM1-ME-r1i1p1', 'NorESM1-M-r1i1p1',
'bcc-csm1-1-m-r1i1p1', 'bcc-csm1-1-r1i1p1', 'inmcm4-r1i1p1'])

dsPr5_clim_std = dsPr5_clim.std('member')

dsPr5_clim_can_std = dsPr5_clim.sel(member=['CanESM2-r1i1p1',
'CanESM2-r2i1p1', 'CanESM2-r3i1p1', 'CanESM2-r4i1p1', 'CanESM2-r5i1p1']).std('member')

dsPr5_clim_csiro_std = dsPr5_clim.sel(member=['CSIRO-Mk3-6-0-r10i1p1', 'CSIRO-Mk3-6-0-r1i1p1',
'CSIRO-Mk3-6-0-r2i1p1', 'CSIRO-Mk3-6-0-r3i1p1', 'CSIRO-Mk3-6-0-r4i1p1',
'CSIRO-Mk3-6-0-r5i1p1', 'CSIRO-Mk3-6-0-r6i1p1', 'CSIRO-Mk3-6-0-r7i1p1',
'CSIRO-Mk3-6-0-r8i1p1', 'CSIRO-Mk3-6-0-r9i1p1']).std('member')

dsPr5_clim_ccsm4_std = dsPr5_clim.sel(member=['CCSM4-r1i1p1', 'CCSM4-r2i1p1', 'CCSM4-r3i1p1', 'CCSM4-r4i1p1',
'CCSM4-r5i1p1', 'CCSM4-r6i1p1']).std('member')

dsPr5_clim_cnrm_std = dsPr5_clim.sel(member=['CNRM-CM5-r10i1p1',
'CNRM-CM5-r1i1p1', 'CNRM-CM5-r2i1p1', 'CNRM-CM5-r4i1p1',
'CNRM-CM5-r6i1p1']).std('member')

dsPr5_clim_earth_std = dsPr5_clim.sel(member=['EC-EARTH-r12i1p1', 'EC-EARTH-r1i1p1', 'EC-EARTH-r2i1p1',
'EC-EARTH-r8i1p1', 'EC-EARTH-r9i1p1']).std('member')

dsPr5_clim_iv_std = xr.concat([dsPr5_clim_can_std,dsPr5_clim_csiro_std,
dsPr5_clim_ccsm4_std,dsPr5_clim_cnrm_std,dsPr5_clim_earth_std],dim='member')

dsPr5_clim_iv_std_mean = dsPr5_clim_iv_std.median('member')

dsPr5_clim_one_std = dsPr5_clim.sel(member=['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1', 'BNU-ESM-r1i1p1',
'CCSM4-r1i1p1', 'CESM1-CAM5-r1i1p1','CNRM-CM5-r1i1p1', 'CSIRO-Mk3-6-0-r1i1p1',
'CanESM2-r1i1p1','EC-EARTH-r1i1p1', 'FGOALS-g2-r1i1p1',
'GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1', 'GFDL-ESM2M-r1i1p1',
'GISS-E2-H-r1i1p1', 'GISS-E2-R-r1i1p1','HadGEM2-ES-r1i1p1', 'IPSL-CM5A-LR-r1i1p1',
'IPSL-CM5A-MR-r1i1p1', 'IPSL-CM5B-LR-r1i1p1', 'MIROC-ESM-r1i1p1',
'MIROC5-r1i1p1', 'MPI-ESM-LR-r1i1p1', 'MPI-ESM-MR-r1i1p1',
'MRI-CGCM3-r1i1p1', 'NorESM1-ME-r1i1p1', 'NorESM1-M-r1i1p1',
'bcc-csm1-1-m-r1i1p1', 'bcc-csm1-1-r1i1p1', 'inmcm4-r1i1p1']).std('member') #

a = np.percentile(dsPr5_clim_one_std.pr,15)
n = np.percentile(dsPr5_clim_iv_std_mean.pr,85)

mask_obs = dsPr5_clim_one_std.where(dsPr5_clim_one_std>a)/dsPr5_clim_one_std.where(dsPr5_clim_one_std>a)

mask_obs_1 = mask_obs.sel(lon=np.arange(1.25,181.25,2.5))
mask_obs_2 = mask_obs.sel(lon=np.arange(181.25,360,2.5))
mask_obs_2['lon'] = mask_obs_2.lon - 360
mask_obs_flip = xr.concat([mask_obs_2,mask_obs_1],dim='lon')

mask_obs_Pr5 = np.nan_to_num(mask_obs_flip.pr)

mask_obs13 = dsPr5_clim_iv_std_mean.where(dsPr5_clim_iv_std_mean<n)/dsPr5_clim_iv_std_mean.where(dsPr5_clim_iv_std_mean<n)

mask_obs13_1 = mask_obs13.sel(lon=np.arange(1.25,181.25,2.5))
mask_obs13_2 = mask_obs13.sel(lon=np.arange(181.25,360,2.5))
mask_obs13_2['lon'] = mask_obs13_2.lon - 360
mask_obs13_flip = xr.concat([mask_obs13_2,mask_obs13_1],dim='lon')

mask_obs13_Pr5 = np.nan_to_num(mask_obs13_flip.pr)

dsPr5_clim_mask = mask_obs * mask_obs13 * dsPr5_clim.squeeze(drop=True)

###############################################################

fig3 = plt.figure(figsize=(8, 9))
gs = fig3.add_gridspec(3, 2)

f3_ax1 = fig3.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsPr6_clim_one_std.pr.plot.pcolormesh(ax=f3_ax1, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=7.5,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',
                                            'fraction':0.046, 'pad':0.04})
f3_ax1.contourf(mask_obs_flip.lon,mask_obs_flip.lat,mask_obs_Pr6,[-1,0,1],colors='none', hatches=['++', None],transform=ccrs.PlateCarree())
f3_ax1.coastlines();
f3_ax1.set_title('a) Annual PR Climatology, \n CMIP6 Inter-model $\sigma$ (mm/day)',fontsize=11)



f3_ax2 = fig3.add_subplot(gs[1, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsPr6_clim_iv_std_mean.pr.plot.pcolormesh(ax=f3_ax2, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=.75, #add_colorbar=False)
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '', #'ticks': [-2.5,0.0,2.5],
                                            'fraction':0.046, 'pad':0.04})
f3_ax2.contourf(mask_obs13_flip.lon,mask_obs13_flip.lat,mask_obs13_Pr6,[-1,0,1],colors='none', hatches=['xx', None],transform=ccrs.PlateCarree())
f3_ax2.coastlines();
f3_ax2.set_title('c) Annual PR Climatology, \n CMIP6 Median Internal $\sigma$ (mm/day)',fontsize=11)



f3_ax4 = fig3.add_subplot(gs[0, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsPr5_clim_one_std.pr.plot.pcolormesh(ax=f3_ax4, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=7.5,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',#'ticks': [-2.5,0.0,2.5],
                                            'fraction':0.046, 'pad':0.04})
f3_ax4.contourf(mask_obs_flip.lon,mask_obs_flip.lat,mask_obs_Pr5,[-1,0,1],colors='none', hatches=['++', None],transform=ccrs.PlateCarree())
f3_ax4.coastlines();
f3_ax4.set_title('b) Annual PR Climatology, \n CMIP5 Inter-model $\sigma$ (mm/day)',fontsize=11)


f3_ax5 = fig3.add_subplot(gs[1, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsPr5_clim_iv_std_mean.pr.plot.pcolormesh(ax=f3_ax5, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.75,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',
                                            'fraction':0.046, 'pad':0.04})
f3_ax5.contourf(mask_obs13_flip.lon,mask_obs13_flip.lat,mask_obs13_Pr5,[-1,0,1],colors='none', hatches=['xx', None],transform=ccrs.PlateCarree())
f3_ax5.coastlines();
f3_ax5.set_title('d) Annual PR Climatology, \n CMIP5 Median Internal $\sigma$ (mm/day)',fontsize=11)


f3_ax7 = fig3.add_subplot(gs[2, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('Blues')
dsPr6_clim_mask.mean('member').pr.plot.pcolormesh(ax=f3_ax7, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=10,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',#'ticks': [-2.5,0.0,2.5],
                                            'fraction':0.046, 'pad':0.04})
f3_ax7.contourf(mask_obs_flip.lon,mask_obs_flip.lat,mask_obs_Pr6,[-1,0,1],colors='none', hatches=['++', None],transform=ccrs.PlateCarree())
f3_ax7.contourf(mask_obs13_flip.lon,mask_obs13_flip.lat,mask_obs13_Pr6,[-1,0,1],colors='none', hatches=['xx', None],transform=ccrs.PlateCarree())
f3_ax7.coastlines();
f3_ax7.set_title('e) Annual PR Climatology, CMIP6 \n Ensemble Mean "Fingerprint" (mm/day)',fontsize=11)

f3_ax8 = fig3.add_subplot(gs[2, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('Blues')
dsPr5_clim_mask.mean('member').pr.plot.pcolormesh(ax=f3_ax8, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=10,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',
                                            'fraction':0.046, 'pad':0.04})
f3_ax8.contourf(mask_obs_flip.lon,mask_obs_flip.lat,mask_obs_Pr5,[-1,0,1],colors='none', hatches=['++', None],transform=ccrs.PlateCarree())
f3_ax8.contourf(mask_obs13_flip.lon,mask_obs13_flip.lat,mask_obs13_Pr5,[-1,0,1],colors='none', hatches=['xx', None],transform=ccrs.PlateCarree())
f3_ax8.coastlines();
f3_ax8.set_title('f) Annual PR Climatology, CMIP5 \n Ensemble Mean "Fingerprint" (mm/day)',fontsize=11)


plt.savefig('SupFig4_fingerprint_precip_1905_2005.png',bbox_inches='tight',dpi=300)
