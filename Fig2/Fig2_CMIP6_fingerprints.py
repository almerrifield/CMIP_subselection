# Figure 2: makes CMIP6 "fingerprint" hatched map plots

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

# Final Version, 15, 15 medians

## Current Set

dirT6 = '/**/CMIP_subselection/Data/'
dsT6 = xr.open_dataset(dirT6 + 'tas_mon_CMIP6_SSP585_g025_v2_ann.nc',use_cftime = True)
dsT6 = dsT6-273.15
dsT6 = dsT6.sortby(dsT6.member)

dsP6 = xr.open_dataset(dirT6 + 'psl_mon_CMIP6_SSP585_g025_v2_ann.nc',use_cftime = True)
dsP6 = dsP6/100
dsP6 = dsP6.sortby(dsP6.member)

dsT6_clim = dsT6.sel(year=slice(1905, 2005)).mean('year')
dsP6_clim = dsP6.sel(year=slice(1905, 2005)).mean('year')

dsT6_clim = dsT6_clim.drop_sel(member=['NorESM2-LM-r1i1p1f1','FIO-ESM-2-0-r1i1p1f1','FIO-ESM-2-0-r2i1p1f1', 'FIO-ESM-2-0-r3i1p1f1'])#,'MCM-UA-1-0-r1i1p1f2'])
dsP6_clim = dsP6_clim.drop_sel(member=['BCC-CSM2-MR-r1i1p1f1','CIESM-r1i1p1f1','FIO-ESM-2-0-r1i1p1f1','FIO-ESM-2-0-r2i1p1f1', 'FIO-ESM-2-0-r3i1p1f1','EC-Earth3-Veg-r6i1p1f1'])#,'MCM-UA-1-0-r1i1p1f2'])

dsT6_clim_std = dsT6_clim.std('member')

dsT6_clim_miroc_std = dsT6_clim.sel(member=['MIROC6-r10i1p1f1', 'MIROC6-r11i1p1f1',
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

dsT6_clim_can_std = dsT6_clim.sel(member=['CanESM5-r10i1p1f1', 'CanESM5-r11i1p1f1',
'CanESM5-r12i1p1f1','CanESM5-r13i1p1f1', 'CanESM5-r14i1p1f1',
'CanESM5-r15i1p1f1', 'CanESM5-r16i1p1f1', 'CanESM5-r17i1p1f1',
'CanESM5-r18i1p1f1', 'CanESM5-r19i1p1f1', 'CanESM5-r1i1p1f1',
'CanESM5-r20i1p1f1','CanESM5-r21i1p1f1', 'CanESM5-r22i1p1f1',
'CanESM5-r23i1p1f1', 'CanESM5-r24i1p1f1', 'CanESM5-r25i1p1f1',
'CanESM5-r2i1p1f1', 'CanESM5-r3i1p1f1', 'CanESM5-r4i1p1f1',
'CanESM5-r5i1p1f1', 'CanESM5-r6i1p1f1', 'CanESM5-r7i1p1f1',
'CanESM5-r8i1p1f1', 'CanESM5-r9i1p1f1']).std('member')

dsT6_clim_access_std = dsT6_clim.sel(member=['ACCESS-ESM1-5-r10i1p1f1',
'ACCESS-ESM1-5-r1i1p1f1', 'ACCESS-ESM1-5-r2i1p1f1',
'ACCESS-ESM1-5-r3i1p1f1', 'ACCESS-ESM1-5-r4i1p1f1',
'ACCESS-ESM1-5-r5i1p1f1', 'ACCESS-ESM1-5-r6i1p1f1',
'ACCESS-ESM1-5-r7i1p1f1', 'ACCESS-ESM1-5-r8i1p1f1',
'ACCESS-ESM1-5-r9i1p1f1']).std('member')

dsT6_clim_cesm_std = dsT6_clim.sel(member=['CESM2-r10i1p1f1', 'CESM2-r11i1p1f1',
'CESM2-r1i1p1f1', 'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1']).std('member')

dsT6_clim_cnrm_std = dsT6_clim.sel(member=['CNRM-CM6-1-r1i1p1f2',
'CNRM-CM6-1-r2i1p1f2', 'CNRM-CM6-1-r3i1p1f2',
'CNRM-CM6-1-r4i1p1f2', 'CNRM-CM6-1-r5i1p1f2',
'CNRM-CM6-1-r6i1p1f2']).std('member')

dsT6_clim_earth_std = dsT6_clim.sel(member=['EC-Earth3-r11i1p1f1', 'EC-Earth3-r13i1p1f1',
'EC-Earth3-r15i1p1f1', 'EC-Earth3-r1i1p1f1', 'EC-Earth3-r3i1p1f1',
'EC-Earth3-r4i1p1f1', 'EC-Earth3-r6i1p1f1', 'EC-Earth3-r9i1p1f1']).std('member')

dsT6_clim_giss_std = dsT6_clim.sel(member=['GISS-E2-1-G-r1i1p3f1',
'GISS-E2-1-G-r2i1p3f1','GISS-E2-1-G-r3i1p3f1', 'GISS-E2-1-G-r4i1p3f1',
'GISS-E2-1-G-r5i1p3f1']).std('member')

dsT6_clim_ipsl_std = dsT6_clim.sel(member=['IPSL-CM6A-LR-r14i1p1f1',
'IPSL-CM6A-LR-r1i1p1f1', 'IPSL-CM6A-LR-r2i1p1f1',
'IPSL-CM6A-LR-r3i1p1f1', 'IPSL-CM6A-LR-r4i1p1f1',
'IPSL-CM6A-LR-r6i1p1f1']).std('member')

dsT6_clim_mpi_std = dsT6_clim.sel(member=['MPI-ESM1-2-LR-r10i1p1f1', 'MPI-ESM1-2-LR-r1i1p1f1',
'MPI-ESM1-2-LR-r2i1p1f1', 'MPI-ESM1-2-LR-r3i1p1f1',
'MPI-ESM1-2-LR-r4i1p1f1', 'MPI-ESM1-2-LR-r5i1p1f1',
'MPI-ESM1-2-LR-r6i1p1f1', 'MPI-ESM1-2-LR-r7i1p1f1',
'MPI-ESM1-2-LR-r8i1p1f1', 'MPI-ESM1-2-LR-r9i1p1f1']).std('member')

dsT6_clim_ukesm_std = dsT6_clim.sel(member=['UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2','UKESM1-0-LL-r8i1p1f2']).std('member')

dsT6_clim_miroc_e_std = dsT6_clim.sel(member=['MIROC-ES2L-r10i1p1f2',
'MIROC-ES2L-r1i1p1f2', 'MIROC-ES2L-r2i1p1f2',
'MIROC-ES2L-r3i1p1f2', 'MIROC-ES2L-r4i1p1f2',
'MIROC-ES2L-r5i1p1f2', 'MIROC-ES2L-r6i1p1f2',
'MIROC-ES2L-r7i1p1f2', 'MIROC-ES2L-r8i1p1f2',
'MIROC-ES2L-r9i1p1f2']).std('member')

dsT6_clim_cnrm_e_std = dsT6_clim.sel(member=['CNRM-ESM2-1-r1i1p1f2',
'CNRM-ESM2-1-r2i1p1f2', 'CNRM-ESM2-1-r3i1p1f2',
'CNRM-ESM2-1-r4i1p1f2', 'CNRM-ESM2-1-r5i1p1f2']).std('member')

dsT6_clim_iv_std = xr.concat([dsT6_clim_miroc_std,dsT6_clim_can_std,dsT6_clim_access_std,
dsT6_clim_cesm_std,dsT6_clim_cnrm_std,dsT6_clim_cnrm_e_std,dsT6_clim_earth_std,
dsT6_clim_giss_std,dsT6_clim_ipsl_std,dsT6_clim_mpi_std,dsT6_clim_miroc_e_std,
dsT6_clim_ukesm_std],dim='member')

dsT6_clim_iv_std_mean = dsT6_clim_iv_std.median('member')

dsT6_clim_one_std = dsT6_clim.sel(member=['ACCESS-CM2-r1i1p1f1',
'ACCESS-ESM1-5-r1i1p1f1', 'AWI-CM-1-1-MR-r1i1p1f1','CAS-ESM2-0-r1i1p1f1',
'CESM2-WACCM-r1i1p1f1', 'CESM2-r1i1p1f1', 'CMCC-CM2-SR5-r1i1p1f1', 'CMCC-ESM2-r1i1p1f1',
'CNRM-CM6-1-HR-r1i1p1f2', 'CNRM-CM6-1-r1i1p1f2','CNRM-ESM2-1-r1i1p1f2','CanESM5-r1i1p1f1',
'E3SM-1-1-r1i1p1f1','EC-Earth3-Veg-r1i1p1f1',
'EC-Earth3-r1i1p1f1','FGOALS-f3-L-r1i1p1f1', 'FGOALS-g3-r1i1p1f1',
'GFDL-CM4-r1i1p1f1','GFDL-ESM4-r1i1p1f1', 'GISS-E2-1-G-r1i1p3f1',
'HadGEM3-GC31-LL-r1i1p1f3', 'HadGEM3-GC31-MM-r1i1p1f3',
'INM-CM4-8-r1i1p1f1','INM-CM5-0-r1i1p1f1','IPSL-CM6A-LR-r1i1p1f1',
'KACE-1-0-G-r1i1p1f1','KIOST-ESM-r1i1p1f1','MCM-UA-1-0-r1i1p1f2', 'MIROC-ES2L-r1i1p1f2',
'MIROC6-r1i1p1f1', 'MPI-ESM1-2-HR-r1i1p1f1','MPI-ESM1-2-LR-r1i1p1f1',
'MRI-ESM2-0-r1i1p1f1','NESM3-r1i1p1f1','NorESM2-MM-r1i1p1f1',
'TaiESM1-r1i1p1f1', 'UKESM1-0-LL-r1i1p1f2']).std('member') #

######
a = np.percentile(dsT6_clim_one_std.tas,15)
n = np.percentile(dsT6_clim_iv_std_mean.tas,85)

mask_obs = dsT6_clim_one_std.where(dsT6_clim_one_std>a)/dsT6_clim_one_std.where(dsT6_clim_one_std>a)

mask_obs_1 = mask_obs.sel(lon=np.arange(1.25,181.25,2.5))
mask_obs_2 = mask_obs.sel(lon=np.arange(181.25,360,2.5))
mask_obs_2['lon'] = mask_obs_2.lon - 360
mask_obs_flip = xr.concat([mask_obs_2,mask_obs_1],dim='lon')

mask_obs_T6 = np.nan_to_num(mask_obs_flip.tas)

mask_obs13 = dsT6_clim_iv_std_mean.where(dsT6_clim_iv_std_mean<n)/dsT6_clim_iv_std_mean.where(dsT6_clim_iv_std_mean<n)

mask_obs13_1 = mask_obs13.sel(lon=np.arange(1.25,181.25,2.5))
mask_obs13_2 = mask_obs13.sel(lon=np.arange(181.25,360,2.5))
mask_obs13_2['lon'] = mask_obs13_2.lon - 360
mask_obs13_flip = xr.concat([mask_obs13_2,mask_obs13_1],dim='lon')

mask_obs13_T6 = np.nan_to_num(mask_obs13_flip.tas)

dsT6_clim_mask = mask_obs * mask_obs13 * dsT6_clim.squeeze(drop=True)

#####################################################################

dsP6_clim_std = dsP6_clim.std('member')

dsP6_clim_miroc_std = dsP6_clim.sel(member=['MIROC6-r10i1p1f1', 'MIROC6-r11i1p1f1',
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

dsP6_clim_can_std = dsP6_clim.sel(member=['CanESM5-r10i1p1f1', 'CanESM5-r11i1p1f1',
'CanESM5-r12i1p1f1','CanESM5-r13i1p1f1', 'CanESM5-r14i1p1f1',
'CanESM5-r15i1p1f1', 'CanESM5-r16i1p1f1', 'CanESM5-r17i1p1f1',
'CanESM5-r18i1p1f1', 'CanESM5-r19i1p1f1', 'CanESM5-r1i1p1f1',
'CanESM5-r20i1p1f1','CanESM5-r21i1p1f1', 'CanESM5-r22i1p1f1',
'CanESM5-r23i1p1f1', 'CanESM5-r24i1p1f1', 'CanESM5-r25i1p1f1',
'CanESM5-r2i1p1f1', 'CanESM5-r3i1p1f1', 'CanESM5-r4i1p1f1',
'CanESM5-r5i1p1f1', 'CanESM5-r6i1p1f1', 'CanESM5-r7i1p1f1',
'CanESM5-r8i1p1f1', 'CanESM5-r9i1p1f1']).std('member')

dsP6_clim_access_std = dsP6_clim.sel(member=['ACCESS-ESM1-5-r10i1p1f1',
'ACCESS-ESM1-5-r1i1p1f1', 'ACCESS-ESM1-5-r2i1p1f1',
'ACCESS-ESM1-5-r3i1p1f1', 'ACCESS-ESM1-5-r4i1p1f1',
'ACCESS-ESM1-5-r5i1p1f1', 'ACCESS-ESM1-5-r6i1p1f1',
'ACCESS-ESM1-5-r7i1p1f1', 'ACCESS-ESM1-5-r8i1p1f1',
'ACCESS-ESM1-5-r9i1p1f1']).std('member')

dsP6_clim_cesm_std = dsP6_clim.sel(member=['CESM2-r10i1p1f1', 'CESM2-r11i1p1f1',
'CESM2-r1i1p1f1', 'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1']).std('member')

dsP6_clim_cnrm_std = dsP6_clim.sel(member=['CNRM-CM6-1-r1i1p1f2',
'CNRM-CM6-1-r2i1p1f2', 'CNRM-CM6-1-r3i1p1f2',
'CNRM-CM6-1-r4i1p1f2', 'CNRM-CM6-1-r5i1p1f2',
'CNRM-CM6-1-r6i1p1f2']).std('member')

dsP6_clim_earth_std = dsP6_clim.sel(member=['EC-Earth3-r11i1p1f1', 'EC-Earth3-r13i1p1f1',
'EC-Earth3-r15i1p1f1', 'EC-Earth3-r1i1p1f1', 'EC-Earth3-r3i1p1f1',
'EC-Earth3-r4i1p1f1', 'EC-Earth3-r6i1p1f1', 'EC-Earth3-r9i1p1f1']).std('member')

dsP6_clim_giss_std = dsP6_clim.sel(member=['GISS-E2-1-G-r1i1p3f1',
'GISS-E2-1-G-r2i1p3f1','GISS-E2-1-G-r3i1p3f1', 'GISS-E2-1-G-r4i1p3f1',
'GISS-E2-1-G-r5i1p3f1']).std('member')

dsP6_clim_ipsl_std = dsP6_clim.sel(member=['IPSL-CM6A-LR-r14i1p1f1',
'IPSL-CM6A-LR-r1i1p1f1', 'IPSL-CM6A-LR-r2i1p1f1',
'IPSL-CM6A-LR-r3i1p1f1', 'IPSL-CM6A-LR-r4i1p1f1',
'IPSL-CM6A-LR-r6i1p1f1']).std('member')

dsP6_clim_mpi_std = dsP6_clim.sel(member=['MPI-ESM1-2-LR-r10i1p1f1', 'MPI-ESM1-2-LR-r1i1p1f1',
'MPI-ESM1-2-LR-r2i1p1f1', 'MPI-ESM1-2-LR-r3i1p1f1',
'MPI-ESM1-2-LR-r4i1p1f1', 'MPI-ESM1-2-LR-r5i1p1f1',
'MPI-ESM1-2-LR-r6i1p1f1', 'MPI-ESM1-2-LR-r7i1p1f1',
'MPI-ESM1-2-LR-r8i1p1f1', 'MPI-ESM1-2-LR-r9i1p1f1']).std('member')

dsP6_clim_ukesm_std = dsP6_clim.sel(member=['UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2','UKESM1-0-LL-r8i1p1f2']).std('member')


dsP6_clim_miroc_e_std = dsP6_clim.sel(member=['MIROC-ES2L-r10i1p1f2',
'MIROC-ES2L-r1i1p1f2', 'MIROC-ES2L-r2i1p1f2',
'MIROC-ES2L-r3i1p1f2', 'MIROC-ES2L-r4i1p1f2',
'MIROC-ES2L-r5i1p1f2', 'MIROC-ES2L-r6i1p1f2',
'MIROC-ES2L-r7i1p1f2', 'MIROC-ES2L-r8i1p1f2',
'MIROC-ES2L-r9i1p1f2']).std('member')

dsP6_clim_cnrm_e_std = dsP6_clim.sel(member=['CNRM-ESM2-1-r1i1p1f2',
'CNRM-ESM2-1-r2i1p1f2', 'CNRM-ESM2-1-r3i1p1f2',
'CNRM-ESM2-1-r4i1p1f2', 'CNRM-ESM2-1-r5i1p1f2']).std('member')

dsP6_clim_iv_std = xr.concat([dsP6_clim_miroc_std,dsP6_clim_can_std,dsP6_clim_access_std,
dsP6_clim_cesm_std,dsP6_clim_cnrm_std,dsP6_clim_cnrm_e_std,dsP6_clim_earth_std,
dsP6_clim_giss_std,dsP6_clim_ipsl_std,dsP6_clim_mpi_std,dsP6_clim_miroc_e_std,
dsP6_clim_ukesm_std],dim='member')

dsP6_clim_iv_std_mean = dsP6_clim_iv_std.median('member')


dsP6_clim_one_std = dsP6_clim.sel(member=['ACCESS-CM2-r1i1p1f1',
'ACCESS-ESM1-5-r1i1p1f1', 'AWI-CM-1-1-MR-r1i1p1f1','CAS-ESM2-0-r1i1p1f1',
'CESM2-WACCM-r1i1p1f1', 'CESM2-r1i1p1f1', 'CMCC-CM2-SR5-r1i1p1f1', 'CMCC-ESM2-r1i1p1f1',
'CNRM-CM6-1-HR-r1i1p1f2', 'CNRM-CM6-1-r1i1p1f2','CNRM-ESM2-1-r1i1p1f2','CanESM5-r1i1p1f1',
'E3SM-1-1-r1i1p1f1','EC-Earth3-Veg-r1i1p1f1',
'EC-Earth3-r1i1p1f1','FGOALS-f3-L-r1i1p1f1', 'FGOALS-g3-r1i1p1f1',
'GFDL-CM4-r1i1p1f1','GFDL-ESM4-r1i1p1f1', 'GISS-E2-1-G-r1i1p3f1',
'HadGEM3-GC31-LL-r1i1p1f3', 'HadGEM3-GC31-MM-r1i1p1f3',
'INM-CM4-8-r1i1p1f1','INM-CM5-0-r1i1p1f1','IPSL-CM6A-LR-r1i1p1f1',
'KACE-1-0-G-r1i1p1f1','KIOST-ESM-r1i1p1f1', 'MCM-UA-1-0-r1i1p1f2','MIROC-ES2L-r1i1p1f2',
'MIROC6-r1i1p1f1', 'MPI-ESM1-2-HR-r1i1p1f1','MPI-ESM1-2-LR-r1i1p1f1',
'MRI-ESM2-0-r1i1p1f1','NESM3-r1i1p1f1','NorESM2-MM-r1i1p1f1',
'TaiESM1-r1i1p1f1', 'UKESM1-0-LL-r1i1p1f2']).std('member') #


#####

a = np.percentile(dsP6_clim_one_std.psl,15)
n = np.percentile(dsP6_clim_iv_std_mean.psl,85)


mask_obs = dsP6_clim_one_std.where(dsP6_clim_one_std>a)/dsP6_clim_one_std.where(dsP6_clim_one_std>a)

mask_obs_1 = mask_obs.sel(lon=np.arange(1.25,181.25,2.5))
mask_obs_2 = mask_obs.sel(lon=np.arange(181.25,360,2.5))
mask_obs_2['lon'] = mask_obs_2.lon - 360
mask_obs_flip = xr.concat([mask_obs_2,mask_obs_1],dim='lon')

mask_obs_P6 = np.nan_to_num(mask_obs_flip.psl)

mask_obs13 = dsP6_clim_iv_std_mean.where(dsP6_clim_iv_std_mean<n)/dsP6_clim_iv_std_mean.where(dsP6_clim_iv_std_mean<n)

mask_obs13_1 = mask_obs13.sel(lon=np.arange(1.25,181.25,2.5))
mask_obs13_2 = mask_obs13.sel(lon=np.arange(181.25,360,2.5))
mask_obs13_2['lon'] = mask_obs13_2.lon - 360
mask_obs13_flip = xr.concat([mask_obs13_2,mask_obs13_1],dim='lon')

mask_obs13_P6 = np.nan_to_num(mask_obs13_flip.psl)

dsP6_clim_mask = mask_obs * mask_obs13 * dsP6_clim.squeeze(drop=True)



#############################
fig3 = plt.figure(figsize=(8, 9))
gs = fig3.add_gridspec(3, 2)

# Annual Global Land SAT climatology; 1905-2005
f3_ax1 = fig3.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsT6_clim_one_std.tas.plot.pcolormesh(ax=f3_ax1, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=7.5,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',
                                            'fraction':0.046, 'pad':0.04})
f3_ax1.contourf(mask_obs_flip.lon,mask_obs_flip.lat,mask_obs_T6,[-1,0,1],colors='none', hatches=['++', None],transform=ccrs.PlateCarree())
f3_ax1.coastlines();
f3_ax1.set_title('a) Annual SAT Climatology, \n Inter-model $\sigma$ (˚C)',fontsize=11)

f3_ax2 = fig3.add_subplot(gs[1, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsT6_clim_iv_std_mean.tas.plot.pcolormesh(ax=f3_ax2, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=.75, #add_colorbar=False)
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '', #'ticks': [-2.5,0.0,2.5],
                                            'fraction':0.046, 'pad':0.04})
f3_ax2.contourf(mask_obs13_flip.lon,mask_obs13_flip.lat,mask_obs13_T6,[-1,0,1],colors='none', hatches=['xx', None],transform=ccrs.PlateCarree())
f3_ax2.coastlines();
f3_ax2.set_title('c) Annual SAT Climatology, \n Median Internal $\sigma$ (˚C)',fontsize=11)

# Annual NH SLP climatology; 1905-2005
f3_ax4 = fig3.add_subplot(gs[0, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_one_std.psl.plot.pcolormesh(ax=f3_ax4, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=7.5,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',#'ticks': [-2.5,0.0,2.5],
                                            'fraction':0.046, 'pad':0.04})
f3_ax4.contourf(mask_obs_flip.lon,mask_obs_flip.lat,mask_obs_P6,[-1,0,1],colors='none', hatches=['++', None],transform=ccrs.PlateCarree())
f3_ax4.coastlines();
f3_ax4.set_title('b) Annual SLP Climatology, \n Inter-model $\sigma$ (hPa)',fontsize=11)

f3_ax5 = fig3.add_subplot(gs[1, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_iv_std_mean.psl.plot.pcolormesh(ax=f3_ax5, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.75,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',
                                            'fraction':0.046, 'pad':0.04})
f3_ax5.contourf(mask_obs13_flip.lon,mask_obs13_flip.lat,mask_obs13_P6,[-1,0,1],colors='none', hatches=['xx', None],transform=ccrs.PlateCarree())
f3_ax5.coastlines();
f3_ax5.set_title('d) Annual SLP Climatology, \n Median Internal $\sigma$ (hPa)',fontsize=11)

# Annual Global Land SAT climatology; 1905-2005
f3_ax7 = fig3.add_subplot(gs[2, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('RdBu_r')
dsT6_clim_mask.mean('member').tas.plot.pcolormesh(ax=f3_ax7, transform=ccrs.PlateCarree(), cmap=cmap,vmin=-40,vmax=40,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',#'ticks': [-2.5,0.0,2.5],
                                            'fraction':0.046, 'pad':0.04})
f3_ax7.contourf(mask_obs_flip.lon,mask_obs_flip.lat,mask_obs_T6,[-1,0,1],colors='none', hatches=['++', None],transform=ccrs.PlateCarree())
f3_ax7.contourf(mask_obs13_flip.lon,mask_obs13_flip.lat,mask_obs13_T6,[-1,0,1],colors='none', hatches=['xx', None],transform=ccrs.PlateCarree())
f3_ax7.coastlines();
f3_ax7.set_title('e) Annual SAT Climatology, \n Ensemble Mean "Fingerprint" (˚C)',fontsize=11)

# Annual NH SLP climatology; 1905-2005
f3_ax8 = fig3.add_subplot(gs[2, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('RdBu_r')
dsP6_clim_mask.mean('member').psl.plot.pcolormesh(ax=f3_ax8, transform=ccrs.PlateCarree(), cmap=cmap,vmin=980,vmax=1030,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',
                                            'fraction':0.046, 'pad':0.04})
f3_ax8.contourf(mask_obs_flip.lon,mask_obs_flip.lat,mask_obs_P6,[-1,0,1],colors='none', hatches=['++', None],transform=ccrs.PlateCarree())
f3_ax8.contourf(mask_obs13_flip.lon,mask_obs13_flip.lat,mask_obs13_P6,[-1,0,1],colors='none', hatches=['xx', None],transform=ccrs.PlateCarree())
f3_ax8.coastlines();
f3_ax8.set_title('f) Annual SLP Climatology, \n Ensemble Mean "Fingerprint" (hPa)',fontsize=11)

plt.savefig('Fig2_CMIP6_fingerprints_1905_2005.png',bbox_inches='tight',dpi=300)
