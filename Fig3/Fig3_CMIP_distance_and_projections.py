# Figure 3: makes tick independence and MDS projection panels, checks triangle inequality

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
dsT6 = xr.open_dataset(dirT6 + 'tas_mon_CMIP6_SSP585_g025_v2_ann.nc',use_cftime = True)
dsT6 = dsT6-273.15
dsT6 = dsT6.sortby(dsT6.member)

dsP6 = xr.open_dataset(dirT6 + 'psl_mon_CMIP6_SSP585_g025_v2_ann.nc',use_cftime = True)
dsP6 = dsP6/100
dsP6 = dsP6.sortby(dsP6.member)

dsT6_clim = dsT6.sel(year=slice(1905, 2005)).mean('year') ### 1950
dsP6_clim = dsP6.sel(year=slice(1905, 2005)).mean('year') ### 1950

dsT6_clim = dsT6_clim.drop_sel(member=['NorESM2-LM-r1i1p1f1','FIO-ESM-2-0-r1i1p1f1','FIO-ESM-2-0-r2i1p1f1', 'FIO-ESM-2-0-r3i1p1f1'])#,'MCM-UA-1-0-r1i1p1f2'])
dsP6_clim = dsP6_clim.drop_sel(member=['BCC-CSM2-MR-r1i1p1f1','CIESM-r1i1p1f1','FIO-ESM-2-0-r1i1p1f1','FIO-ESM-2-0-r2i1p1f1', 'FIO-ESM-2-0-r3i1p1f1','EC-Earth3-Veg-r6i1p1f1'])#,'MCM-UA-1-0-r1i1p1f2'])

###############
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

###########

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

#################

# Compute Independence Matrix

weights = [np.cos(np.deg2rad(dsT6_clim.lat))]*144
weights = xr.concat(weights, "lon")
weights['lon'] = dsT6_clim.lon

def get_error(ds,weights):
    mod_coords = ds.member.values
    nmod = len(mod_coords)
    res = xr.DataArray(np.empty(shape=(nmod, nmod)),
                        dims=("member1", "member2"), coords=dict(member1=mod_coords, member2=mod_coords))

    for mod1 in ds.transpose("member", ...):
        for mod2 in ds.transpose("member", ...):
            a = xskillscore.rmse(mod1,mod2,dim=['lat','lon'],weights=weights,skipna=True)
            res.loc[dict(member1=mod1.member, member2=mod2.member)] = a

    return res

dsT6_clim_test = get_error(dsT6_clim_mask.tas,weights)
dsT6_clim_test = dsT6_clim_test.where(dsT6_clim_test!=0)
dsT6_clim_test_norm = dsT6_clim_test/np.nanmean(dsT6_clim_test)

dsP6_clim_test = get_error(dsP6_clim_mask.psl,weights)
dsP6_clim_test = dsP6_clim_test.where(dsP6_clim_test!=0)
dsP6_clim_test_norm = dsP6_clim_test/np.nanmean(dsP6_clim_test)

dsWi = (dsT6_clim_test_norm + dsP6_clim_test_norm)/2 ###############

####### Check if distance metric satisfies the triangle inequality

import itertools

ind = itertools.combinations(list(range(len(dsWi))), 3)

for a,b,c in ind:
    dist_ab = dsWi.isel(member1=a,member2=b).data
    dist_ac = dsWi.isel(member1=a,member2=c).data
    dist_bc = dsWi.isel(member1=b,member2=c).data
    if dist_ab > dist_ac+dist_bc:
        raise RuntimeError('Triangular Criterion Failed')

####################################

access_fam = ['ACCESS-ESM1-5-r10i1p1f1',
'ACCESS-ESM1-5-r1i1p1f1', 'ACCESS-ESM1-5-r2i1p1f1',
'ACCESS-ESM1-5-r3i1p1f1', 'ACCESS-ESM1-5-r4i1p1f1',
'ACCESS-ESM1-5-r5i1p1f1', 'ACCESS-ESM1-5-r6i1p1f1',
'ACCESS-ESM1-5-r7i1p1f1', 'ACCESS-ESM1-5-r8i1p1f1',
'ACCESS-ESM1-5-r9i1p1f1']
# closest relative: 'CMCC-CM2-SR5-r1i1p1f1'

dsWi_access_fam = dsWi.sel(member1=access_fam,member2=access_fam)
access_fam_all = np.unique(dsWi_access_fam)
access_fam_all = access_fam_all[~np.isnan(access_fam_all)]

dsWi_access_rest = dsWi.sel(member1=access_fam).drop_sel(member2=access_fam)
access_rest_all = np.unique(dsWi_access_rest)
access_rest_all = access_rest_all[~np.isnan(access_rest_all)]

#####################################

can_fam = ['CanESM5-r10i1p1f1', 'CanESM5-r11i1p1f1',
'CanESM5-r12i1p1f1','CanESM5-r13i1p1f1', 'CanESM5-r14i1p1f1',
'CanESM5-r15i1p1f1', 'CanESM5-r16i1p1f1', 'CanESM5-r17i1p1f1',
'CanESM5-r18i1p1f1', 'CanESM5-r19i1p1f1', 'CanESM5-r1i1p1f1',
'CanESM5-r20i1p1f1','CanESM5-r21i1p1f1', 'CanESM5-r22i1p1f1',
'CanESM5-r23i1p1f1', 'CanESM5-r24i1p1f1', 'CanESM5-r25i1p1f1',
'CanESM5-r2i1p1f1', 'CanESM5-r3i1p1f1', 'CanESM5-r4i1p1f1',
'CanESM5-r5i1p1f1', 'CanESM5-r6i1p1f1', 'CanESM5-r7i1p1f1',
'CanESM5-r8i1p1f1', 'CanESM5-r9i1p1f1',
'CanESM5-r10i1p2f1', 'CanESM5-r11i1p2f1',
'CanESM5-r12i1p2f1','CanESM5-r13i1p2f1', 'CanESM5-r14i1p2f1',
'CanESM5-r15i1p2f1', 'CanESM5-r16i1p2f1', 'CanESM5-r17i1p2f1',
'CanESM5-r18i1p2f1', 'CanESM5-r19i1p2f1', 'CanESM5-r1i1p2f1',
'CanESM5-r20i1p2f1','CanESM5-r21i1p2f1', 'CanESM5-r22i1p2f1',
'CanESM5-r23i1p2f1', 'CanESM5-r24i1p2f1', 'CanESM5-r25i1p2f1',
'CanESM5-r2i1p2f1', 'CanESM5-r3i1p2f1', 'CanESM5-r4i1p2f1',
'CanESM5-r5i1p2f1', 'CanESM5-r6i1p2f1', 'CanESM5-r7i1p2f1',
'CanESM5-r8i1p2f1', 'CanESM5-r9i1p2f1']

dsWi_can_fam = dsWi.sel(member1=can_fam,member2=can_fam)
can_fam_all = np.unique(dsWi_can_fam)
can_fam_all = can_fam_all[~np.isnan(can_fam_all)]

dsWi_can_rest = dsWi.sel(member1=can_fam).drop_sel(member2=can_fam)
can_rest_all = np.unique(dsWi_can_rest)
can_rest_all = can_rest_all[~np.isnan(can_rest_all)]


#####################################

miroc_fam = ['MIROC6-r10i1p1f1', 'MIROC6-r11i1p1f1',
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
'MIROC6-r7i1p1f1', 'MIROC6-r8i1p1f1', 'MIROC6-r9i1p1f1']
# closest relative: MIROC-ES2L-r7i1p1f2'

dsWi_miroc_fam = dsWi.sel(member1=miroc_fam,member2=miroc_fam)
miroc_fam_all = np.unique(dsWi_miroc_fam)
miroc_fam_all = miroc_fam_all[~np.isnan(miroc_fam_all)]

dsWi_miroc_rest = dsWi.sel(member1=miroc_fam).drop_sel(member2=miroc_fam)
miroc_rest_all = np.unique(dsWi_miroc_rest)
miroc_rest_all = miroc_rest_all[~np.isnan(miroc_rest_all)]

#####################################

miroc_esl_fam = ['MIROC-ES2L-r10i1p1f2',
'MIROC-ES2L-r1i1p1f2', 'MIROC-ES2L-r2i1p1f2',
'MIROC-ES2L-r3i1p1f2', 'MIROC-ES2L-r4i1p1f2',
'MIROC-ES2L-r5i1p1f2', 'MIROC-ES2L-r6i1p1f2',
'MIROC-ES2L-r7i1p1f2', 'MIROC-ES2L-r8i1p1f2',
'MIROC-ES2L-r9i1p1f2']
# closest relative: 'MIROC6-r26i1p1f1'

dsWi_miroc_esl_fam = dsWi.sel(member1=miroc_esl_fam,member2=miroc_esl_fam)
miroc_esl_fam_all = np.unique(dsWi_miroc_esl_fam)
miroc_esl_fam_all = miroc_esl_fam_all[~np.isnan(miroc_esl_fam_all)]

dsWi_miroc_esl_rest = dsWi.sel(member1=miroc_esl_fam).drop_sel(member2=miroc_esl_fam)
miroc_esl_rest_all = np.unique(dsWi_miroc_esl_rest)
miroc_esl_rest_all = miroc_esl_rest_all[~np.isnan(miroc_esl_rest_all)]

#####################################

mpi_hr_fam = ['MPI-ESM1-2-HR-r1i1p1f1','MPI-ESM1-2-HR-r2i1p1f1']

dsWi_mpi_hr_fam = dsWi.sel(member1=mpi_hr_fam,member2=mpi_hr_fam)
mpi_hr_fam_all = np.unique(dsWi_mpi_hr_fam)
mpi_hr_fam_all = mpi_hr_fam_all[~np.isnan(mpi_hr_fam_all)]

dsWi_mpi_hr_rest = dsWi.sel(member1=mpi_hr_fam).drop_sel(member2=mpi_hr_fam)
mpi_hr_rest_all = np.unique(dsWi_mpi_hr_rest)
mpi_hr_rest_all = mpi_hr_rest_all[~np.isnan(mpi_hr_rest_all)]

mpi_full_fam = ['MPI-ESM1-2-LR-r10i1p1f1', 'MPI-ESM1-2-LR-r1i1p1f1',
'MPI-ESM1-2-LR-r2i1p1f1', 'MPI-ESM1-2-LR-r3i1p1f1',
'MPI-ESM1-2-LR-r4i1p1f1', 'MPI-ESM1-2-LR-r5i1p1f1',
'MPI-ESM1-2-LR-r6i1p1f1', 'MPI-ESM1-2-LR-r7i1p1f1',
'MPI-ESM1-2-LR-r8i1p1f1', 'MPI-ESM1-2-LR-r9i1p1f1',
'NESM3-r1i1p1f1','NESM3-r2i1p1f1','AWI-CM-1-1-MR-r1i1p1f1']

dsWi_mpi_hr_fam_rest = dsWi.sel(member1=mpi_hr_fam,member2=mpi_full_fam)
mpi_hr_fam_rest_all = np.unique(dsWi_mpi_hr_fam_rest)
mpi_hr_fam_rest_all = mpi_hr_fam_rest_all[~np.isnan(mpi_hr_fam_rest_all)]

########

mpi_fam = ['MPI-ESM1-2-LR-r10i1p1f1', 'MPI-ESM1-2-LR-r1i1p1f1',
'MPI-ESM1-2-LR-r2i1p1f1', 'MPI-ESM1-2-LR-r3i1p1f1',
'MPI-ESM1-2-LR-r4i1p1f1', 'MPI-ESM1-2-LR-r5i1p1f1',
'MPI-ESM1-2-LR-r6i1p1f1', 'MPI-ESM1-2-LR-r7i1p1f1',
'MPI-ESM1-2-LR-r8i1p1f1', 'MPI-ESM1-2-LR-r9i1p1f1']

dsWi_mpi_fam = dsWi.sel(member1=mpi_fam,member2=mpi_fam)
mpi_fam_all = np.unique(dsWi_mpi_fam)
mpi_fam_all = mpi_fam_all[~np.isnan(mpi_fam_all)]

dsWi_mpi_rest = dsWi.sel(member1=mpi_fam).drop_sel(member2=mpi_fam)
mpi_rest_all = np.unique(dsWi_mpi_rest)
mpi_rest_all = mpi_rest_all[~np.isnan(mpi_rest_all)]

mpi_full_fam = ['MPI-ESM1-2-HR-r1i1p1f1','MPI-ESM1-2-HR-r2i1p1f1',
'NESM3-r1i1p1f1','NESM3-r2i1p1f1','AWI-CM-1-1-MR-r1i1p1f1']

dsWi_mpi_fam_rest = dsWi.sel(member1=mpi_fam,member2=mpi_full_fam)
mpi_fam_rest_all = np.unique(dsWi_mpi_fam_rest)
mpi_fam_rest_all = mpi_fam_rest_all[~np.isnan(mpi_fam_rest_all)]


######################################
nesm3_fam = ['NESM3-r1i1p1f1','NESM3-r2i1p1f1']

dsWi_nesm3_fam = dsWi.sel(member1=nesm3_fam,member2=nesm3_fam)
nesm3_fam_all = np.unique(dsWi_nesm3_fam)
nesm3_fam_all = nesm3_fam_all[~np.isnan(nesm3_fam_all)]

dsWi_nesm3_rest = dsWi.sel(member1=nesm3_fam).drop_sel(member2=nesm3_fam)
nesm3_rest_all = np.unique(dsWi_nesm3_rest)
nesm3_rest_all = nesm3_rest_all[~np.isnan(nesm3_rest_all)]

mpi_full_fam = ['MPI-ESM1-2-LR-r10i1p1f1', 'MPI-ESM1-2-LR-r1i1p1f1',
'MPI-ESM1-2-LR-r2i1p1f1', 'MPI-ESM1-2-LR-r3i1p1f1',
'MPI-ESM1-2-LR-r4i1p1f1', 'MPI-ESM1-2-LR-r5i1p1f1',
'MPI-ESM1-2-LR-r6i1p1f1', 'MPI-ESM1-2-LR-r7i1p1f1',
'MPI-ESM1-2-LR-r8i1p1f1', 'MPI-ESM1-2-LR-r9i1p1f1',
'MPI-ESM1-2-HR-r1i1p1f1','MPI-ESM1-2-HR-r2i1p1f1',
'AWI-CM-1-1-MR-r1i1p1f1']

dsWi_nesm3_fam_rest = dsWi.sel(member1=nesm3_fam,member2=mpi_full_fam)
nesm3_fam_rest_all = np.unique(dsWi_nesm3_fam_rest)
nesm3_fam_rest_all = nesm3_fam_rest_all[~np.isnan(nesm3_fam_rest_all)]

#####################################

fgoals_g_fam = ['FGOALS-g3-r1i1p1f1', 'FGOALS-g3-r2i1p1f1',
'FGOALS-g3-r3i1p1f1', 'FGOALS-g3-r4i1p1f1']
# closest relative: 'E3SM-1-1-r1i1p1f1'

dsWi_fgoals_g_fam = dsWi.sel(member1=fgoals_g_fam,member2=fgoals_g_fam)
fgoals_g_fam_all = np.unique(dsWi_fgoals_g_fam)
fgoals_g_fam_all = fgoals_g_fam_all[~np.isnan(fgoals_g_fam_all)]

dsWi_fgoals_g_rest = dsWi.sel(member1=fgoals_g_fam).drop_sel(member2=fgoals_g_fam)
fgoals_g_rest_all = np.unique(dsWi_fgoals_g_rest)
fgoals_g_rest_all = fgoals_g_rest_all[~np.isnan(fgoals_g_rest_all)]

#####################################

access_2_fam = ['ACCESS-CM2-r1i1p1f1', 'ACCESS-CM2-r2i1p1f1',
'ACCESS-CM2-r3i1p1f1']

dsWi_access_2_fam = dsWi.sel(member1=access_2_fam,member2=access_2_fam)
access_2_fam_all = np.unique(dsWi_access_2_fam)
access_2_fam_all = access_2_fam_all[~np.isnan(access_2_fam_all)]

dsWi_access_2_rest = dsWi.sel(member1=access_2_fam).drop_sel(member2=access_2_fam)
access_2_rest_all = np.unique(dsWi_access_2_rest)
access_2_rest_all = access_2_rest_all[~np.isnan(access_2_rest_all)]

had_full_fam = ['HadGEM3-GC31-LL-r1i1p1f3',
'HadGEM3-GC31-LL-r2i1p1f3', 'HadGEM3-GC31-LL-r3i1p1f3',
'HadGEM3-GC31-LL-r4i1p1f3','HadGEM3-GC31-MM-r1i1p1f3',
'HadGEM3-GC31-MM-r2i1p1f3', 'HadGEM3-GC31-MM-r3i1p1f3',
'HadGEM3-GC31-MM-r4i1p1f3','KACE-1-0-G-r1i1p1f1',
'KACE-1-0-G-r2i1p1f1', 'KACE-1-0-G-r3i1p1f1',
'UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2','UKESM1-0-LL-r8i1p1f2']

dsWi_access_2_fam_rest = dsWi.sel(member1=access_2_fam,member2=had_full_fam)
access_2_fam_rest_all = np.unique(dsWi_access_2_fam_rest)
access_2_fam_rest_all = access_2_fam_rest_all[~np.isnan(access_2_fam_rest_all)]

##
had_ll_fam = ['HadGEM3-GC31-LL-r1i1p1f3',
'HadGEM3-GC31-LL-r2i1p1f3', 'HadGEM3-GC31-LL-r3i1p1f3',
'HadGEM3-GC31-LL-r4i1p1f3']

dsWi_had_ll_fam = dsWi.sel(member1=had_ll_fam,member2=had_ll_fam)
had_ll_fam_all = np.unique(dsWi_had_ll_fam)
had_ll_fam_all = had_ll_fam_all[~np.isnan(had_ll_fam_all)]

dsWi_had_ll_rest = dsWi.sel(member1=had_ll_fam).drop_sel(member2=had_ll_fam)
had_ll_rest_all = np.unique(dsWi_had_ll_rest)
had_ll_rest_all = had_ll_rest_all[~np.isnan(had_ll_rest_all)]

had_full_fam = ['ACCESS-CM2-r1i1p1f1', 'ACCESS-CM2-r2i1p1f1',
'ACCESS-CM2-r3i1p1f1','HadGEM3-GC31-MM-r1i1p1f3',
'HadGEM3-GC31-MM-r2i1p1f3', 'HadGEM3-GC31-MM-r3i1p1f3',
'HadGEM3-GC31-MM-r4i1p1f3','KACE-1-0-G-r1i1p1f1',
'KACE-1-0-G-r2i1p1f1', 'KACE-1-0-G-r3i1p1f1',
'UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2','UKESM1-0-LL-r8i1p1f2']

dsWi_had_ll_fam_rest = dsWi.sel(member1=had_ll_fam,member2=had_full_fam)
had_ll_fam_rest_all = np.unique(dsWi_had_ll_fam_rest)
had_ll_fam_rest_all = had_ll_fam_rest_all[~np.isnan(had_ll_fam_rest_all)]

##
had_mm_fam = ['HadGEM3-GC31-MM-r1i1p1f3',
'HadGEM3-GC31-MM-r2i1p1f3', 'HadGEM3-GC31-MM-r3i1p1f3',
'HadGEM3-GC31-MM-r4i1p1f3']

dsWi_had_mm_fam = dsWi.sel(member1=had_mm_fam,member2=had_mm_fam)
had_mm_fam_all = np.unique(dsWi_had_mm_fam)
had_mm_fam_all = had_mm_fam_all[~np.isnan(had_mm_fam_all)]

dsWi_had_mm_rest = dsWi.sel(member1=had_mm_fam).drop_sel(member2=had_mm_fam)
had_mm_rest_all = np.unique(dsWi_had_mm_rest)
had_mm_rest_all = had_mm_rest_all[~np.isnan(had_mm_rest_all)]

had_full_fam = ['ACCESS-CM2-r1i1p1f1', 'ACCESS-CM2-r2i1p1f1',
'ACCESS-CM2-r3i1p1f1','HadGEM3-GC31-LL-r1i1p1f3',
'HadGEM3-GC31-LL-r2i1p1f3', 'HadGEM3-GC31-LL-r3i1p1f3',
'HadGEM3-GC31-LL-r4i1p1f3','KACE-1-0-G-r1i1p1f1',
'KACE-1-0-G-r2i1p1f1', 'KACE-1-0-G-r3i1p1f1',
'UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2','UKESM1-0-LL-r8i1p1f2']

dsWi_had_mm_fam_rest = dsWi.sel(member1=had_mm_fam,member2=had_full_fam)
had_mm_fam_rest_all = np.unique(dsWi_had_mm_fam_rest)
had_mm_fam_rest_all = had_mm_fam_rest_all[~np.isnan(had_mm_fam_rest_all)]

##
kace_fam = ['KACE-1-0-G-r1i1p1f1','KACE-1-0-G-r2i1p1f1', 'KACE-1-0-G-r3i1p1f1']

dsWi_kace_fam = dsWi.sel(member1=kace_fam,member2=kace_fam)
kace_fam_all = np.unique(dsWi_kace_fam)
kace_fam_all = kace_fam_all[~np.isnan(kace_fam_all)]

dsWi_kace_rest = dsWi.sel(member1=kace_fam).drop_sel(member2=kace_fam)
kace_rest_all = np.unique(dsWi_kace_rest)
kace_rest_all = kace_rest_all[~np.isnan(kace_rest_all)]

had_full_fam = ['ACCESS-CM2-r1i1p1f1', 'ACCESS-CM2-r2i1p1f1',
'ACCESS-CM2-r3i1p1f1','HadGEM3-GC31-LL-r1i1p1f3',
'HadGEM3-GC31-LL-r2i1p1f3', 'HadGEM3-GC31-LL-r3i1p1f3',
'HadGEM3-GC31-LL-r4i1p1f3','HadGEM3-GC31-MM-r1i1p1f3',
'HadGEM3-GC31-MM-r2i1p1f3', 'HadGEM3-GC31-MM-r3i1p1f3',
'HadGEM3-GC31-MM-r4i1p1f3','UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2','UKESM1-0-LL-r8i1p1f2']

dsWi_kace_fam_rest = dsWi.sel(member1=kace_fam,member2=had_full_fam)
kace_fam_rest_all = np.unique(dsWi_kace_fam_rest)
kace_fam_rest_all = kace_fam_rest_all[~np.isnan(kace_fam_rest_all)]

##
ukesm1_fam = ['UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2','UKESM1-0-LL-r8i1p1f2']

dsWi_ukesm1_fam = dsWi.sel(member1=ukesm1_fam,member2=ukesm1_fam)
ukesm1_fam_all = np.unique(dsWi_ukesm1_fam)
ukesm1_fam_all = ukesm1_fam_all[~np.isnan(ukesm1_fam_all)]

dsWi_ukesm1_rest = dsWi.sel(member1=ukesm1_fam).drop_sel(member2=ukesm1_fam)
ukesm1_rest_all = np.unique(dsWi_ukesm1_rest)
ukesm1_rest_all = ukesm1_rest_all[~np.isnan(ukesm1_rest_all)]

had_full_fam = ['ACCESS-CM2-r1i1p1f1', 'ACCESS-CM2-r2i1p1f1',
'ACCESS-CM2-r3i1p1f1','HadGEM3-GC31-LL-r1i1p1f3',
'HadGEM3-GC31-LL-r2i1p1f3', 'HadGEM3-GC31-LL-r3i1p1f3',
'HadGEM3-GC31-LL-r4i1p1f3','HadGEM3-GC31-MM-r1i1p1f3',
'HadGEM3-GC31-MM-r2i1p1f3', 'HadGEM3-GC31-MM-r3i1p1f3',
'HadGEM3-GC31-MM-r4i1p1f3','KACE-1-0-G-r1i1p1f1','KACE-1-0-G-r2i1p1f1',
'KACE-1-0-G-r3i1p1f1']

dsWi_ukesm1_fam_rest = dsWi.sel(member1=ukesm1_fam,member2=had_full_fam)
ukesm1_fam_rest_all = np.unique(dsWi_ukesm1_fam_rest)
ukesm1_fam_rest_all = ukesm1_fam_rest_all[~np.isnan(ukesm1_fam_rest_all)]


#####################################

cas_fam = ['CAS-ESM2-0-r1i1p1f1', 'CAS-ESM2-0-r3i1p1f1']
# closest relative: 'GFDL-ESM4-r1i1p1f1'

dsWi_cas_fam = dsWi.sel(member1=cas_fam,member2=cas_fam)
cas_fam_all = np.unique(dsWi_cas_fam)
cas_fam_all = cas_fam_all[~np.isnan(cas_fam_all)]

dsWi_cas_rest = dsWi.sel(member1=cas_fam).drop_sel(member2=cas_fam)
cas_rest_all = np.unique(dsWi_cas_rest)
cas_rest_all = cas_rest_all[~np.isnan(cas_rest_all)]

#####################################


cesm_waccm_fam = ['CESM2-WACCM-r1i1p1f1', 'CESM2-WACCM-r2i1p1f1',
'CESM2-WACCM-r3i1p1f1']

dsWi_cesm_waccm_fam = dsWi.sel(member1=cesm_waccm_fam,member2=cesm_waccm_fam)
cesm_waccm_fam_all = np.unique(dsWi_cesm_waccm_fam)
cesm_waccm_fam_all = cesm_waccm_fam_all[~np.isnan(cesm_waccm_fam_all)]

dsWi_cesm_waccm_rest = dsWi.sel(member1=cesm_waccm_fam).drop_sel(member2=cesm_waccm_fam)
cesm_waccm_rest_all = np.unique(dsWi_cesm_waccm_rest)
cesm_waccm_rest_all = cesm_waccm_rest_all[~np.isnan(cesm_waccm_rest_all)]

cesm_full_fam = ['CESM2-r10i1p1f1', 'CESM2-r11i1p1f1',
'CESM2-r1i1p1f1', 'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1','TaiESM1-r1i1p1f1',
'CMCC-CM2-SR5-r1i1p1f1','CMCC-ESM2-r1i1p1f1','NorESM2-MM-r1i1p1f1']

dsWi_cesm_waccm_fam_rest = dsWi.sel(member1=cesm_waccm_fam,member2=cesm_full_fam)
cesm_waccm_fam_rest_all = np.unique(dsWi_cesm_waccm_fam_rest)
cesm_waccm_fam_rest_all = cesm_waccm_fam_rest_all[~np.isnan(cesm_waccm_fam_rest_all)]

##

cesm_fam = ['CESM2-r10i1p1f1', 'CESM2-r11i1p1f1',
'CESM2-r1i1p1f1', 'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1']

# closest relative: 'AWI-CM-1-1-MR-r1i1p1f1'

dsWi_cesm_fam = dsWi.sel(member1=cesm_fam,member2=cesm_fam)
cesm_fam_all = np.unique(dsWi_cesm_fam)
cesm_fam_all = cesm_fam_all[~np.isnan(cesm_fam_all)]

dsWi_cesm_rest = dsWi.sel(member1=cesm_fam).drop_sel(member2=cesm_fam)
cesm_rest_all = np.unique(dsWi_cesm_rest)
cesm_rest_all = cesm_rest_all[~np.isnan(cesm_rest_all)]

cesm_full_fam = ['CESM2-WACCM-r1i1p1f1', 'CESM2-WACCM-r2i1p1f1',
'CESM2-WACCM-r3i1p1f1','TaiESM1-r1i1p1f1',
'CMCC-CM2-SR5-r1i1p1f1','CMCC-ESM2-r1i1p1f1','NorESM2-MM-r1i1p1f1']

dsWi_cesm_fam_rest = dsWi.sel(member1=cesm_fam,member2=cesm_full_fam)
cesm_fam_rest_all = np.unique(dsWi_cesm_fam_rest)
cesm_fam_rest_all = cesm_fam_rest_all[~np.isnan(cesm_fam_rest_all)]


#####################################

cnrm_esm_fam = ['CNRM-ESM2-1-r1i1p1f2',
'CNRM-ESM2-1-r2i1p1f2', 'CNRM-ESM2-1-r3i1p1f2',
'CNRM-ESM2-1-r4i1p1f2', 'CNRM-ESM2-1-r5i1p1f2']

dsWi_cnrm_esm_fam = dsWi.sel(member1=cnrm_esm_fam,member2=cnrm_esm_fam)
cnrm_esm_fam_all = np.unique(dsWi_cnrm_esm_fam)
cnrm_esm_fam_all = cnrm_esm_fam_all[~np.isnan(cnrm_esm_fam_all)]

dsWi_cnrm_esm_rest = dsWi.sel(member1=cnrm_esm_fam).drop_sel(member2=cnrm_esm_fam)
cnrm_esm_rest_all = np.unique(dsWi_cnrm_esm_rest)
cnrm_esm_rest_all = cnrm_esm_rest_all[~np.isnan(cnrm_esm_rest_all)]

cnrm_full_fam = ['CNRM-CM6-1-r1i1p1f2','CNRM-CM6-1-r2i1p1f2',
'CNRM-CM6-1-r3i1p1f2','CNRM-CM6-1-r4i1p1f2',
'CNRM-CM6-1-r5i1p1f2','CNRM-CM6-1-r6i1p1f2',
'IPSL-CM6A-LR-r14i1p1f1','IPSL-CM6A-LR-r1i1p1f1', 'IPSL-CM6A-LR-r2i1p1f1',
'IPSL-CM6A-LR-r3i1p1f1', 'IPSL-CM6A-LR-r4i1p1f1',
'IPSL-CM6A-LR-r6i1p1f1','CNRM-CM6-1-HR-r1i1p1f2']

dsWi_cnrm_esm_fam_rest = dsWi.sel(member1=cnrm_esm_fam,member2=cnrm_full_fam)
cnrm_esm_fam_rest_all = np.unique(dsWi_cnrm_esm_fam_rest)
cnrm_esm_fam_rest_all = cnrm_esm_fam_rest_all[~np.isnan(cnrm_esm_fam_rest_all)]

##
cnrm_fam = ['CNRM-CM6-1-r1i1p1f2','CNRM-CM6-1-r2i1p1f2',
'CNRM-CM6-1-r3i1p1f2','CNRM-CM6-1-r4i1p1f2',
'CNRM-CM6-1-r5i1p1f2','CNRM-CM6-1-r6i1p1f2']

# closest relative: 'CNRM-CM6-1-HR-r1i1p1f2' ??

dsWi_cnrm_fam = dsWi.sel(member1=cnrm_fam,member2=cnrm_fam)
cnrm_fam_all = np.unique(dsWi_cnrm_fam)
cnrm_fam_all = cnrm_fam_all[~np.isnan(cnrm_fam_all)]

dsWi_cnrm_rest = dsWi.sel(member1=cnrm_fam).drop_sel(member2=cnrm_fam)
cnrm_rest_all = np.unique(dsWi_cnrm_rest)
cnrm_rest_all = cnrm_rest_all[~np.isnan(cnrm_rest_all)]

cnrm_full_fam = ['CNRM-ESM2-1-r1i1p1f2',
'CNRM-ESM2-1-r2i1p1f2', 'CNRM-ESM2-1-r3i1p1f2',
'CNRM-ESM2-1-r4i1p1f2', 'CNRM-ESM2-1-r5i1p1f2',
'IPSL-CM6A-LR-r14i1p1f1',
'IPSL-CM6A-LR-r1i1p1f1', 'IPSL-CM6A-LR-r2i1p1f1',
'IPSL-CM6A-LR-r3i1p1f1', 'IPSL-CM6A-LR-r4i1p1f1',
'IPSL-CM6A-LR-r6i1p1f1','CNRM-CM6-1-HR-r1i1p1f2']

dsWi_cnrm_fam_rest = dsWi.sel(member1=cnrm_fam,member2=cnrm_full_fam)
cnrm_fam_rest_all = np.unique(dsWi_cnrm_fam_rest)
cnrm_fam_rest_all = cnrm_fam_rest_all[~np.isnan(cnrm_fam_rest_all)]

#####################################

ec_earth_veg_fam = ['EC-Earth3-Veg-r1i1p1f1', 'EC-Earth3-Veg-r2i1p1f1',
'EC-Earth3-Veg-r3i1p1f1', 'EC-Earth3-Veg-r4i1p1f1']

dsWi_earth_veg_fam = dsWi.sel(member1=ec_earth_veg_fam,member2=ec_earth_veg_fam)
earth_veg_fam_all = np.unique(dsWi_earth_veg_fam)
earth_veg_fam_all = earth_veg_fam_all[~np.isnan(earth_veg_fam_all)]

dsWi_earth_veg_rest = dsWi.sel(member1=ec_earth_veg_fam).drop_sel(member2=ec_earth_veg_fam)
earth_veg_rest_all = np.unique(dsWi_earth_veg_rest)
earth_veg_rest_all = earth_veg_rest_all[~np.isnan(earth_veg_rest_all)]

earth_full_fam = ['EC-Earth3-r11i1p1f1', 'EC-Earth3-r13i1p1f1',
'EC-Earth3-r15i1p1f1', 'EC-Earth3-r1i1p1f1', 'EC-Earth3-r3i1p1f1',
'EC-Earth3-r4i1p1f1', 'EC-Earth3-r6i1p1f1', 'EC-Earth3-r9i1p1f1']

dsWi_earth_veg_fam_rest = dsWi.sel(member1=ec_earth_veg_fam,member2=earth_full_fam)
earth_veg_fam_rest_all = np.unique(dsWi_earth_veg_fam_rest)
earth_veg_fam_rest_all = earth_veg_fam_rest_all[~np.isnan(earth_veg_fam_rest_all)]

##
ec_earth_fam = ['EC-Earth3-r11i1p1f1', 'EC-Earth3-r13i1p1f1',
'EC-Earth3-r15i1p1f1', 'EC-Earth3-r1i1p1f1', 'EC-Earth3-r3i1p1f1',
'EC-Earth3-r4i1p1f1', 'EC-Earth3-r6i1p1f1', 'EC-Earth3-r9i1p1f1']
# closest relative: 'HadGEM3-GC31-MM-r4i1p1f3'

dsWi_earth_fam = dsWi.sel(member1=ec_earth_fam,member2=ec_earth_fam)
earth_fam_all = np.unique(dsWi_earth_fam)
earth_fam_all = earth_fam_all[~np.isnan(earth_fam_all)]

dsWi_earth_rest = dsWi.sel(member1=ec_earth_fam).drop_sel(member2=ec_earth_fam)
earth_rest_all = np.unique(dsWi_earth_rest)
earth_rest_all = earth_rest_all[~np.isnan(earth_rest_all)]

earth_full_fam = ['EC-Earth3-Veg-r1i1p1f1', 'EC-Earth3-Veg-r2i1p1f1',
'EC-Earth3-Veg-r3i1p1f1', 'EC-Earth3-Veg-r4i1p1f1']

dsWi_earth_fam_rest = dsWi.sel(member1=ec_earth_fam,member2=earth_full_fam)
earth_fam_rest_all = np.unique(dsWi_earth_fam_rest)
earth_fam_rest_all = earth_fam_rest_all[~np.isnan(earth_fam_rest_all)]


#####################################

giss_fam = ['GISS-E2-1-G-r1i1p3f1',
'GISS-E2-1-G-r2i1p3f1','GISS-E2-1-G-r3i1p3f1', 'GISS-E2-1-G-r4i1p3f1',
'GISS-E2-1-G-r5i1p3f1','GISS-E2-1-G-r1i1p5f1']
# closest relative: 'MPI-ESM1-2-HR-r1i1p1f1'

dsWi_giss_fam = dsWi.sel(member1=giss_fam,member2=giss_fam)
giss_fam_all = np.unique(dsWi_giss_fam)
giss_fam_all = giss_fam_all[~np.isnan(giss_fam_all)]

dsWi_giss_rest = dsWi.sel(member1=giss_fam).drop_sel(member2=giss_fam)
giss_rest_all = np.unique(dsWi_giss_rest)
giss_rest_all = giss_rest_all[~np.isnan(giss_rest_all)]

#####################################

ipsl_fam = ['IPSL-CM6A-LR-r14i1p1f1',
'IPSL-CM6A-LR-r1i1p1f1', 'IPSL-CM6A-LR-r2i1p1f1',
'IPSL-CM6A-LR-r3i1p1f1', 'IPSL-CM6A-LR-r4i1p1f1',
'IPSL-CM6A-LR-r6i1p1f1']
# closest relative: 'CNRM-CM6-1-r3i1p1f2'

dsWi_ipsl_fam = dsWi.sel(member1=ipsl_fam,member2=ipsl_fam)
ipsl_fam_all = np.unique(dsWi_ipsl_fam)
ipsl_fam_all = ipsl_fam_all[~np.isnan(ipsl_fam_all)]

dsWi_ipsl_rest = dsWi.sel(member1=ipsl_fam).drop_sel(member2=ipsl_fam)
ipsl_rest_all = np.unique(dsWi_ipsl_rest)
ipsl_rest_all = ipsl_rest_all[~np.isnan(ipsl_rest_all)]

cnrm_full_fam = ['CNRM-ESM2-1-r1i1p1f2',
'CNRM-ESM2-1-r2i1p1f2', 'CNRM-ESM2-1-r3i1p1f2',
'CNRM-ESM2-1-r4i1p1f2', 'CNRM-ESM2-1-r5i1p1f2',
'CNRM-CM6-1-r1i1p1f2','CNRM-CM6-1-r2i1p1f2',
'CNRM-CM6-1-r3i1p1f2','CNRM-CM6-1-r4i1p1f2',
'CNRM-CM6-1-r5i1p1f2','CNRM-CM6-1-r6i1p1f2',
'CNRM-CM6-1-HR-r1i1p1f2']

dsWi_ipsl_fam_rest = dsWi.sel(member1=ipsl_fam,member2=cnrm_full_fam)
ipsl_fam_rest_all = np.unique(dsWi_ipsl_fam_rest)
ipsl_fam_rest_all = ipsl_fam_rest_all[~np.isnan(ipsl_fam_rest_all)]


#####################################

mri_fam = ['MRI-ESM2-0-r1i1p1f1', 'MRI-ESM2-0-r1i2p1f1']

# closest relative: 'CESM2-r10i1p1f1'

dsWi_mri_fam = dsWi.sel(member1=mri_fam,member2=mri_fam)
mri_fam_all = np.unique(dsWi_mri_fam)
mri_fam_all = mri_fam_all[~np.isnan(mri_fam_all)]

dsWi_mri_rest = dsWi.sel(member1=mri_fam).drop_sel(member2=mri_fam)
mri_rest_all = np.unique(dsWi_mri_rest)
mri_rest_all = mri_rest_all[~np.isnan(mri_rest_all)]

#####################################

# Individuals

dsWi_taiesm1_rest = dsWi.sel(member1='TaiESM1-r1i1p1f1').drop_sel(member2='TaiESM1-r1i1p1f1')
taiesm1_rest_all = np.unique(dsWi_taiesm1_rest)
taiesm1_rest_all = taiesm1_rest_all[~np.isnan(taiesm1_rest_all)]

cesm_full_fam = ['CESM2-WACCM-r1i1p1f1', 'CESM2-WACCM-r2i1p1f1',
'CESM2-WACCM-r3i1p1f1','CESM2-r10i1p1f1', 'CESM2-r11i1p1f1',
'CESM2-r1i1p1f1', 'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1',
'CMCC-CM2-SR5-r1i1p1f1','CMCC-ESM2-r1i1p1f1','NorESM2-MM-r1i1p1f1']

dsWi_taiesm1_fam_rest = dsWi.sel(member1='TaiESM1-r1i1p1f1',member2=cesm_full_fam)
taiesm1_fam_rest_all = np.unique(dsWi_taiesm1_fam_rest)
taiesm1_fam_rest_all = taiesm1_fam_rest_all[~np.isnan(taiesm1_fam_rest_all)]



dsWi_mcm_rest = dsWi.sel(member1='MCM-UA-1-0-r1i1p1f2').drop_sel(member2='MCM-UA-1-0-r1i1p1f2')
mcm_rest_all = np.unique(dsWi_mcm_rest)
mcm_rest_all = mcm_rest_all[~np.isnan(mcm_rest_all)]


dsWi_kiost_rest = dsWi.sel(member1='KIOST-ESM-r1i1p1f1').drop_sel(member2='KIOST-ESM-r1i1p1f1')
kiost_rest_all = np.unique(dsWi_kiost_rest)
kiost_rest_all = kiost_rest_all[~np.isnan(kiost_rest_all)]


dsWi_fgoals_rest = dsWi.sel(member1='FGOALS-f3-L-r1i1p1f1').drop_sel(member2='FGOALS-f3-L-r1i1p1f1')
fgoals_rest_all = np.unique(dsWi_fgoals_rest)
fgoals_rest_all = fgoals_rest_all[~np.isnan(fgoals_rest_all)]


dsWi_e3sm_rest = dsWi.sel(member1='E3SM-1-1-r1i1p1f1').drop_sel(member2='E3SM-1-1-r1i1p1f1')
e3sm_rest_all = np.unique(dsWi_e3sm_rest)
e3sm_rest_all = e3sm_rest_all[~np.isnan(e3sm_rest_all)]


dsWi_cnrm_hr_rest = dsWi.sel(member1='CNRM-CM6-1-HR-r1i1p1f2').drop_sel(member2='CNRM-CM6-1-HR-r1i1p1f2')
cnrm_hr_rest_all = np.unique(dsWi_cnrm_hr_rest)
cnrm_hr_rest_all = cnrm_hr_rest_all[~np.isnan(cnrm_hr_rest_all)]

cnrm_full_fam = ['CNRM-ESM2-1-r1i1p1f2',
'CNRM-ESM2-1-r2i1p1f2', 'CNRM-ESM2-1-r3i1p1f2',
'CNRM-ESM2-1-r4i1p1f2', 'CNRM-ESM2-1-r5i1p1f2',
'CNRM-CM6-1-r1i1p1f2','CNRM-CM6-1-r2i1p1f2',
'CNRM-CM6-1-r3i1p1f2','CNRM-CM6-1-r4i1p1f2',
'CNRM-CM6-1-r5i1p1f2','CNRM-CM6-1-r6i1p1f2',
'IPSL-CM6A-LR-r14i1p1f1',
'IPSL-CM6A-LR-r1i1p1f1', 'IPSL-CM6A-LR-r2i1p1f1',
'IPSL-CM6A-LR-r3i1p1f1', 'IPSL-CM6A-LR-r4i1p1f1',
'IPSL-CM6A-LR-r6i1p1f1']

dsWi_cnrm_hr_fam_rest = dsWi.sel(member1='CNRM-CM6-1-HR-r1i1p1f2',member2=cnrm_full_fam)
cnrm_hr_fam_rest_all = np.unique(dsWi_cnrm_hr_fam_rest)
cnrm_hr_fam_rest_all = cnrm_hr_fam_rest_all[~np.isnan(cnrm_hr_fam_rest_all)]

##

dsWi_awi_rest = dsWi.sel(member1='AWI-CM-1-1-MR-r1i1p1f1').drop_sel(member2='AWI-CM-1-1-MR-r1i1p1f1')
awi_rest_all = np.unique(dsWi_awi_rest)
awi_rest_all = awi_rest_all[~np.isnan(awi_rest_all)]

mpi_full_fam = ['MPI-ESM1-2-LR-r10i1p1f1', 'MPI-ESM1-2-LR-r1i1p1f1',
'MPI-ESM1-2-LR-r2i1p1f1', 'MPI-ESM1-2-LR-r3i1p1f1',
'MPI-ESM1-2-LR-r4i1p1f1', 'MPI-ESM1-2-LR-r5i1p1f1',
'MPI-ESM1-2-LR-r6i1p1f1', 'MPI-ESM1-2-LR-r7i1p1f1',
'MPI-ESM1-2-LR-r8i1p1f1', 'MPI-ESM1-2-LR-r9i1p1f1',
'MPI-ESM1-2-HR-r1i1p1f1','MPI-ESM1-2-HR-r2i1p1f1',
'NESM3-r1i1p1f1','NESM3-r2i1p1f1']

dsWi_awi_fam_rest = dsWi.sel(member1='AWI-CM-1-1-MR-r1i1p1f1',member2=mpi_full_fam)
awi_fam_rest_all = np.unique(dsWi_awi_fam_rest)
awi_fam_rest_all = awi_fam_rest_all[~np.isnan(awi_fam_rest_all)]

##
dsWi_cmcc_5_rest = dsWi.sel(member1='CMCC-CM2-SR5-r1i1p1f1').drop_sel(member2='CMCC-CM2-SR5-r1i1p1f1')
cmcc_5_rest_all = np.unique(dsWi_cmcc_5_rest)
cmcc_5_rest_all = cmcc_5_rest_all[~np.isnan(cmcc_5_rest_all)]

cesm_full_fam = ['CESM2-WACCM-r1i1p1f1', 'CESM2-WACCM-r2i1p1f1',
'CESM2-WACCM-r3i1p1f1','CESM2-r10i1p1f1', 'CESM2-r11i1p1f1',
'CESM2-r1i1p1f1', 'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1','TaiESM1-r1i1p1f1',
'CMCC-ESM2-r1i1p1f1','NorESM2-MM-r1i1p1f1']

dsWi_cmcc_5_fam_rest = dsWi.sel(member1='CMCC-CM2-SR5-r1i1p1f1',member2=cesm_full_fam)
cmcc_5_fam_rest_all = np.unique(dsWi_cmcc_5_fam_rest)
cmcc_5_fam_rest_all = cmcc_5_fam_rest_all[~np.isnan(cmcc_5_fam_rest_all)]

##
dsWi_cmcc_rest = dsWi.sel(member1='CMCC-ESM2-r1i1p1f1').drop_sel(member2='CMCC-ESM2-r1i1p1f1')
cmcc_rest_all = np.unique(dsWi_cmcc_rest)
cmcc_rest_all = cmcc_rest_all[~np.isnan(cmcc_rest_all)]

cesm_full_fam = ['CESM2-WACCM-r1i1p1f1', 'CESM2-WACCM-r2i1p1f1',
'CESM2-WACCM-r3i1p1f1','CESM2-r10i1p1f1', 'CESM2-r11i1p1f1',
'CESM2-r1i1p1f1', 'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1','TaiESM1-r1i1p1f1',
'CMCC-CM2-SR5-r1i1p1f1','NorESM2-MM-r1i1p1f1']

dsWi_cmcc_fam_rest = dsWi.sel(member1='CMCC-ESM2-r1i1p1f1',member2=cesm_full_fam)
cmcc_fam_rest_all = np.unique(dsWi_cmcc_fam_rest)
cmcc_fam_rest_all = cmcc_fam_rest_all[~np.isnan(cmcc_fam_rest_all)]

##
dsWi_noresm_rest = dsWi.sel(member1='NorESM2-MM-r1i1p1f1').drop_sel(member2='NorESM2-MM-r1i1p1f1')
noresm_rest_all = np.unique(dsWi_noresm_rest)
noresm_rest_all = noresm_rest_all[~np.isnan(noresm_rest_all)]

cesm_full_fam = ['CESM2-WACCM-r1i1p1f1', 'CESM2-WACCM-r2i1p1f1',
'CESM2-WACCM-r3i1p1f1','CESM2-r10i1p1f1', 'CESM2-r11i1p1f1',
'CESM2-r1i1p1f1', 'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1','TaiESM1-r1i1p1f1',
'CMCC-CM2-SR5-r1i1p1f1','CMCC-ESM2-r1i1p1f1']

dsWi_noresm_fam_rest = dsWi.sel(member1='NorESM2-MM-r1i1p1f1',member2=cesm_full_fam)
noresm_fam_rest_all = np.unique(dsWi_noresm_fam_rest)
noresm_fam_rest_all = noresm_fam_rest_all[~np.isnan(noresm_fam_rest_all)]


dsWi_gfdl_rest = dsWi.sel(member1='GFDL-CM4-r1i1p1f1').drop_sel(member2='GFDL-CM4-r1i1p1f1')
gfdl_rest_all = np.unique(dsWi_gfdl_rest)
gfdl_rest_all = gfdl_rest_all[~np.isnan(gfdl_rest_all)]

dsWi_gfdl_e_rest = dsWi.sel(member1='GFDL-ESM4-r1i1p1f1').drop_sel(member2='GFDL-ESM4-r1i1p1f1')
gfdl_e_rest_all = np.unique(dsWi_gfdl_e_rest)
gfdl_e_rest_all = gfdl_e_rest_all[~np.isnan(gfdl_e_rest_all)]

dsWi_inm_4_rest = dsWi.sel(member1='INM-CM4-8-r1i1p1f1').drop_sel(member2='INM-CM4-8-r1i1p1f1')
inm_4_rest_all = np.unique(dsWi_inm_4_rest)
inm_4_rest_all = inm_4_rest_all[~np.isnan(inm_4_rest_all)]

dsWi_inm_5_rest = dsWi.sel(member1='INM-CM5-0-r1i1p1f1').drop_sel(member2='INM-CM5-0-r1i1p1f1')
inm_5_rest_all = np.unique(dsWi_inm_5_rest)
inm_5_rest_all = inm_5_rest_all[~np.isnan(inm_5_rest_all)]

########################

fig = plt.figure(figsize=(8,7))
ax = plt.subplot(111)

# Annual Global Land SAT climatology; 1950-2014
plt.plot(access_fam_all,0*np.ones(np.size(access_fam_all)),'|',color='tab:red')
plt.plot(access_rest_all,0*np.ones(np.size(access_rest_all)),'|',color='silver')

plt.plot(had_mm_fam_all,1*np.ones(np.size(had_mm_fam_all)),'|',color='tab:red')
plt.plot(had_mm_rest_all,1*np.ones(np.size(had_mm_rest_all)),'|',color='silver')
plt.plot(had_mm_fam_rest_all,1*np.ones(np.size(had_mm_fam_rest_all)),'|',color='dimgray')

plt.plot(kace_fam_all,2*np.ones(np.size(kace_fam_all)),'|',color='tab:red')
plt.plot(kace_rest_all,2*np.ones(np.size(kace_rest_all)),'|',color='silver')
plt.plot(kace_fam_rest_all,2*np.ones(np.size(kace_fam_rest_all)),'|',color='dimgray')

plt.plot(access_2_fam_all,3*np.ones(np.size(access_2_fam_all)),'|',color='tab:red')
plt.plot(access_2_rest_all,3*np.ones(np.size(access_2_rest_all)),'|',color='silver')
plt.plot(access_2_fam_rest_all,3*np.ones(np.size(access_2_fam_rest_all)),'|',color='dimgray')

plt.plot(had_ll_fam_all,4*np.ones(np.size(had_ll_fam_all)),'|',color='tab:red')
plt.plot(had_ll_rest_all,4*np.ones(np.size(had_ll_rest_all)),'|',color='silver')
plt.plot(had_ll_fam_rest_all,4*np.ones(np.size(had_ll_fam_rest_all)),'|',color='dimgray')

plt.plot(ukesm1_fam_all,5*np.ones(np.size(ukesm1_fam_all)),'|',color='tab:red')
plt.plot(ukesm1_rest_all,5*np.ones(np.size(ukesm1_rest_all)),'|',color='silver')
plt.plot(ukesm1_fam_rest_all,5*np.ones(np.size(ukesm1_fam_rest_all)),'|',color='dimgray')

plt.axhline(6,color='silver')
#############

plt.plot(taiesm1_rest_all,7*np.ones(np.size(taiesm1_rest_all)),'|',color='silver')
plt.plot(taiesm1_fam_rest_all,7*np.ones(np.size(taiesm1_fam_rest_all)),'|',color='dimgray')

plt.plot(cmcc_rest_all,8*np.ones(np.size(cmcc_rest_all)),'|',color='silver')
plt.plot(cmcc_fam_rest_all,8*np.ones(np.size(cmcc_fam_rest_all)),'|',color='dimgray')

plt.plot(cmcc_5_rest_all,9*np.ones(np.size(cmcc_5_rest_all)),'|',color='silver')
plt.plot(cmcc_5_fam_rest_all,9*np.ones(np.size(cmcc_5_fam_rest_all)),'|',color='dimgray')

plt.plot(noresm_rest_all,10*np.ones(np.size(noresm_rest_all)),'|',color='silver')
plt.plot(noresm_fam_rest_all,10*np.ones(np.size(noresm_fam_rest_all)),'|',color='dimgray')

plt.plot(cesm_waccm_fam_all,11*np.ones(np.size(cesm_waccm_fam_all)),'|',color='darkgoldenrod')
plt.plot(cesm_waccm_rest_all,11*np.ones(np.size(cesm_waccm_rest_all)),'|',color='silver')
plt.plot(cesm_waccm_fam_rest_all,11*np.ones(np.size(cesm_waccm_fam_rest_all)),'|',color='dimgray')

plt.plot(cesm_fam_all,12*np.ones(np.size(cesm_fam_all)),'|',color='darkgoldenrod')
plt.plot(cesm_rest_all,12*np.ones(np.size(cesm_rest_all)),'|',color='silver')
plt.plot(cesm_fam_rest_all,12*np.ones(np.size(cesm_fam_rest_all)),'|',color='dimgray')

plt.axhline(13,color='silver')
#############

plt.plot(cnrm_hr_rest_all,14*np.ones(np.size(cnrm_hr_rest_all)),'|',color='silver')
plt.plot(cnrm_hr_fam_rest_all,14*np.ones(np.size(cnrm_hr_fam_rest_all)),'|',color='dimgray')

plt.plot(cnrm_esm_fam_all,15*np.ones(np.size(cnrm_esm_fam_all)),'|',color='cornflowerblue')
plt.plot(cnrm_esm_rest_all,15*np.ones(np.size(cnrm_esm_rest_all)),'|',color='silver')
plt.plot(cnrm_esm_fam_rest_all,15*np.ones(np.size(cnrm_esm_fam_rest_all)),'|',color='dimgray')

plt.plot(ipsl_fam_all,16*np.ones(np.size(ipsl_fam_all)),'|',color='royalblue')
plt.plot(ipsl_rest_all,16*np.ones(np.size(ipsl_rest_all)),'|',color='silver')
plt.plot(ipsl_fam_rest_all,16*np.ones(np.size(ipsl_fam_rest_all)),'|',color='dimgray')

plt.plot(cnrm_fam_all,17*np.ones(np.size(cnrm_fam_all)),'|',color='cornflowerblue')
plt.plot(cnrm_rest_all,17*np.ones(np.size(cnrm_rest_all)),'|',color='silver')
plt.plot(cnrm_fam_rest_all,17*np.ones(np.size(cnrm_fam_rest_all)),'|',color='dimgray')

plt.axhline(18,color='silver')
################

plt.plot(awi_rest_all,19*np.ones(np.size(awi_rest_all)),'|',color='silver')
plt.plot(awi_fam_rest_all,19*np.ones(np.size(awi_fam_rest_all)),'|',color='dimgray')

plt.plot(nesm3_fam_all,20*np.ones(np.size(nesm3_fam_all)),'|',color='tab:orange')
plt.plot(nesm3_rest_all,20*np.ones(np.size(nesm3_rest_all)),'|',color='silver')
plt.plot(nesm3_fam_rest_all,20*np.ones(np.size(nesm3_fam_rest_all)),'|',color='dimgray')

plt.plot(mpi_fam_all,21*np.ones(np.size(mpi_fam_all)),'|',color='tab:orange')
plt.plot(mpi_rest_all,21*np.ones(np.size(mpi_rest_all)),'|',color='silver')
plt.plot(mpi_fam_rest_all,21*np.ones(np.size(mpi_fam_rest_all)),'|',color='dimgray')

plt.plot(mpi_hr_fam_all,22*np.ones(np.size(mpi_hr_fam_all)),'|',color='tab:orange')
plt.plot(mpi_hr_rest_all,22*np.ones(np.size(mpi_hr_rest_all)),'|',color='silver')
plt.plot(mpi_hr_fam_rest_all,22*np.ones(np.size(mpi_hr_fam_rest_all)),'|',color='dimgray')

plt.axhline(23,color='silver')
################

plt.plot(gfdl_rest_all,24*np.ones(np.size(gfdl_rest_all)),'|',color='silver')
plt.plot(gfdl_rest_all[0],24,'|',color='dimgray')
plt.plot(gfdl_e_rest_all,25*np.ones(np.size(gfdl_e_rest_all)),'|',color='silver')
plt.plot(gfdl_e_rest_all[0],25,'|',color='dimgray')

plt.axhline(26,color='silver')
################

plt.plot(earth_rest_all,27*np.ones(np.size(earth_rest_all)),'|',color='silver')
plt.plot(earth_fam_rest_all,27*np.ones(np.size(earth_fam_rest_all)),'|',color='dimgray')
plt.plot(earth_fam_all,27*np.ones(np.size(earth_fam_all)),'|',color='darkgreen')

plt.plot(earth_veg_rest_all,28*np.ones(np.size(earth_veg_rest_all)),'|',color='silver')
plt.plot(earth_veg_fam_rest_all,28*np.ones(np.size(earth_veg_fam_rest_all)),'|',color='dimgray')
plt.plot(earth_veg_fam_all,28*np.ones(np.size(earth_veg_fam_all)),'|',color='darkgreen')

plt.axhline(29,color='silver')
################

plt.plot(fgoals_rest_all,30*np.ones(np.size(fgoals_rest_all)),'|',color='silver')

plt.plot(fgoals_g_fam_all,31*np.ones(np.size(fgoals_g_fam_all)),'|',color='maroon')
plt.plot(fgoals_g_rest_all,31*np.ones(np.size(fgoals_g_rest_all)),'|',color='silver')

plt.axhline(32,color='silver')
################

plt.plot(inm_4_rest_all,33*np.ones(np.size(inm_4_rest_all)),'|',color='silver')
plt.plot(inm_4_rest_all[0],33,'|',color='dimgray')
plt.plot(inm_5_rest_all,34*np.ones(np.size(inm_5_rest_all)),'|',color='silver')
plt.plot(inm_5_rest_all[0],34,'|',color='dimgray')

plt.axhline(35,color='silver')
################

plt.plot(miroc_fam_all,36*np.ones(np.size(miroc_fam_all)),'|',color='lightsalmon')
plt.plot(miroc_rest_all,36*np.ones(np.size(miroc_rest_all)),'|',color='silver')

plt.plot(miroc_esl_fam_all,37*np.ones(np.size(miroc_esl_fam_all)),'|',color='lightsalmon')
plt.plot(miroc_esl_rest_all,37*np.ones(np.size(miroc_esl_rest_all)),'|',color='silver')

plt.axhline(38,color='silver')
################

plt.plot(mri_fam_all,39*np.ones(np.size(mri_fam_all)),'|',color='palevioletred')
plt.plot(mri_rest_all,39*np.ones(np.size(mri_rest_all)),'|',color='silver')

#################

plt.plot(e3sm_rest_all,40*np.ones(np.size(e3sm_rest_all)),'|',color='silver')

#################

plt.plot(can_fam_all,41*np.ones(np.size(can_fam_all)),'|',color='dodgerblue')
plt.plot(can_rest_all,41*np.ones(np.size(can_rest_all)),'|',color='silver')

################

plt.plot(cas_fam_all,42*np.ones(np.size(cas_fam_all)),'|',color='tab:cyan')
plt.plot(cas_rest_all,42*np.ones(np.size(cas_rest_all)),'|',color='silver')

################

plt.plot(giss_fam_all,43*np.ones(np.size(giss_fam_all)),'|',color='blueviolet')
plt.plot(giss_rest_all,43*np.ones(np.size(giss_rest_all)),'|',color='silver')

################

plt.plot(mcm_rest_all,44*np.ones(np.size(mcm_rest_all)),'|',color='silver')
plt.plot(kiost_rest_all,45*np.ones(np.size(kiost_rest_all)),'|',color='silver')

################
plt.xlim([0,2.2]) ###########
plt.ylim([-1,46])
yticks = np.arange(0,46,1)
ax.set_yticks(yticks)
xticks = np.arange(0,2.4,0.2) ###########
ax.set_xticks(xticks)
labels = ['1) ACCESS-ESM1-5','2) HadGEM3-GC31-MM','3) KACE-1-0-G','4) ACCESS-CM2','5) HadGEM3-GC31-LL','6) UKESM1-0-LL','',
'7) TaiESM1','8) CMCC-ESM2','9) CMCC-CM2-SR5','10) NorESM2-MM','11) CESM2-WACCM','12) CESM2','',
'13) CNRM-CM6-1-HR','14) CNRM-ESM2-1','15) IPSL-CM6A-LR','16) CNRM-CM6-1','',
'17) AWI-CM-1-1-MR','18) NESM3','19) MPI-ESM1-2-LR','20) MPI-ESM1-2-HR','',
'21) GFDL-CM4','22) GFDL-ESM4','','23) EC-Earth3','24) EC-Earth3-Veg','',
'25) FGOALS-f3-L','26) FGOALS-g3','','27) INM-CM4-8','28) INM-CM5-0','','29) MIROC6','30) MIROC-ES2L','','31) MRI-ESM2-0',''
'32) E3SM-1-1','33) CanESM5','34) CAS-ESM2-0','35) GISS-E2-1-G','36) MCM-UA-1-0','37) KIOST-ESM']
xlabels = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2]
ax.set_yticklabels(labels,fontsize=9,rotation = 0)
ax.set_xticklabels(xlabels,fontsize=9)
plt.xlabel('intermember distance',fontsize=11)
ax.set_axisbelow(True)
ax.invert_yaxis()
plt.title('a) CMIP6 intermember distances',fontsize=12,fontweight='bold',loc='left') #,pad=14)
plt.savefig('Fig3_dists_CMIP6.png',bbox_inches='tight',dpi=300)

###############

from sklearn.datasets import load_digits
from sklearn.manifold import MDS

embedding = MDS(n_components=2,metric=True,dissimilarity='euclidean',random_state=0)
dsWi_z = np.nan_to_num(dsWi,nan=0)

init = [( 2.44906514e+00, -2.99519089e+00),( 2.38034640e+00, -3.02393303e+00),( 2.50275855e+00, -2.93680616e+00),( 2.76399203e+00, -3.95688630e+00),( 2.79927896e+00, -4.07504050e+00),( 2.85300175e+00, -3.96141481e+00),( 2.82821401e+00, -4.02721861e+00),( 2.81663436e+00, -3.99677659e+00),( 2.83367819e+00, -4.03885002e+00),( 2.87794817e+00, -3.95262993e+00),( 2.84584185e+00, -3.94482828e+00),( 2.81937089e+00, -4.01741401e+00),( 2.84083582e+00, -3.88861328e+00),( 5.98970638e-01, -3.02340691e+00),( 8.74993779e-01,  7.92337048e-01),( 8.42537629e-01,  8.38464169e-01),( 1.35882897e+00, -4.69128566e+00),( 1.45893154e+00, -4.73544682e+00),( 1.51767844e+00, -4.75379368e+00),( 1.47513436e+00, -4.68169318e+00),( 1.57677177e+00, -4.75481223e+00),( 1.61448664e+00, -4.79183098e+00),( 1.57394204e+00, -4.72854635e+00),( 1.55372365e+00, -4.74309294e+00),( 1.26604496e+00, -3.46945298e+00),( 1.22285778e+00, -3.47201686e+00),(-1.99354088e+00, -5.58365083e-01),(-1.79556464e+00, -2.22102434e+00),(-1.81105121e+00, -2.11652415e+00),(-1.79812085e+00, -2.13397818e+00),(-1.77512904e+00, -2.13178686e+00),(-1.84945921e+00, -2.14128350e+00),(-1.82017973e+00, -2.05153616e+00),(-1.27034468e+00, -1.45738625e+00),(-1.25610176e+00, -1.44564786e+00),(-1.19833454e+00, -1.33339782e+00),(-1.23401651e+00, -1.35143895e+00),(-1.20545307e+00, -1.44445606e+00),(-3.37459344e+00, -6.13113915e+00),(-3.15132253e+00, -6.39895375e+00),(-3.40644619e+00, -6.08109333e+00),(-3.14717929e+00, -6.38148932e+00),(-3.37603155e+00, -6.06563943e+00),(-3.20029303e+00, -6.29268363e+00),(-3.42142719e+00, -6.07366099e+00),(-3.18437526e+00, -6.39679198e+00),(-3.40952483e+00, -6.13589988e+00),(-3.12823063e+00, -6.38776080e+00),(-3.40057294e+00, -6.17939758e+00),(-3.19076206e+00, -6.33674145e+00),(-3.42889594e+00, -6.08952106e+00),(-3.17472987e+00, -6.35844168e+00),(-3.40440708e+00, -6.13712842e+00),(-3.15718737e+00, -6.34242425e+00),(-3.41706526e+00, -6.12226214e+00),(-3.08057998e+00, -6.42120951e+00),(-3.43287747e+00, -6.17437472e+00),(-3.08032625e+00, -6.41552911e+00),(-3.35016537e+00, -5.99197937e+00),(-3.22316542e+00, -6.29803732e+00),(-3.40006317e+00, -6.15852464e+00),(-3.10050879e+00, -6.38542099e+00),(-3.42458703e+00, -6.09975411e+00),(-3.15185684e+00, -6.34645060e+00),(-3.41620493e+00, -6.17360898e+00),(-3.11783540e+00, -6.38379493e+00),(-3.37196201e+00, -6.15210085e+00),(-3.12552646e+00, -6.40324125e+00),(-3.34379943e+00, -6.17173073e+00),(-3.12386879e+00, -6.41041892e+00),(-3.42036446e+00, -6.15065475e+00),(-3.17565337e+00, -6.37507242e+00),(-3.34400065e+00, -6.07654453e+00),(-3.18730520e+00, -6.27144977e+00),(-3.42480320e+00, -6.05896351e+00),(-3.13543954e+00, -6.30291766e+00),(-3.35493380e+00, -6.00992289e+00),(-3.17147966e+00, -6.30258810e+00),(-3.29832644e+00, -6.05591876e+00),(-3.18893996e+00, -6.36219752e+00),(-3.36850292e+00, -5.99325129e+00),(-3.12820066e+00, -6.33315945e+00),(-3.38011176e+00, -6.04724835e+00),(-3.17535273e+00, -6.34315085e+00),(-3.39725580e+00, -6.10228769e+00),(-3.14016468e+00, -6.38671303e+00),(-3.39644495e+00, -6.11242701e+00),(-3.14499946e+00, -6.35949907e+00),(-5.14207513e-01, -2.29233120e+00),( 2.66805373e+00, -9.39675889e-01),( 2.58495139e+00, -1.03527377e+00),( 2.38211162e+00, -8.12521557e-01),( 2.63487545e+00, -1.03273271e+00),( 2.63233634e+00, -8.62988696e-01),( 2.54894234e+00, -8.15473119e-01),( 2.54096266e+00, -9.26069511e-01),( 2.56232329e+00, -1.02921143e+00),( 2.51581566e+00, -9.98875667e-01),( 2.50547089e+00, -7.77433293e-01),( 2.57516798e+00, -9.86180880e-01),( 2.58948500e+00, -9.03404337e-01),(-9.91685315e-01,  4.21276581e-01),( 2.23567063e+00, -5.67806797e+00),( 2.22446963e+00, -5.71477160e+00),( 2.21362567e+00, -5.68884408e+00),( 2.13918421e+00, -5.71163859e+00),( 1.00478637e-01, -2.58743265e+00),( 4.08093396e-01, -2.32118044e+00),(-2.66168043e-04,  7.08878819e-01),( 1.14421493e-01,  6.91232262e-01),( 2.66194744e-02,  7.57181656e-01),( 6.39760201e-02,  7.19677772e-01),( 6.69643464e-02,  7.73443998e-01),( 6.78640744e-02,  7.50541891e-01),( 1.99655073e+00, -3.73633839e+00),( 2.07366539e+00, -3.59936284e+00),( 2.03282318e+00, -3.61729615e+00),( 2.10223122e+00, -3.53375570e+00),( 2.26641453e+00, -2.76705896e+00),( 2.37842894e+00, -2.71855254e+00),( 2.38533644e+00, -2.51282756e+00),( 2.36536134e+00, -2.58987473e+00),( 2.26015374e-01, -3.00375901e-01),( 6.70975796e-02, -7.76232735e-01),(-2.34751663e+00, -2.73995037e+00),(-2.31938552e+00, -2.71172483e+00),(-2.36328978e+00, -2.80921579e+00),(-2.30138489e+00, -2.84017591e+00),(-2.38368090e+00, -2.70922306e+00),(-2.41901183e+00, -2.78953067e+00),( 2.23413074e+00, -2.67619675e+00),( 2.19122428e+00, -2.68732843e+00),( 2.16633115e+00, -2.67816000e+00),( 4.34967588e+00, -2.35922516e+00),( 3.81832687e+00, -3.84699124e+00),(-5.39236328e+00,  9.47458050e+00),(-5.34643752e+00,  9.45124957e+00),(-5.27490610e+00,  9.46219037e+00),(-5.41959037e+00,  9.45226664e+00),(-5.27217693e+00,  9.39189710e+00),(-5.23241212e+00,  9.36637135e+00),(-5.25693262e+00,  9.35369972e+00),(-5.16643338e+00,  9.34604103e+00),(-5.30228839e+00,  9.44956936e+00),(-5.32402800e+00,  9.41300674e+00),( 2.10860166e+00,  9.64842837e+00),( 2.07586728e+00,  9.74830260e+00),( 2.10471499e+00,  9.70649660e+00),( 2.02717639e+00,  9.77587394e+00),( 1.99536719e+00,  9.79408168e+00),( 2.10172660e+00,  9.72788225e+00),( 1.96985428e+00,  9.71816384e+00),( 2.12989834e+00,  9.71465673e+00),( 2.11572112e+00,  9.70642305e+00),( 1.93438022e+00,  9.77572222e+00),( 2.06344188e+00,  9.68699505e+00),( 2.07783090e+00,  9.71293100e+00),( 2.07331596e+00,  9.76515187e+00),( 2.02175657e+00,  9.72335577e+00),( 2.09104843e+00,  9.75806615e+00),( 2.05157057e+00,  9.78498880e+00),( 2.12017220e+00,  9.71680125e+00),( 1.98476130e+00,  9.78660223e+00),( 2.03538781e+00,  9.79245528e+00),( 2.06419322e+00,  9.76402334e+00),( 2.09422076e+00,  9.73506368e+00),( 2.05663767e+00,  9.71942975e+00),( 1.99129400e+00,  9.78282248e+00),( 2.12828550e+00,  9.68661731e+00),( 2.09466462e+00,  9.71226222e+00),( 2.09018753e+00,  9.68396349e+00),( 2.04639572e+00,  9.73503162e+00),( 2.12538942e+00,  9.66104750e+00),( 2.09569882e+00,  9.67024900e+00),( 2.04677838e+00,  9.76437944e+00),( 2.10338761e+00,  9.74321849e+00),( 2.04303137e+00,  9.78841720e+00),( 2.05897936e+00,  9.72950010e+00),( 2.14972220e+00,  9.69243794e+00),( 2.03390288e+00,  9.69042835e+00),( 1.95885499e+00,  9.81697136e+00),( 2.11253296e+00,  9.71010931e+00),( 1.94597658e+00,  9.80130027e+00),( 2.03121476e+00,  9.76560165e+00),( 2.04644943e+00,  9.71446132e+00),( 2.08832302e+00,  9.73435409e+00),( 2.02705198e+00,  9.79101270e+00),( 2.10588647e+00,  9.76782611e+00),( 2.03520081e+00,  9.73240318e+00),( 2.05992738e+00,  9.77651275e+00),( 2.11668536e+00,  9.64777198e+00),( 2.00956475e+00,  9.72300969e+00),( 2.13652360e+00,  9.68280118e+00),( 2.13550045e+00,  9.69536744e+00),( 1.92216740e+00,  9.72826305e+00),( 6.69063202e-01, -2.05992724e+00),( 6.58307145e-01, -1.93348146e+00),( 8.57720301e-01, -1.82138562e+00),( 8.15451764e-01, -1.76808726e+00),( 8.79334046e-01, -1.79247173e+00),( 8.95969129e-01, -1.77639521e+00),( 8.32961210e-01, -1.78778586e+00),( 8.66559194e-01, -1.71548712e+00),( 8.57894504e-01, -1.79623709e+00),( 8.29254633e-01, -1.73784357e+00),( 8.12716844e-01, -1.73008850e+00),( 8.60502837e-01, -1.78861361e+00),(-2.93265744e-02, -3.38032297e+00),( 3.22323205e-02, -3.18015662e+00),( 8.96819180e-01, -2.60512687e+00),( 9.94682950e-01, -2.43001786e+00),( 9.62103792e-01, -3.37412890e+00),( 8.55058836e-01, -3.61796580e+00),( 1.80603286e+00, -4.10736231e+00),( 1.83444984e+00, -4.05361262e+00),( 1.77384737e+00, -4.08756435e+00),( 1.71384696e+00, -3.97193643e+00),( 1.86989344e+00, -4.08546140e+00)]

dsWi_transformed = embedding.fit_transform(dsWi_z,init=np.array(init))

import matplotlib.gridspec as gridspec

ylim  = [-1, 7]
ylim2 = [-10.5, -8]
ylimratio = (ylim[1]-ylim[0])/(ylim2[1]-ylim2[0]+ylim[1]-ylim[0])
ylim2ratio = (ylim2[1]-ylim2[0])/(ylim2[1]-ylim2[0]+ylim[1]-ylim[0])
gs1 = gridspec.GridSpec(2, 1, height_ratios=[ylimratio, ylim2ratio])
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(gs1[0])
#plt.title('b) CMIP6 multidimensional scaling of intermember distances',fontsize=12,fontweight='bold',loc='left') #,pad=14)
ax2 = fig.add_subplot(gs1[1])

ang=np.deg2rad(177.5)
rot = np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
pos = x,y = np.dot(rot,dsWi_transformed.transpose())

def radius_var(data):
    ens_center = np.mean(data,axis=1).reshape(2,1)
    ens_center_rep = np.repeat(ens_center,data.shape[1], axis=1)
    dist = np.linalg.norm(data-ens_center_rep,axis=0)
    return 2.5*np.max(dist)


ax.scatter(x[0:3], y[0:3],s=2,  color="tab:red", label="original") #access-cm2
circle = plt.Circle((np.mean(x[0:3]), np.mean(y[0:3])),radius_var(pos[:,0:3]), color='tab:red',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[3:13], y[3:13],s=2, color="tab:red", label="original") # access_esm1_5
circle = plt.Circle((np.mean(x[3:13]), np.mean(y[3:13])), radius_var(pos[:,3:13]), color='tab:red',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[13], y[13],s=10, color="tab:orange", label="original") #awi

#########################

ax.scatter(x[14:16], y[14:16],s=2, color="tab:cyan", label="original") # cas_esm2_0
circle = plt.Circle((np.mean(x[14:16]), np.mean(y[14:16])), radius_var(pos[:,14:16]), color='tab:cyan',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[16:19], y[16:19],s=2, color="darkgoldenrod", label="original") #cesm_waccm
circle = plt.Circle((np.mean(x[16:19]), np.mean(y[16:19])), radius_var(pos[:,16:19]), color='darkgoldenrod',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[19:24], y[19:24],s=2, color="darkgoldenrod", label="original") #cesm
circle = plt.Circle((np.mean(x[19:24]), np.mean(y[19:24])), radius_var(pos[:,19:24]), color='darkgoldenrod',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[24], y[24],s=10, color="darkgoldenrod", label="original") #CMCC-CM2-SR5

ax.scatter(x[25], y[25],s=10, color="darkgoldenrod", label="original") #CMCC-ESM2

ax.scatter(x[26], y[26],s=10, color="cornflowerblue", label="original") #cnrm_cm6_1_hr

ax.scatter(x[27:33], y[27:33],s=2, color="cornflowerblue", label="original") #cnrm
circle = plt.Circle((np.mean(x[27:33]), np.mean(y[27:33])), radius_var(pos[:,27:33]), color='cornflowerblue',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[33:38], y[33:38],s=2, color="cornflowerblue", label="original") #cnrm_esm
circle = plt.Circle((np.mean(x[33:38]), np.mean(y[33:38])), radius_var(pos[:,33:38]), color='cornflowerblue',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[38:88], y[38:88],s=2, color="dodgerblue", label="original") #canesm5
circle = plt.Circle((np.mean(x[38:88]), np.mean(y[38:88])), radius_var(pos[:,38:88]), color='dodgerblue',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[88], y[88],s=10, color="k", label="original") #esm

ax.scatter(x[89:93], y[89:93],s=2, color="darkgreen", label="original") #ec_earth3_veg
circle = plt.Circle((np.mean(x[89:93]), np.mean(y[89:93])), radius_var(pos[:,89:93]), color='darkgreen',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[93:101], y[93:101],s=2, color="darkgreen", label="original") #ec_earth3
circle = plt.Circle((np.mean(x[93:101]), np.mean(y[93:101])), radius_var(pos[:,93:101]), color='darkgreen',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[101], y[101],s=10, color="maroon", label="original") #fgoals

ax.scatter(x[102:106], y[102:106],s=2, color="maroon", label="original") #fgoals_g3 #106
circle = plt.Circle((np.mean(x[102:106]), np.mean(y[102:106])), radius_var(pos[:,102:106]), color='maroon',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[106], y[106],s=10, color="indigo", label="original") #gfdl_cm4

ax.scatter(x[107], y[107],s=10, color="indigo", label="original") #gfdl_esm

ax.scatter(x[108:114], y[108:114],s=2, color="blueviolet", label="original") #giss_e2_1_g
circle = plt.Circle((np.mean(x[108:114]), np.mean(y[108:114])), radius_var(pos[:,108:114]), color='blueviolet',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[114:118], y[114:118],s=2, color="tab:red", label="original") #hadgem
circle = plt.Circle((np.mean(x[114:118]), np.mean(y[114:118])), radius_var(pos[:,114:118]), color='tab:red',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[118:122], y[118:122],s=2, color="tab:red", label="original") #hadgem
circle = plt.Circle((np.mean(x[118:122]), np.mean(y[118:122])), radius_var(pos[:,118:122]), color='tab:red',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[122], y[122],s=10, color="mediumseagreen", label="original") #inm_48

ax.scatter(x[123], y[123],s=10, color="mediumseagreen", label="original") #inm_50

ax.scatter(x[124:130], y[124:130],s=2, color="royalblue", label="original") #ipsl_cm6a_lr
circle = plt.Circle((np.mean(x[124:130]), np.mean(y[124:130])), radius_var(pos[:,124:130]), color='royalblue',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[130:133], y[130:133],s=2, color="tab:red", label="original") #kace
circle = plt.Circle((np.mean(x[130:133]), np.mean(y[130:133])), radius_var(pos[:,130:133]), color='tab:red',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[133], y[133],s=10, color="darkslateblue", label="original") #kiost

ax.scatter(x[134], y[134],s=10, color="tab:olive", label="original") #mcm

#########################

ax.scatter(x[135:145], y[135:145],s=2, color="lightsalmon", label="original") #miroc_es2l
circle = plt.Circle((np.mean(x[135:145]), np.mean(y[135:145])), radius_var(pos[:,135:145]), color='lightsalmon',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[145:195], y[145:195],s=2, color="lightsalmon", label="original") #miroc6
circle = plt.Circle((np.mean(x[145:195]), np.mean(y[145:195])), radius_var(pos[:,145:195]), color='lightsalmon',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[195:197], y[195:197],s=2, color="tab:orange", label="original") #mpi_hr
circle = plt.Circle((np.mean(x[195:197]), np.mean(y[195:197])), radius_var(pos[:,195:197]), color='tab:orange',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[197:207], y[197:207],s=2,  color="tab:orange", label="original") #mpi
circle = plt.Circle((np.mean(x[197:207]), np.mean(y[197:207])), radius_var(pos[:,197:207]), color='tab:orange',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[207:209], y[207:209],s=2, color="palevioletred", label="original") #mri_esm2_0
circle = plt.Circle((np.mean(x[207:209]), np.mean(y[207:209])), radius_var(pos[:,207:209]), color='palevioletred',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[209:211], y[209:211],s=2, color="tab:orange", label="original") #nesm
circle = plt.Circle((np.mean(x[209:211]), np.mean(y[209:211])), radius_var(pos[:,209:211]), color='tab:orange',alpha=0.2)
ax.add_patch(circle)

#########################

ax.scatter(x[211], y[211],s=10, color="darkgoldenrod", label="original") #noresm

ax.scatter(x[212], y[212],s=10, color="darkgoldenrod", label="original") #taiesm

ax.scatter(x[213:218], y[213:218],s=2, color="tab:red", label="original") #ukesm
circle = plt.Circle((np.mean(x[213:218]), np.mean(y[213:218])), radius_var(pos[:,213:218]), color='tab:red',alpha=0.2)
ax.add_patch(circle)

#########################

ax2.scatter(x[0:3], y[0:3],s=2,  color="tab:red", label="original") #access-cm2
circle = plt.Circle((np.mean(x[0:3]), np.mean(y[0:3])),radius_var(pos[:,0:3]), color='tab:red',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[3:13], y[3:13],s=2, color="tab:red", label="original") # access_esm1_5
circle = plt.Circle((np.mean(x[3:13]), np.mean(y[3:13])), radius_var(pos[:,3:13]), color='tab:red',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[13], y[13],s=10, color="tab:orange", label="original") #awi

ax2.scatter(x[14:16], y[14:16],s=2, color="tab:cyan", label="original") # cas_esm2_0
circle = plt.Circle((np.mean(x[14:16]), np.mean(y[14:16])), radius_var(pos[:,14:16]), color='tab:cyan',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[16:19], y[16:19],s=2, color="darkgoldenrod", label="original") #cesm_waccm
circle = plt.Circle((np.mean(x[16:19]), np.mean(y[16:19])), radius_var(pos[:,16:19]), color='darkgoldenrod',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[19:24], y[19:24],s=2, color="darkgoldenrod", label="original") #cesm
circle = plt.Circle((np.mean(x[19:24]), np.mean(y[19:24])), radius_var(pos[:,19:24]), color='darkgoldenrod',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[24], y[24],s=10, color="darkgoldenrod", label="original") #CMCC-CM2-SR5

ax2.scatter(x[25], y[25],s=10, color="darkgoldenrod", label="original") #CMCC-ESM2

ax2.scatter(x[26], y[26],s=10, color="cornflowerblue", label="original") #cnrm_cm6_1_hr

ax2.scatter(x[27:33], y[27:33],s=2, color="cornflowerblue", label="original") #cnrm
circle = plt.Circle((np.mean(x[27:33]), np.mean(y[27:33])), radius_var(pos[:,27:33]), color='cornflowerblue',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[33:38], y[33:38],s=2, color="cornflowerblue", label="original") #cnrm_esm
circle = plt.Circle((np.mean(x[33:38]), np.mean(y[33:38])), radius_var(pos[:,33:38]), color='cornflowerblue',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[38:88], y[38:88],s=2, color="dodgerblue", label="original") #canesm5
circle = plt.Circle((np.mean(x[38:88]), np.mean(y[38:88])), radius_var(pos[:,38:88]), color='dodgerblue',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[88], y[88],s=10, color="k", label="original") #esm

ax2.scatter(x[89:93], y[89:93],s=2, color="darkgreen", label="original") #ec_earth3_veg
circle = plt.Circle((np.mean(x[89:93]), np.mean(y[89:93])), radius_var(pos[:,89:93]), color='darkgreen',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[93:101], y[93:101],s=2, color="darkgreen", label="original") #ec_earth3
circle = plt.Circle((np.mean(x[93:101]), np.mean(y[93:101])), radius_var(pos[:,93:101]), color='darkgreen',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[101], y[101],s=10, color="maroon", label="original") #fgoals

ax2.scatter(x[102:106], y[102:106],s=2, color="maroon", label="original") #fgoals_g3 #106
circle = plt.Circle((np.mean(x[102:106]), np.mean(y[102:106])), radius_var(pos[:,102:106]), color='maroon',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[106], y[106],s=10, color="indigo", label="original") #gfdl_cm4

ax2.scatter(x[107], y[107],s=10, color="indigo", label="original") #gfdl_esm

ax2.scatter(x[108:114], y[108:114],s=2, color="blueviolet", label="original") #giss_e2_1_g
circle = plt.Circle((np.mean(x[108:114]), np.mean(y[108:114])), radius_var(pos[:,108:114]), color='blueviolet',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[114:118], y[114:118],s=2, color="tab:red", label="original") #hadgem
circle = plt.Circle((np.mean(x[114:118]), np.mean(y[114:118])), radius_var(pos[:,114:118]), color='tab:red',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[118:122], y[118:122],s=2, color="tab:red", label="original") #hadgem
circle = plt.Circle((np.mean(x[118:122]), np.mean(y[118:122])), radius_var(pos[:,118:122]), color='tab:red',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[122], y[122],s=10, color="mediumseagreen", label="original") #inm_48

ax2.scatter(x[123], y[123],s=10, color="mediumseagreen", label="original") #inm_50

ax2.scatter(x[124:130], y[124:130],s=2, color="royalblue", label="original") #ipsl_cm6a_lr
circle = plt.Circle((np.mean(x[124:130]), np.mean(y[124:130])), radius_var(pos[:,124:130]), color='royalblue',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[130:133], y[130:133],s=2, color="tab:red", label="original") #kace
circle = plt.Circle((np.mean(x[130:133]), np.mean(y[130:133])), radius_var(pos[:,130:133]), color='tab:red',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[133], y[133],s=10, color="darkslateblue", label="original") #kiost

ax2.scatter(x[134], y[134],s=10, color="tab:olive", label="original") #mcm

ax2.scatter(x[135:145], y[135:145],s=2, color="lightsalmon", label="original") #miroc_es2l
circle = plt.Circle((np.mean(x[135:145]), np.mean(y[135:145])), radius_var(pos[:,135:145]), color='lightsalmon',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[145:195], y[145:195],s=2, color="lightsalmon", label="original") #miroc6
circle = plt.Circle((np.mean(x[145:195]), np.mean(y[145:195])), radius_var(pos[:,145:195]), color='lightsalmon',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[195:197], y[195:197],s=2, color="tab:orange", label="original") #mpi_hr
circle = plt.Circle((np.mean(x[195:197]), np.mean(y[195:197])), radius_var(pos[:,195:197]), color='tab:orange',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[197:207], y[197:207],s=2,  color="tab:orange", label="original") #mpi
circle = plt.Circle((np.mean(x[197:207]), np.mean(y[197:207])), radius_var(pos[:,197:207]), color='tab:orange',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[207:209], y[207:209],s=2, color="palevioletred", label="original") #mri_esm2_0
circle = plt.Circle((np.mean(x[207:209]), np.mean(y[207:209])), radius_var(pos[:,207:209]), color='palevioletred',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[209:211], y[209:211],s=2, color="tab:orange", label="original") #nesm
circle = plt.Circle((np.mean(x[209:211]), np.mean(y[209:211])), radius_var(pos[:,209:211]), color='tab:orange',alpha=0.2)
ax2.add_patch(circle)

#########################

ax2.scatter(x[211], y[211],s=10, color="darkgoldenrod", label="original") #noresm

ax2.scatter(x[212], y[212],s=10, color="darkgoldenrod", label="original") #taiesm

ax2.scatter(x[213:218], y[213:218],s=2, color="tab:red", label="original") #ukesm
circle = plt.Circle((np.mean(x[213:218]), np.mean(y[213:218])), radius_var(pos[:,213:218]), color='tab:red',alpha=0.2)
ax2.add_patch(circle)

ax.set_ylim(ylim)
ax2.set_ylim(ylim2)

ax.set_xlim([-6,6])
ax2.set_xlim([-6,6])
plt.subplots_adjust(hspace=0.03)

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')
ax2.xaxis.tick_bottom()

ax2.set_xlabel('Coordinate 1')
ax2.set_ylabel('Coordinate 2')
ax2.yaxis.set_label_coords(0.05, 0.5, transform=fig.transFigure)

kwargs = dict(color='k', clip_on=False)
xlim = ax.get_xlim()
dx = .02*(xlim[1]-xlim[0])
dy = .01*(ylim[1]-ylim[0])/ylimratio
ax.plot((xlim[0]-dx,xlim[0]+dx), (ylim[0]-dy,ylim[0]+dy), **kwargs)
ax.plot((xlim[1]-dx,xlim[1]+dx), (ylim[0]-dy,ylim[0]+dy), **kwargs)
dy = .01*(ylim2[1]-ylim2[0])/ylim2ratio
ax2.plot((xlim[0]-dx,xlim[0]+dx), (ylim2[1]-dy,ylim2[1]+dy), **kwargs)
ax2.plot((xlim[1]-dx,xlim[1]+dx), (ylim2[1]-dy,ylim2[1]+dy), **kwargs)
ax.set_xlim(xlim)
ax2.set_xlim(xlim)

yticks = np.arange(0,8,1)
ax.set_yticks(yticks)

yticks2 = [-9,-10]
ax2.set_yticks(yticks2)

plt.savefig('Fig3_CMIP6_projections_all.png',bbox_inches='tight',dpi=300)

#############################################
# CMIP5
#############################################

dirT5 = '/**/CMIP_subselection/Data/'

dsT5 = xr.open_dataset(dirT5 + 'tas_mon_CMIP5_rcp85_g025_v2_ann.nc',use_cftime = True)
dsT5 = dsT5-273.15
dsT5 = dsT5.sortby(dsT5.member)

dsP5 = xr.open_dataset(dirT5 + 'psl_mon_CMIP5_rcp85_g025_v2_ann.nc',use_cftime = True)
dsP5 = dsP5/100
dsP5 = dsP5.sortby(dsP5.member)

dsT5_clim = dsT5.sel(year=slice(1905, 2005)).mean('year') #### 1950,2014
dsP5_clim = dsP5.sel(year=slice(1905, 2005)).mean('year') #### 1950,2014

dsT5_clim = dsT5_clim.sel(member=['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1', 'BNU-ESM-r1i1p1',
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

dsP5_clim = dsP5_clim.sel(member=['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1', 'BNU-ESM-r1i1p1',
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

##########################################################

dsT5_clim_can_std = dsT5_clim.sel(member=['CanESM2-r1i1p1',
'CanESM2-r2i1p1', 'CanESM2-r3i1p1', 'CanESM2-r4i1p1', 'CanESM2-r5i1p1']).std('member')

dsT5_clim_csiro_std = dsT5_clim.sel(member=['CSIRO-Mk3-6-0-r10i1p1', 'CSIRO-Mk3-6-0-r1i1p1',
'CSIRO-Mk3-6-0-r2i1p1', 'CSIRO-Mk3-6-0-r3i1p1', 'CSIRO-Mk3-6-0-r4i1p1',
'CSIRO-Mk3-6-0-r5i1p1', 'CSIRO-Mk3-6-0-r6i1p1', 'CSIRO-Mk3-6-0-r7i1p1',
'CSIRO-Mk3-6-0-r8i1p1', 'CSIRO-Mk3-6-0-r9i1p1']).std('member')

dsT5_clim_ccsm4_std = dsT5_clim.sel(member=['CCSM4-r1i1p1', 'CCSM4-r2i1p1', 'CCSM4-r3i1p1', 'CCSM4-r4i1p1',
'CCSM4-r5i1p1', 'CCSM4-r6i1p1']).std('member')

dsT5_clim_cnrm_std = dsT5_clim.sel(member=['CNRM-CM5-r10i1p1',
'CNRM-CM5-r1i1p1', 'CNRM-CM5-r2i1p1', 'CNRM-CM5-r4i1p1',
'CNRM-CM5-r6i1p1']).std('member')

dsT5_clim_earth_std = dsT5_clim.sel(member=['EC-EARTH-r12i1p1', 'EC-EARTH-r1i1p1', 'EC-EARTH-r2i1p1',
'EC-EARTH-r8i1p1', 'EC-EARTH-r9i1p1']).std('member')

dsT5_clim_iv_std = xr.concat([dsT5_clim_can_std,dsT5_clim_csiro_std,
dsT5_clim_ccsm4_std,dsT5_clim_cnrm_std,dsT5_clim_earth_std],dim='member')

dsT5_clim_iv_std_mean = dsT5_clim_iv_std.median('member')

dsT5_clim_one_std = dsT5_clim.sel(member=['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1', 'BNU-ESM-r1i1p1',
'CCSM4-r1i1p1', 'CESM1-CAM5-r1i1p1','CNRM-CM5-r1i1p1', 'CSIRO-Mk3-6-0-r1i1p1',
'CanESM2-r1i1p1','EC-EARTH-r1i1p1', 'FGOALS-g2-r1i1p1',
'GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1', 'GFDL-ESM2M-r1i1p1',
'GISS-E2-H-r1i1p1', 'GISS-E2-R-r1i1p1','HadGEM2-ES-r1i1p1', 'IPSL-CM5A-LR-r1i1p1',
'IPSL-CM5A-MR-r1i1p1', 'IPSL-CM5B-LR-r1i1p1', 'MIROC-ESM-r1i1p1',
'MIROC5-r1i1p1', 'MPI-ESM-LR-r1i1p1', 'MPI-ESM-MR-r1i1p1',
'MRI-CGCM3-r1i1p1', 'NorESM1-ME-r1i1p1', 'NorESM1-M-r1i1p1',
'bcc-csm1-1-m-r1i1p1', 'bcc-csm1-1-r1i1p1', 'inmcm4-r1i1p1']).std('member') #

a = np.percentile(dsT5_clim_one_std.tas,15)
n = np.percentile(dsT5_clim_iv_std_mean.tas,85)

mask_obs = dsT5_clim_one_std.where(dsT5_clim_one_std>a)/dsT5_clim_one_std.where(dsT5_clim_one_std>a)

mask_obs_1 = mask_obs.sel(lon=np.arange(1.25,181.25,2.5))
mask_obs_2 = mask_obs.sel(lon=np.arange(181.25,360,2.5))
mask_obs_2['lon'] = mask_obs_2.lon - 360
mask_obs_flip = xr.concat([mask_obs_2,mask_obs_1],dim='lon')

mask_obs_T5 = np.nan_to_num(mask_obs_flip.tas)

mask_obs13 = dsT5_clim_iv_std_mean.where(dsT5_clim_iv_std_mean<n)/dsT5_clim_iv_std_mean.where(dsT5_clim_iv_std_mean<n)

mask_obs13_1 = mask_obs13.sel(lon=np.arange(1.25,181.25,2.5))
mask_obs13_2 = mask_obs13.sel(lon=np.arange(181.25,360,2.5))
mask_obs13_2['lon'] = mask_obs13_2.lon - 360
mask_obs13_flip = xr.concat([mask_obs13_2,mask_obs13_1],dim='lon')

mask_obs13_T5 = np.nan_to_num(mask_obs13_flip.tas)

dsT5_clim_mask = mask_obs * mask_obs13 * dsT5_clim.squeeze(drop=True)

############
dsP5_clim_can_std = dsP5_clim.sel(member=['CanESM2-r1i1p1',
'CanESM2-r2i1p1', 'CanESM2-r3i1p1', 'CanESM2-r4i1p1', 'CanESM2-r5i1p1']).std('member')

dsP5_clim_csiro_std = dsP5_clim.sel(member=['CSIRO-Mk3-6-0-r10i1p1', 'CSIRO-Mk3-6-0-r1i1p1',
'CSIRO-Mk3-6-0-r2i1p1', 'CSIRO-Mk3-6-0-r3i1p1', 'CSIRO-Mk3-6-0-r4i1p1',
'CSIRO-Mk3-6-0-r5i1p1', 'CSIRO-Mk3-6-0-r6i1p1', 'CSIRO-Mk3-6-0-r7i1p1',
'CSIRO-Mk3-6-0-r8i1p1', 'CSIRO-Mk3-6-0-r9i1p1']).std('member')

dsP5_clim_ccsm4_std = dsP5_clim.sel(member=['CCSM4-r1i1p1', 'CCSM4-r2i1p1', 'CCSM4-r3i1p1', 'CCSM4-r4i1p1',
'CCSM4-r5i1p1', 'CCSM4-r6i1p1']).std('member')

dsP5_clim_cnrm_std = dsP5_clim.sel(member=['CNRM-CM5-r10i1p1',
'CNRM-CM5-r1i1p1', 'CNRM-CM5-r2i1p1', 'CNRM-CM5-r4i1p1',
'CNRM-CM5-r6i1p1']).std('member')

dsP5_clim_earth_std = dsP5_clim.sel(member=['EC-EARTH-r12i1p1', 'EC-EARTH-r1i1p1', 'EC-EARTH-r2i1p1',
'EC-EARTH-r8i1p1', 'EC-EARTH-r9i1p1']).std('member')

dsP5_clim_iv_std = xr.concat([dsP5_clim_can_std,dsP5_clim_csiro_std,
dsP5_clim_ccsm4_std,dsP5_clim_cnrm_std,dsP5_clim_earth_std],dim='member')

dsP5_clim_iv_std_mean = dsP5_clim_iv_std.median('member')

dsP5_clim_one_std = dsP5_clim.sel(member=['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1', 'BNU-ESM-r1i1p1',
'CCSM4-r1i1p1', 'CESM1-CAM5-r1i1p1','CNRM-CM5-r1i1p1', 'CSIRO-Mk3-6-0-r1i1p1',
'CanESM2-r1i1p1','EC-EARTH-r1i1p1', 'FGOALS-g2-r1i1p1',
'GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1', 'GFDL-ESM2M-r1i1p1',
'GISS-E2-H-r1i1p1', 'GISS-E2-R-r1i1p1','HadGEM2-ES-r1i1p1', 'IPSL-CM5A-LR-r1i1p1',
'IPSL-CM5A-MR-r1i1p1', 'IPSL-CM5B-LR-r1i1p1', 'MIROC-ESM-r1i1p1',
'MIROC5-r1i1p1', 'MPI-ESM-LR-r1i1p1', 'MPI-ESM-MR-r1i1p1',
'MRI-CGCM3-r1i1p1', 'NorESM1-ME-r1i1p1', 'NorESM1-M-r1i1p1',
'bcc-csm1-1-m-r1i1p1', 'bcc-csm1-1-r1i1p1', 'inmcm4-r1i1p1']).std('member') #


a = np.percentile(dsP5_clim_one_std.psl,15)
n = np.percentile(dsP5_clim_iv_std_mean.psl,85)

mask_obs = dsP5_clim_one_std.where(dsP5_clim_one_std>a)/dsP5_clim_one_std.where(dsP5_clim_one_std>a)

mask_obs_1 = mask_obs.sel(lon=np.arange(1.25,181.25,2.5))
mask_obs_2 = mask_obs.sel(lon=np.arange(181.25,360,2.5))
mask_obs_2['lon'] = mask_obs_2.lon - 360
mask_obs_flip = xr.concat([mask_obs_2,mask_obs_1],dim='lon')

mask_obs_P5 = np.nan_to_num(mask_obs_flip.psl)

mask_obs13 = dsP5_clim_iv_std_mean.where(dsP5_clim_iv_std_mean<n)/dsP5_clim_iv_std_mean.where(dsP5_clim_iv_std_mean<n)

mask_obs13_1 = mask_obs13.sel(lon=np.arange(1.25,181.25,2.5))
mask_obs13_2 = mask_obs13.sel(lon=np.arange(181.25,360,2.5))
mask_obs13_2['lon'] = mask_obs13_2.lon - 360
mask_obs13_flip = xr.concat([mask_obs13_2,mask_obs13_1],dim='lon')

mask_obs13_P5 = np.nan_to_num(mask_obs13_flip.psl)

dsP5_clim_mask = mask_obs * mask_obs13 * dsP5_clim.squeeze(drop=True)

################################
# Compute Independence Matrix

weights = [np.cos(np.deg2rad(dsT5_clim.lat))]*144
weights = xr.concat(weights, "lon")
weights['lon'] = dsT5_clim.lon

def get_error(ds,weights):
    mod_coords = ds.member.values
    nmod = len(mod_coords)
    res = xr.DataArray(np.empty(shape=(nmod, nmod)),
                        dims=("member1", "member2"), coords=dict(member1=mod_coords, member2=mod_coords))

    for mod1 in ds.transpose("member", ...):
        for mod2 in ds.transpose("member", ...):
            a = xskillscore.rmse(mod1,mod2,dim=['lat','lon'],weights=weights,skipna=True)
            res.loc[dict(member1=mod1.member, member2=mod2.member)] = a

    return res

dsT5_clim_test = get_error(dsT5_clim_mask.tas,weights)
dsT5_clim_test = dsT5_clim_test.where(dsT5_clim_test!=0)
dsT5_clim_test_norm = dsT5_clim_test/np.nanmean(dsT5_clim_test)

dsP5_clim_test = get_error(dsP5_clim_mask.psl,weights)
dsP5_clim_test = dsP5_clim_test.where(dsP5_clim_test!=0)
dsP5_clim_test_norm = dsP5_clim_test/np.nanmean(dsP5_clim_test)

dsWi = (dsT5_clim_test_norm + dsP5_clim_test_norm)/2 #############

##################################
import itertools

ind = itertools.combinations(list(range(len(dsWi))), 3)

for a,b,c in ind:
    dist_ab = dsWi.isel(member1=a,member2=b).data
    dist_ac = dsWi.isel(member1=a,member2=c).data
    dist_bc = dsWi.isel(member1=b,member2=c).data
    if dist_ab > dist_ac+dist_bc:
        raise RuntimeError('Triangular Criterion Failed')


#####################################

ec_fam = ['EC-EARTH-r12i1p1', 'EC-EARTH-r1i1p1','EC-EARTH-r2i1p1', 'EC-EARTH-r8i1p1', 'EC-EARTH-r9i1p1']
# closest relative: CNRM-CM5

dsWi_can_fam = dsWi.sel(member1=ec_fam,member2=ec_fam)
ec_fam_all = np.unique(dsWi_can_fam)
ec_fam_all = ec_fam_all[~np.isnan(ec_fam_all)]

dsWi_can_rest = dsWi.sel(member1=ec_fam).drop_sel(member2=ec_fam)
ec_rest_all = np.unique(dsWi_can_rest)
ec_rest_all = ec_rest_all[~np.isnan(ec_rest_all)]

# ##
cnrm_full_fam = ['CNRM-CM5-r10i1p1','CNRM-CM5-r1i1p1', 'CNRM-CM5-r2i1p1', 'CNRM-CM5-r4i1p1','CNRM-CM5-r6i1p1']

dsWi_ec_fam_rest = dsWi.sel(member1=ec_fam,member2=cnrm_full_fam)
ec_fam_rest_all = np.unique(dsWi_ec_fam_rest)
ec_fam_rest_all = ec_fam_rest_all[~np.isnan(ec_fam_rest_all)]

#####################################

can_fam = ['CanESM2-r1i1p1','CanESM2-r2i1p1', 'CanESM2-r3i1p1', 'CanESM2-r4i1p1','CanESM2-r5i1p1']
# closest relative: MPI

dsWi_can_fam = dsWi.sel(member1=can_fam,member2=can_fam)
can_fam_all = np.unique(dsWi_can_fam)
can_fam_all = can_fam_all[~np.isnan(can_fam_all)]

dsWi_can_rest = dsWi.sel(member1=can_fam).drop_sel(member2=can_fam)
can_rest_all = np.unique(dsWi_can_rest)
can_rest_all = can_rest_all[~np.isnan(can_rest_all)]

#####################################

cnrm_fam = ['CNRM-CM5-r10i1p1','CNRM-CM5-r1i1p1', 'CNRM-CM5-r2i1p1', 'CNRM-CM5-r4i1p1','CNRM-CM5-r6i1p1']
# closest relative: EC-EARTH

dsWi_cnrm_fam = dsWi.sel(member1=cnrm_fam,member2=cnrm_fam)
cnrm_fam_all = np.unique(dsWi_cnrm_fam)
cnrm_fam_all = cnrm_fam_all[~np.isnan(cnrm_fam_all)]

dsWi_cnrm_rest = dsWi.sel(member1=cnrm_fam).drop_sel(member2=cnrm_fam)
cnrm_rest_all = np.unique(dsWi_cnrm_rest)
cnrm_rest_all = cnrm_rest_all[~np.isnan(cnrm_rest_all)]

# ##
cnrm_full_fam = ['EC-EARTH-r12i1p1', 'EC-EARTH-r1i1p1','EC-EARTH-r2i1p1', 'EC-EARTH-r8i1p1', 'EC-EARTH-r9i1p1']

dsWi_cnrm_fam_rest = dsWi.sel(member1=cnrm_fam,member2=cnrm_full_fam)
cnrm_fam_rest_all = np.unique(dsWi_cnrm_fam_rest)
cnrm_fam_rest_all = cnrm_fam_rest_all[~np.isnan(cnrm_fam_rest_all)]


#####################################

csiro_fam = ['CSIRO-Mk3-6-0-r10i1p1', 'CSIRO-Mk3-6-0-r1i1p1','CSIRO-Mk3-6-0-r2i1p1',
'CSIRO-Mk3-6-0-r3i1p1','CSIRO-Mk3-6-0-r4i1p1', 'CSIRO-Mk3-6-0-r5i1p1',
'CSIRO-Mk3-6-0-r6i1p1', 'CSIRO-Mk3-6-0-r7i1p1',
'CSIRO-Mk3-6-0-r8i1p1', 'CSIRO-Mk3-6-0-r9i1p1']

dsWi_csiro_fam = dsWi.sel(member1=csiro_fam,member2=csiro_fam)
csiro_fam_all = np.unique(dsWi_csiro_fam)
csiro_fam_all = csiro_fam_all[~np.isnan(csiro_fam_all)]

dsWi_csiro_rest = dsWi.sel(member1=csiro_fam).drop_sel(member2=csiro_fam)
csiro_rest_all = np.unique(dsWi_csiro_rest)
csiro_rest_all = csiro_rest_all[~np.isnan(csiro_rest_all)]

#####################################

ccsm4_fam = ['CCSM4-r1i1p1', 'CCSM4-r2i1p1',
'CCSM4-r3i1p1', 'CCSM4-r4i1p1','CCSM4-r5i1p1',
'CCSM4-r6i1p1']

dsWi_ccsm4_fam = dsWi.sel(member1=ccsm4_fam,member2=ccsm4_fam)
ccsm4_fam_all = np.unique(dsWi_ccsm4_fam)
ccsm4_fam_all = ccsm4_fam_all[~np.isnan(ccsm4_fam_all)]

dsWi_ccsm4_rest = dsWi.sel(member1=ccsm4_fam).drop_sel(member2=ccsm4_fam)
ccsm4_rest_all = np.unique(dsWi_ccsm4_rest)
ccsm4_rest_all = ccsm4_rest_all[~np.isnan(ccsm4_rest_all)]

#####################################

ipsl_fam = ['IPSL-CM5A-LR-r1i1p1', 'IPSL-CM5A-LR-r2i1p1',
'IPSL-CM5A-LR-r3i1p1', 'IPSL-CM5A-LR-r4i1p1']

dsWi_ipsl_fam = dsWi.sel(member1=ipsl_fam,member2=ipsl_fam)
ipsl_fam_all = np.unique(dsWi_ipsl_fam)
ipsl_fam_all = ipsl_fam_all[~np.isnan(ipsl_fam_all)]

dsWi_ipsl_rest = dsWi.sel(member1=ipsl_fam).drop_sel(member2=ipsl_fam)
ipsl_rest_all = np.unique(dsWi_ipsl_rest)
ipsl_rest_all = ipsl_rest_all[~np.isnan(ipsl_rest_all)]

# ##
ipsl_full_fam = ['IPSL-CM5A-MR-r1i1p1']

dsWi_ipsl_fam_rest = dsWi.sel(member1=ipsl_fam,member2=ipsl_full_fam)
ipsl_fam_rest_all = np.unique(dsWi_ipsl_fam_rest)
ipsl_fam_rest_all = ipsl_fam_rest_all[~np.isnan(ipsl_fam_rest_all)]



#####################################

miroc_fam = ['MIROC5-r1i1p1', 'MIROC5-r2i1p1', 'MIROC5-r3i1p1']

dsWi_miroc_fam = dsWi.sel(member1=miroc_fam,member2=miroc_fam)
miroc_fam_all = np.unique(dsWi_miroc_fam)
miroc_fam_all = miroc_fam_all[~np.isnan(miroc_fam_all)]

dsWi_miroc_rest = dsWi.sel(member1=miroc_fam).drop_sel(member2=miroc_fam)
miroc_rest_all = np.unique(dsWi_miroc_rest)
miroc_rest_all = miroc_rest_all[~np.isnan(miroc_rest_all)]

#####################################

cesm1_fam = ['CESM1-CAM5-r1i1p1','CESM1-CAM5-r2i1p1', 'CESM1-CAM5-r3i1p1']

dsWi_cesm1_fam = dsWi.sel(member1=cesm1_fam,member2=cesm1_fam)
cesm1_fam_all = np.unique(dsWi_cesm1_fam)
cesm1_fam_all = cesm1_fam_all[~np.isnan(cesm1_fam_all)]

dsWi_cesm1_rest = dsWi.sel(member1=cesm1_fam).drop_sel(member2=cesm1_fam)
cesm1_rest_all = np.unique(dsWi_cesm1_rest)
cesm1_rest_all = cesm1_rest_all[~np.isnan(cesm1_rest_all)]

#####################################

giss_h_fam = ['GISS-E2-H-r1i1p1', 'GISS-E2-H-r1i1p2',
'GISS-E2-H-r1i1p3', 'GISS-E2-H-r2i1p1', 'GISS-E2-H-r2i1p3']

dsWi_giss_h_fam = dsWi.sel(member1=giss_h_fam,member2=giss_h_fam)
giss_h_fam_all = np.unique(dsWi_giss_h_fam)
giss_h_fam_all = giss_h_fam_all[~np.isnan(giss_h_fam_all)]

dsWi_giss_h_rest = dsWi.sel(member1=giss_h_fam).drop_sel(member2=giss_h_fam)
giss_h_rest_all = np.unique(dsWi_giss_h_rest)
giss_h_rest_all = giss_h_rest_all[~np.isnan(giss_h_rest_all)]

# ##
giss_full_fam = ['GISS-E2-R-r1i1p1', 'GISS-E2-R-r1i1p2', 'GISS-E2-R-r1i1p3',
'GISS-E2-R-r2i1p1', 'GISS-E2-R-r2i1p3']

dsWi_giss_h_fam_rest = dsWi.sel(member1=giss_h_fam,member2=giss_full_fam)
giss_h_fam_rest_all = np.unique(dsWi_giss_h_fam_rest)
giss_h_fam_rest_all = giss_h_fam_rest_all[~np.isnan(giss_h_fam_rest_all)]


giss_r_fam = ['GISS-E2-R-r1i1p1', 'GISS-E2-R-r1i1p2', 'GISS-E2-R-r1i1p3',
'GISS-E2-R-r2i1p1', 'GISS-E2-R-r2i1p3']

dsWi_giss_r_fam = dsWi.sel(member1=giss_r_fam,member2=giss_r_fam)
giss_r_fam_all = np.unique(dsWi_giss_r_fam)
giss_r_fam_all = giss_r_fam_all[~np.isnan(giss_r_fam_all)]

dsWi_giss_r_rest = dsWi.sel(member1=giss_r_fam).drop_sel(member2=giss_r_fam)
giss_r_rest_all = np.unique(dsWi_giss_r_rest)
giss_r_rest_all = giss_r_rest_all[~np.isnan(giss_r_rest_all)]

# ##
giss_full_fam = ['GISS-E2-H-r1i1p1', 'GISS-E2-H-r1i1p2',
'GISS-E2-H-r1i1p3', 'GISS-E2-H-r2i1p1', 'GISS-E2-H-r2i1p3']

dsWi_giss_r_fam_rest = dsWi.sel(member1=giss_r_fam,member2=giss_full_fam)
giss_r_fam_rest_all = np.unique(dsWi_giss_r_fam_rest)
giss_r_fam_rest_all = giss_r_fam_rest_all[~np.isnan(giss_r_fam_rest_all)]

#####################################

mpi_fam = ['MPI-ESM-LR-r1i1p1', 'MPI-ESM-LR-r2i1p1', 'MPI-ESM-LR-r3i1p1']

dsWi_mpi_fam = dsWi.sel(member1=mpi_fam,member2=mpi_fam)
mpi_fam_all = np.unique(dsWi_mpi_fam)
mpi_fam_all = mpi_fam_all[~np.isnan(mpi_fam_all)]

dsWi_mpi_rest = dsWi.sel(member1=mpi_fam).drop_sel(member2=mpi_fam)
mpi_rest_all = np.unique(dsWi_mpi_rest)
mpi_rest_all = mpi_rest_all[~np.isnan(mpi_rest_all)]

# ##
mpi_full_fam = ['MPI-ESM-MR-r1i1p1']

dsWi_mpi_fam_rest = dsWi.sel(member1=mpi_fam,member2=mpi_full_fam)
mpi_fam_rest_all = np.unique(dsWi_mpi_fam_rest)
mpi_fam_rest_all = mpi_fam_rest_all[~np.isnan(mpi_fam_rest_all)]


#####################################

hadgem_fam = ['HadGEM2-ES-r1i1p1','HadGEM2-ES-r2i1p1', 'HadGEM2-ES-r3i1p1', 'HadGEM2-ES-r4i1p1']

dsWi_hadgem_fam = dsWi.sel(member1=hadgem_fam,member2=hadgem_fam)
hadgem_fam_all = np.unique(dsWi_hadgem_fam)
hadgem_fam_all = hadgem_fam_all[~np.isnan(hadgem_fam_all)]

dsWi_hadgem_rest = dsWi.sel(member1=hadgem_fam).drop_sel(member2=hadgem_fam)
hadgem_rest_all = np.unique(dsWi_hadgem_rest)
hadgem_rest_all = hadgem_rest_all[~np.isnan(hadgem_rest_all)]

# ##
had_full_fam = ['ACCESS1-0-r1i1p1','ACCESS1-3-r1i1p1']

dsWi_hadgem_fam_rest = dsWi.sel(member1=hadgem_fam,member2=had_full_fam)
hadgem_fam_rest_all = np.unique(dsWi_hadgem_fam_rest)
hadgem_fam_rest_all = hadgem_fam_rest_all[~np.isnan(hadgem_fam_rest_all)]


#####################################

nor_me_fam = ['NorESM1-ME-r1i1p1']

dsWi_nor_me_rest = dsWi.sel(member1=nor_me_fam).drop_sel(member2=nor_me_fam)
nor_me_rest_all = np.unique(dsWi_nor_me_rest)
nor_me_rest_all = nor_me_rest_all[~np.isnan(nor_me_rest_all)]

nor_m_fam = ['NorESM1-M-r1i1p1']

dsWi_nor_m_rest = dsWi.sel(member1=nor_m_fam).drop_sel(member2=nor_m_fam)
nor_m_rest_all = np.unique(dsWi_nor_m_rest)
nor_m_rest_all = nor_m_rest_all[~np.isnan(nor_m_rest_all)]

#####################################

gfdlesm_g_fam = ['GFDL-ESM2G-r1i1p1']

dsWi_gfdlesm_g_rest = dsWi.sel(member1=gfdlesm_g_fam).drop_sel(member2=gfdlesm_g_fam)
gfdlesm_g_rest_all = np.unique(dsWi_gfdlesm_g_rest)
gfdlesm_g_rest_all = gfdlesm_g_rest_all[~np.isnan(gfdlesm_g_rest_all)]

gfdlesm_m_fam = ['GFDL-ESM2M-r1i1p1']

dsWi_gfdlesm_m_rest = dsWi.sel(member1=gfdlesm_m_fam).drop_sel(member2=gfdlesm_m_fam)
gfdlesm_m_rest_all = np.unique(dsWi_gfdlesm_m_rest)
gfdlesm_m_rest_all = gfdlesm_m_rest_all[~np.isnan(gfdlesm_m_rest_all)]

#####################################

bcc_fam = ['bcc-csm1-1-r1i1p1']

dsWi_bcc_rest = dsWi.sel(member1=bcc_fam).drop_sel(member2=bcc_fam)
bcc_rest_all = np.unique(dsWi_bcc_rest)
bcc_rest_all = bcc_rest_all[~np.isnan(bcc_rest_all)]


bnu_fam = ['BNU-ESM-r1i1p1']

dsWi_bnu_rest = dsWi.sel(member1=bnu_fam).drop_sel(member2=bnu_fam)
bnu_rest_all = np.unique(dsWi_bnu_rest)
bnu_rest_all = bnu_rest_all[~np.isnan(bnu_rest_all)]

#####################################

gfdl_fam = ['GFDL-CM3-r1i1p1']

dsWi_gfdl_rest = dsWi.sel(member1=gfdl_fam).drop_sel(member2=gfdl_fam)
gfdl_rest_all = np.unique(dsWi_gfdl_rest)
gfdl_rest_all = gfdl_rest_all[~np.isnan(gfdl_rest_all)]

#####################################

fgoals_g2_fam = ['FGOALS-g2-r1i1p1']

dsWi_fgoals_g2_rest = dsWi.sel(member1=fgoals_g2_fam).drop_sel(member2=fgoals_g2_fam)
fgoals_g2_rest_all = np.unique(dsWi_fgoals_g2_rest)
fgoals_g2_rest_all = fgoals_g2_rest_all[~np.isnan(fgoals_g2_rest_all)]

#####################################

miroc_esm_fam = ['MIROC-ESM-r1i1p1']

dsWi_miroc_esm_rest = dsWi.sel(member1=miroc_esm_fam).drop_sel(member2=miroc_esm_fam)
miroc_esm_rest_all = np.unique(dsWi_miroc_esm_rest)
miroc_esm_rest_all = miroc_esm_rest_all[~np.isnan(miroc_esm_rest_all)]

#####################################

mri_fam = ['MRI-CGCM3-r1i1p1']

dsWi_mri_rest = dsWi.sel(member1=mri_fam).drop_sel(member2=mri_fam)
mri_rest_all = np.unique(dsWi_mri_rest)
mri_rest_all = mri_rest_all[~np.isnan(mri_rest_all)]

#####################################

inmcm4_fam = ['inmcm4-r1i1p1']

dsWi_inmcm4_rest = dsWi.sel(member1=inmcm4_fam).drop_sel(member2=inmcm4_fam)
inmcm4_rest_all = np.unique(dsWi_inmcm4_rest)
inmcm4_rest_all = inmcm4_rest_all[~np.isnan(inmcm4_rest_all)]

#####################################

bcc_m_fam = ['bcc-csm1-1-m-r1i1p1']

dsWi_bcc_m_rest = dsWi.sel(member1=bcc_m_fam).drop_sel(member2=bcc_m_fam)
bcc_m_rest_all = np.unique(dsWi_bcc_m_rest)
bcc_m_rest_all = bcc_m_rest_all[~np.isnan(bcc_m_rest_all)]

#####################################

ipsl_b_fam = ['IPSL-CM5B-LR-r1i1p1']

dsWi_ipsl_b_rest = dsWi.sel(member1=ipsl_b_fam).drop_sel(member2=ipsl_b_fam)
ipsl_b_rest_all = np.unique(dsWi_ipsl_b_rest)
ipsl_b_rest_all = ipsl_b_rest_all[~np.isnan(ipsl_b_rest_all)]

ipsl_mr_fam = ['IPSL-CM5A-MR-r1i1p1']

dsWi_ipsl_mr_rest = dsWi.sel(member1=ipsl_mr_fam).drop_sel(member2=ipsl_mr_fam)
ipsl_mr_rest_all = np.unique(dsWi_ipsl_mr_rest)
ipsl_mr_rest_all = ipsl_mr_rest_all[~np.isnan(ipsl_mr_rest_all)]

# ##
ipsl_full_fam = ['IPSL-CM5A-LR-r1i1p1', 'IPSL-CM5A-LR-r2i1p1',
'IPSL-CM5A-LR-r3i1p1', 'IPSL-CM5A-LR-r4i1p1']

dsWi_ipsl_mr_fam_rest = dsWi.sel(member1=ipsl_mr_fam,member2=ipsl_full_fam)
ipsl_mr_fam_rest_all = np.unique(dsWi_ipsl_mr_fam_rest)
ipsl_mr_fam_rest_all = ipsl_mr_fam_rest_all[~np.isnan(ipsl_mr_fam_rest_all)]


mpi_mr_fam = ['MPI-ESM-MR-r1i1p1']

dsWi_mpi_mr_rest = dsWi.sel(member1=mpi_mr_fam).drop_sel(member2=mpi_mr_fam)
mpi_mr_rest_all = np.unique(dsWi_mpi_mr_rest)
mpi_mr_rest_all = mpi_mr_rest_all[~np.isnan(mpi_mr_rest_all)]

# ##
mpi_full_fam = ['MPI-ESM-LR-r1i1p1', 'MPI-ESM-LR-r2i1p1', 'MPI-ESM-LR-r3i1p1']

dsWi_mpi_mr_fam_rest = dsWi.sel(member1=mpi_mr_fam,member2=mpi_full_fam)
mpi_mr_fam_rest_all = np.unique(dsWi_mpi_mr_fam_rest)
mpi_mr_fam_rest_all = mpi_mr_fam_rest_all[~np.isnan(mpi_mr_fam_rest_all)]


access_0_fam = ['ACCESS1-0-r1i1p1']

dsWi_access_0_rest = dsWi.sel(member1=access_0_fam).drop_sel(member2=access_0_fam)
access_0_rest_all = np.unique(dsWi_access_0_rest)
access_0_rest_all = access_0_rest_all[~np.isnan(access_0_rest_all)]

# ##
had_full_fam = ['HadGEM2-ES-r1i1p1','HadGEM2-ES-r2i1p1', 'HadGEM2-ES-r3i1p1', 'HadGEM2-ES-r4i1p1','ACCESS1-3-r1i1p1']

dsWi_access_0_fam_rest = dsWi.sel(member1=access_0_fam,member2=had_full_fam)
access_0_fam_rest_all = np.unique(dsWi_access_0_fam_rest)
access_0_fam_rest_all = access_0_fam_rest_all[~np.isnan(access_0_fam_rest_all)]


access_3_fam = ['ACCESS1-3-r1i1p1']

dsWi_access_3_rest = dsWi.sel(member1=access_3_fam).drop_sel(member2=access_3_fam)
access_3_rest_all = np.unique(dsWi_access_3_rest)
access_3_rest_all = access_3_rest_all[~np.isnan(access_3_rest_all)]

# ##
had_full_fam = ['HadGEM2-ES-r1i1p1','HadGEM2-ES-r2i1p1', 'HadGEM2-ES-r3i1p1', 'HadGEM2-ES-r4i1p1','ACCESS1-0-r1i1p1']

dsWi_access_3_fam_rest = dsWi.sel(member1=access_3_fam,member2=had_full_fam)
access_3_fam_rest_all = np.unique(dsWi_access_3_fam_rest)
access_3_fam_rest_all = access_3_fam_rest_all[~np.isnan(access_3_fam_rest_all)]


#####################################

fig = plt.figure(figsize=(8,7))
ax = plt.subplot(111)

##################
plt.plot(access_0_rest_all,0*np.ones(np.size(access_0_rest_all)),'|',color='silver')
plt.plot(access_0_fam_rest_all,0*np.ones(np.size(access_0_fam_rest_all)),'|',color='dimgray')

plt.plot(access_3_rest_all,1*np.ones(np.size(access_3_rest_all)),'|',color='silver')
plt.plot(access_3_fam_rest_all,1*np.ones(np.size(access_3_fam_rest_all)),'|',color='dimgray')

plt.plot(hadgem_fam_all,2*np.ones(np.size(hadgem_fam_all)),'|',color='tab:red')
plt.plot(hadgem_rest_all,2*np.ones(np.size(hadgem_rest_all)),'|',color='silver')
plt.plot(hadgem_fam_rest_all,2*np.ones(np.size(hadgem_fam_rest_all)),'|',color='dimgray')

plt.axhline(3,color='silver')
##################

plt.plot(nor_me_rest_all,4*np.ones(np.size(nor_me_rest_all)),'|',color='silver')
plt.plot(nor_me_rest_all[0],4*np.ones(np.size(nor_me_rest_all[0])),'|',color='dimgray')

#
plt.plot(nor_m_rest_all,5*np.ones(np.size(nor_m_rest_all)),'|',color='silver')
plt.plot(nor_m_rest_all[0],5*np.ones(np.size(nor_m_rest_all[0])),'|',color='dimgray')

plt.plot(ccsm4_fam_all,6*np.ones(np.size(ccsm4_fam_all)),'|',color='darkgoldenrod')
plt.plot(ccsm4_rest_all,6*np.ones(np.size(ccsm4_rest_all)),'|',color='silver')
#
plt.plot(cesm1_fam_all,7*np.ones(np.size(cesm1_fam_all)),'|',color='darkgoldenrod')
plt.plot(cesm1_rest_all,7*np.ones(np.size(cesm1_rest_all)),'|',color='silver')

plt.axhline(8,color='silver')
##################

plt.plot(ipsl_b_rest_all,9*np.ones(np.size(ipsl_b_rest_all)),'|',color='silver')

plt.plot(ipsl_mr_rest_all,10*np.ones(np.size(ipsl_mr_rest_all)),'|',color='silver')
plt.plot(ipsl_mr_fam_rest_all,10*np.ones(np.size(ipsl_mr_fam_rest_all)),'|',color='dimgray')

#
plt.plot(ipsl_fam_all,11*np.ones(np.size(ipsl_fam_all)),'|',color='royalblue')
plt.plot(ipsl_rest_all,11*np.ones(np.size(ipsl_rest_all)),'|',color='silver')
plt.plot(ipsl_fam_rest_all,11*np.ones(np.size(ipsl_fam_rest_all)),'|',color='dimgray')

plt.axhline(12,color='silver')

##################

plt.plot(ec_fam_all,13*np.ones(np.size(ec_fam_all)),'|',color='darkgreen')
plt.plot(ec_rest_all,13*np.ones(np.size(ec_rest_all)),'|',color='silver')
plt.plot(ec_fam_rest_all,13*np.ones(np.size(ec_fam_rest_all)),'|',color='dimgray')

plt.plot(cnrm_fam_all,14*np.ones(np.size(cnrm_fam_all)),'|',color='cornflowerblue')
plt.plot(cnrm_rest_all,14*np.ones(np.size(cnrm_rest_all)),'|',color='silver')
plt.plot(cnrm_fam_rest_all,14*np.ones(np.size(cnrm_fam_rest_all)),'|',color='dimgray')

#
plt.axhline(15,color='silver')
##################

plt.plot(mpi_mr_rest_all,16*np.ones(np.size(mpi_mr_rest_all)),'|',color='silver')
plt.plot(mpi_mr_fam_rest_all,16*np.ones(np.size(mpi_mr_fam_rest_all)),'|',color='dimgray')

#
plt.plot(mpi_fam_all,17*np.ones(np.size(mpi_fam_all)),'|',color='tab:orange')
plt.plot(mpi_rest_all,17*np.ones(np.size(mpi_rest_all)),'|',color='silver')
plt.plot(mpi_fam_rest_all,17*np.ones(np.size(mpi_fam_rest_all)),'|',color='dimgray')

#
plt.axhline(18,color='silver')
##################

plt.plot(gfdlesm_g_rest_all,19*np.ones(np.size(gfdlesm_g_rest_all)),'|',color='silver')
plt.plot(gfdlesm_g_rest_all[0],19*np.ones(np.size(gfdlesm_g_rest_all[0])),'|',color='dimgray')

plt.plot(gfdlesm_m_rest_all,20*np.ones(np.size(gfdlesm_m_rest_all)),'|',color='silver')
plt.plot(gfdlesm_m_rest_all[0],20*np.ones(np.size(gfdlesm_m_rest_all[0])),'|',color='dimgray')

plt.plot(gfdl_rest_all,21*np.ones(np.size(gfdl_rest_all)),'|',color='silver')

plt.axhline(22,color='silver')
##################

plt.plot(miroc_fam_all,23*np.ones(np.size(miroc_fam_all)),'|',color='lightsalmon')
plt.plot(miroc_rest_all,23*np.ones(np.size(miroc_rest_all)),'|',color='silver')
#
plt.plot(miroc_esm_rest_all,24*np.ones(np.size(miroc_esm_rest_all)),'|',color='silver')

plt.axhline(25,color='silver')
##################

plt.plot(giss_h_fam_all,26*np.ones(np.size(giss_h_fam_all)),'|',color='blueviolet')
plt.plot(giss_h_rest_all,26*np.ones(np.size(giss_h_rest_all)),'|',color='silver')
plt.plot(giss_h_fam_rest_all,26*np.ones(np.size(giss_h_fam_rest_all)),'|',color='dimgray')

#
plt.plot(giss_r_fam_all,27*np.ones(np.size(giss_r_fam_all)),'|',color='blueviolet')
plt.plot(giss_r_rest_all,27*np.ones(np.size(giss_r_rest_all)),'|',color='silver')
plt.plot(giss_r_fam_rest_all,27*np.ones(np.size(giss_r_fam_rest_all)),'|',color='dimgray')

plt.axhline(28,color='silver')
##################

plt.plot(bcc_rest_all,29*np.ones(np.size(bcc_rest_all)),'|',color='silver')
plt.plot(bcc_rest_all[0],29*np.ones(np.size(bcc_rest_all[0])),'|',color='dimgray')

plt.plot(bcc_m_rest_all,30*np.ones(np.size(bcc_m_rest_all)),'|',color='silver')
#
##################

plt.plot(bnu_rest_all,31*np.ones(np.size(bnu_rest_all)),'|',color='silver')
plt.plot(bnu_rest_all[0],31*np.ones(np.size(bnu_rest_all[0])),'|',color='dimgray')

plt.axhline(32,color='silver')
##################

plt.plot(inmcm4_rest_all,33*np.ones(np.size(inmcm4_rest_all)),'|',color='silver')

##################

plt.plot(can_fam_all,34*np.ones(np.size(can_fam_all)),'|',color='dodgerblue')
plt.plot(can_rest_all,34*np.ones(np.size(can_rest_all)),'|',color='silver')

##################

plt.plot(mri_rest_all,35*np.ones(np.size(mri_rest_all)),'|',color='silver')

##################
plt.plot(csiro_fam_all,36*np.ones(np.size(csiro_fam_all)),'|',color='deeppink')
plt.plot(csiro_rest_all,36*np.ones(np.size(csiro_rest_all)),'|',color='silver')

##################

plt.plot(fgoals_g2_rest_all,37*np.ones(np.size(fgoals_g2_rest_all)),'|',color='silver')

##################
plt.xlim([0,2.2]) #################
plt.ylim([-1,38])
yticks = np.arange(0,38,1)
ax.set_yticks(yticks)
xticks = np.arange(0,2.4,0.2) #################
ax.set_xticks(xticks)
labels = ['1) ACCESS1-0','2) ACCESS1-3','3) HadGEM2-ES','','4) NorESM1-ME','5) NorESM1-M','6) CCSM4','7) CESM1-CAM5','',
'8) IPSL-CM5B-LR','9) IPSL-CM5A-MR','10) IPSL-CM5A-LR','','11) EC-EARTH','12) CNRM-CM5','','13) MPI-ESM-MR','14) MPI-ESM-LR','',
'15) GFDL-ESM2G','16) GFDL-ESM2M','17) GFDL-CM3','','18) MIROC5','19) MIROC-ESM','','20) GISS-E2-H','21) GISS-E2-R','',
'22) bcc-csm1-1','23) bcc-csm1-1-m','24) BNU-ESM','','25) inmcm4','26) CanESM2','27) MRI-CGCM3','28) CSIRO-Mk3-6-0','29) FGOALS-g2']
xlabels = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2] #,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8]
ax.set_yticklabels(labels,fontsize=9,rotation = 0)
ax.set_xticklabels(xlabels,fontsize=9)
plt.xlabel('intermember distance',fontsize=11)
ax.set_axisbelow(True)
ax.invert_yaxis()
plt.title('c) CMIP5 intermember distances',fontsize=12,fontweight='bold',loc='left') #,pad=14)
plt.savefig('Fig3_dists_CMIP5.png',bbox_inches='tight',dpi=300)

###################################

from sklearn.datasets import load_digits
from sklearn.manifold import MDS

embedding = MDS(n_components=2,metric=True,dissimilarity='euclidean',random_state=0)
dsWi_z = np.nan_to_num(dsWi,nan=0)

init = [(-1.40160113,  0.32645433),(-1.44302971,  0.74229915),
( 0.94974055,  1.74415789),
( 0.18063757,  3.28211836),( 0.15655257,  3.27192949),( 0.25130927,  3.28081435),( 0.25466986,  3.28790469),( 0.20215925,  3.28366917),( 0.23571454,  3.29565411),
(-1.29700709,  1.53227029),(-1.30172729,  1.50813119),(-1.30020357,  1.55482682),
( 0.14503746,  0.20916407),( 0.09933225,  0.28488248),( 0.07983861,  0.29713738),( 0.11642106,  0.30838589),( 0.12135818,  0.2693496 ),
(-2.63511539, -1.94648835),(-2.67557825, -1.88503757),(-2.69881224, -1.84667255),(-2.70519416, -1.87337981),(-2.69095109, -1.85957018),(-2.6641584 , -1.91597465),(-2.65501491, -1.92619098),(-2.66756863, -1.8990822 ),(-2.70292865, -1.89309091),(-2.69894167, -1.88123039),
(-2.18003442,  1.40905907),(-2.18280392,  1.37474708),(-2.19241719,  1.40025052),(-2.17468343,  1.36859213),(-2.15232937,  1.40900118),
(-0.22234675,  1.43755513),(-0.194369  ,  1.42540779),(-0.23434719,  1.44624399),(-0.23772224,  1.46606363),(-0.2044867 ,  1.44644708),
( 0.17339039, -1.6268998 ), ##
(-0.78752167,  0.19574092),(-0.05285538, -1.01052932),( 0.27419582, -0.92093055),( 3.70843425, -0.61407214),( 3.79975052, -0.73108555),( 4.00525238, -0.88986126),( 3.72337019, -0.6664153 ),( 3.9334624 , -0.81282437),( 2.59326079, -0.03915344),( 2.73173719, -0.12576098),( 2.75549079, -0.08435663),( 2.58977219, -0.0557036 ),( 2.77657692, -0.10736301),(-1.47185164,  0.11536438),(-1.43665848,  0.11840512),(-1.39093026,  0.10725418),(-1.49450327,  0.1659599 ),( 1.59088469, -4.25721622),( 1.56139342, -4.21452079),( 1.55250066, -4.24756604),( 1.53899698, -4.20491269),( 1.24556887, -3.05118288),( 3.25299639, -4.35346688),( 1.75685258,  1.28474841),( 1.62238207, -1.69438146),( 1.52605467, -1.66681149),( 1.50644868, -1.66331047),(-0.96898656,  0.60893332),(-0.94523771,  0.61711593),(-0.98632426,  0.65344962),(-0.71670602,  0.56814377),( 0.91657871,  0.77746407),(-0.76534024,  2.28314581),(-0.85553464,  1.9402052 ),( 1.8298508 ,  2.14848541),( 0.67286032,  1.51062295),( 0.9549887 ,  0.20748661)]

dsWi_transformed = embedding.fit_transform(dsWi_z,init=np.array(init))

def radius_var(data):
    ens_center = np.mean(data,axis=1).reshape(2,1)
    ens_center_rep = np.repeat(ens_center,data.shape[1], axis=1)
    dist = np.linalg.norm(data-ens_center_rep,axis=0)
    return 2.5*np.max(dist)


fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111)
pos = x, y = dsWi_transformed.transpose()
plt.title('d) CMIP5 multidimensional scaling of intermember distances',fontsize=12,fontweight='bold',loc='left') #,pad=14)
ax.scatter(x[0], y[0],s=10,  color="tab:red", label="original") #access1-0
ax.scatter(x[1], y[1],s=10,  color="tab:red", label="original") #access1-3
ax.scatter(x[2], y[2],s=10, color="tab:gray", label="original") #bnu_esm
ax.scatter(x[3:9], y[3:9],s=2, color="darkgoldenrod", label="original") #ccsm4
circle = plt.Circle((np.mean(x[3:9]), np.mean(y[3:9])),radius_var(pos[:,3:9]), color='darkgoldenrod',alpha=0.2)
ax.add_patch(circle)
ax.scatter(x[9:12], y[9:12],s=2, color="darkgoldenrod", label="original") #cesm1-cam5
circle = plt.Circle((np.mean(x[9:12]), np.mean(y[9:12])),radius_var(pos[:,9:12]), color='darkgoldenrod',alpha=0.2)
ax.add_patch(circle)
ax.scatter(x[12:17], y[12:17],s=2, color="cornflowerblue", label="original") #cnrm
circle = plt.Circle((np.mean(x[12:17]), np.mean(y[12:17])),radius_var(pos[:,12:17]), color='cornflowerblue',alpha=0.2)
ax.add_patch(circle)
ax.scatter(x[17:27], y[17:27],s=2, color="deeppink", label="original") #csiro
circle = plt.Circle((np.mean(x[17:27]), np.mean(y[17:27])),radius_var(pos[:,17:27]), color='deeppink',alpha=0.2)
ax.add_patch(circle)
ax.scatter(x[27:32], y[27:32],s=2, color="dodgerblue", label="original") #canesm2
circle = plt.Circle((np.mean(x[27:32]), np.mean(y[27:32])),radius_var(pos[:,27:32]), color='dodgerblue',alpha=0.2)
ax.add_patch(circle)
ax.scatter(x[32:37], y[32:37],s=2, color="darkgreen", label="original") #ec_earth
circle = plt.Circle((np.mean(x[32:37]), np.mean(y[32:37])),radius_var(pos[:,32:37]), color='darkgreen',alpha=0.2)
ax.add_patch(circle)
ax.scatter(x[37], y[37],s=10, color="maroon", label="original") #fgoals
ax.scatter(x[38], y[38],s=10, color="indigo", label="original") #gfdl_cm3
ax.scatter(x[39], y[39],s=10, color="indigo", label="original") #gfdl_esmg
ax.scatter(x[40], y[40],s=10, color="indigo", label="original") #gfdl_esmm
ax.scatter(x[41:46], y[41:46],s=2, color="blueviolet", label="original") #giss-h
circle = plt.Circle((np.mean(x[41:46]), np.mean(y[41:46])),radius_var(pos[:,41:46]), color='blueviolet',alpha=0.2)
ax.add_patch(circle)
ax.scatter(x[46:51], y[46:51],s=2, color="blueviolet", label="original") #giss-r
circle = plt.Circle((np.mean(x[46:51]), np.mean(y[46:51])),radius_var(pos[:,46:51]), color='blueviolet',alpha=0.2)
ax.add_patch(circle)
ax.scatter(x[51:55], y[51:55],s=2, color="tab:red", label="original") #hadgem
circle = plt.Circle((np.mean(x[51:55]), np.mean(y[51:55])),radius_var(pos[:,51:55]), color='tab:red',alpha=0.2)
ax.add_patch(circle)
ax.scatter(x[55:59], y[55:59],s=2, color="royalblue", label="original") #ipsl_cm5a_lr
circle = plt.Circle((np.mean(x[55:59]), np.mean(y[55:59])),radius_var(pos[:,55:59]), color='royalblue',alpha=0.2)
ax.add_patch(circle)
ax.scatter(x[59], y[59],s=10, color="royalblue", label="original") #ipsl_cm5a_mr
ax.scatter(x[60], y[60],s=10, color="royalblue", label="original") #ipsl_cm5b_lr
ax.scatter(x[61], y[61],s=10, color="lightsalmon", label="original") #miroc_esm
ax.scatter(x[62:65], y[62:65],s=2, color="lightsalmon", label="original") #MIROC5
circle = plt.Circle((np.mean(x[62:65]), np.mean(y[62:65])),radius_var(pos[:,62:65]), color='lightsalmon',alpha=0.2)
ax.add_patch(circle)
ax.scatter(x[65:68], y[65:68],s=2,  color="tab:orange", label="original") #mpi
circle = plt.Circle((np.mean(x[65:68]), np.mean(y[65:68])),radius_var(pos[:,65:68]), color='tab:orange',alpha=0.2)
ax.add_patch(circle)
ax.scatter(x[68], y[68],s=10,  color="tab:orange", label="original") #mpi_mr
ax.scatter(x[69], y[69],s=10, color="palevioletred", label="original") #mri_cgcm3
ax.scatter(x[70], y[70],s=10, color="darkgoldenrod", label="original") #noresm-me
ax.scatter(x[71], y[71],s=10, color="darkgoldenrod", label="original") #noresm-m
ax.scatter(x[72], y[72],s=10, color="silver", label="original") #bcc-csm1-1-m
ax.scatter(x[73], y[73],s=10, color="silver", label="original") #bcc-csm1-1
ax.scatter(x[74], y[74],s=10, color="mediumseagreen", label="original") #inm_48
plt.xlim([-5,5])
plt.ylim([-5,4])
plt.xlabel('Coordinate 1')
plt.ylabel('Coordinate 2')
plt.savefig('Fig3_CMIP5_projections_all.png',bbox_inches='tight',dpi=300)
