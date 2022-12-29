# Figure 6: five model example

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

## Load in

members_cmip6 = ['ACCESS-CM2-r1i1p1f1', 'ACCESS-CM2-r2i1p1f1', 'ACCESS-CM2-r3i1p1f1',
       'ACCESS-ESM1-5-r10i1p1f1', 'ACCESS-ESM1-5-r1i1p1f1',
       'ACCESS-ESM1-5-r2i1p1f1', 'ACCESS-ESM1-5-r3i1p1f1',
       'ACCESS-ESM1-5-r4i1p1f1', 'ACCESS-ESM1-5-r5i1p1f1',
       'ACCESS-ESM1-5-r6i1p1f1', 'ACCESS-ESM1-5-r7i1p1f1',
       'ACCESS-ESM1-5-r8i1p1f1', 'ACCESS-ESM1-5-r9i1p1f1',
       'AWI-CM-1-1-MR-r1i1p1f1', 'CAS-ESM2-0-r1i1p1f1', 'CAS-ESM2-0-r3i1p1f1',
       'CESM2-WACCM-r1i1p1f1', 'CESM2-WACCM-r2i1p1f1', 'CESM2-WACCM-r3i1p1f1',
       'CESM2-r10i1p1f1', 'CESM2-r11i1p1f1', 'CESM2-r1i1p1f1',
       'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1', 'CMCC-CM2-SR5-r1i1p1f1',
       'CMCC-ESM2-r1i1p1f1', 'CNRM-CM6-1-HR-r1i1p1f2', 'CNRM-CM6-1-r1i1p1f2',
       'CNRM-CM6-1-r2i1p1f2', 'CNRM-CM6-1-r3i1p1f2', 'CNRM-CM6-1-r4i1p1f2',
       'CNRM-CM6-1-r5i1p1f2', 'CNRM-CM6-1-r6i1p1f2', 'CNRM-ESM2-1-r1i1p1f2',
       'CNRM-ESM2-1-r2i1p1f2', 'CNRM-ESM2-1-r3i1p1f2', 'CNRM-ESM2-1-r4i1p1f2',
       'CNRM-ESM2-1-r5i1p1f2', 'CanESM5-r10i1p1f1', 'CanESM5-r10i1p2f1',
       'CanESM5-r11i1p1f1', 'CanESM5-r11i1p2f1', 'CanESM5-r12i1p1f1',
       'CanESM5-r12i1p2f1', 'CanESM5-r13i1p1f1', 'CanESM5-r13i1p2f1',
       'CanESM5-r14i1p1f1', 'CanESM5-r14i1p2f1', 'CanESM5-r15i1p1f1',
       'CanESM5-r15i1p2f1', 'CanESM5-r16i1p1f1', 'CanESM5-r16i1p2f1',
       'CanESM5-r17i1p1f1', 'CanESM5-r17i1p2f1', 'CanESM5-r18i1p1f1',
       'CanESM5-r18i1p2f1', 'CanESM5-r19i1p1f1', 'CanESM5-r19i1p2f1',
       'CanESM5-r1i1p1f1', 'CanESM5-r1i1p2f1', 'CanESM5-r20i1p1f1',
       'CanESM5-r20i1p2f1', 'CanESM5-r21i1p1f1', 'CanESM5-r21i1p2f1',
       'CanESM5-r22i1p1f1', 'CanESM5-r22i1p2f1', 'CanESM5-r23i1p1f1',
       'CanESM5-r23i1p2f1', 'CanESM5-r24i1p1f1', 'CanESM5-r24i1p2f1',
       'CanESM5-r25i1p1f1', 'CanESM5-r25i1p2f1', 'CanESM5-r2i1p1f1',
       'CanESM5-r2i1p2f1', 'CanESM5-r3i1p1f1', 'CanESM5-r3i1p2f1',
       'CanESM5-r4i1p1f1', 'CanESM5-r4i1p2f1', 'CanESM5-r5i1p1f1',
       'CanESM5-r5i1p2f1', 'CanESM5-r6i1p1f1', 'CanESM5-r6i1p2f1',
       'CanESM5-r7i1p1f1', 'CanESM5-r7i1p2f1', 'CanESM5-r8i1p1f1',
       'CanESM5-r8i1p2f1', 'CanESM5-r9i1p1f1', 'CanESM5-r9i1p2f1',
       'E3SM-1-1-r1i1p1f1', 'FGOALS-f3-L-r1i1p1f1', 'FGOALS-g3-r1i1p1f1',
       'FGOALS-g3-r2i1p1f1', 'GFDL-CM4-r1i1p1f1', 'GFDL-ESM4-r1i1p1f1',
       'GISS-E2-1-G-r1i1p3f1', 'HadGEM3-GC31-LL-r1i1p1f3',
       'HadGEM3-GC31-LL-r2i1p1f3', 'HadGEM3-GC31-LL-r3i1p1f3',
       'HadGEM3-GC31-LL-r4i1p1f3', 'HadGEM3-GC31-MM-r1i1p1f3',
       'HadGEM3-GC31-MM-r2i1p1f3', 'HadGEM3-GC31-MM-r3i1p1f3',
       'HadGEM3-GC31-MM-r4i1p1f3', 'INM-CM4-8-r1i1p1f1', 'INM-CM5-0-r1i1p1f1',
       'IPSL-CM6A-LR-r14i1p1f1', 'IPSL-CM6A-LR-r1i1p1f1',
       'IPSL-CM6A-LR-r2i1p1f1', 'IPSL-CM6A-LR-r3i1p1f1',
       'IPSL-CM6A-LR-r4i1p1f1', 'IPSL-CM6A-LR-r6i1p1f1', 'KACE-1-0-G-r2i1p1f1',
       'KACE-1-0-G-r3i1p1f1', 'KIOST-ESM-r1i1p1f1', 'MIROC-ES2L-r10i1p1f2',
       'MIROC-ES2L-r1i1p1f2', 'MIROC-ES2L-r2i1p1f2', 'MIROC-ES2L-r3i1p1f2',
       'MIROC-ES2L-r4i1p1f2', 'MIROC-ES2L-r5i1p1f2', 'MIROC-ES2L-r6i1p1f2',
       'MIROC-ES2L-r7i1p1f2', 'MIROC-ES2L-r8i1p1f2', 'MIROC-ES2L-r9i1p1f2',
       'MIROC6-r10i1p1f1', 'MIROC6-r11i1p1f1', 'MIROC6-r12i1p1f1',
       'MIROC6-r13i1p1f1', 'MIROC6-r14i1p1f1', 'MIROC6-r15i1p1f1',
       'MIROC6-r16i1p1f1', 'MIROC6-r17i1p1f1', 'MIROC6-r18i1p1f1',
       'MIROC6-r19i1p1f1', 'MIROC6-r1i1p1f1', 'MIROC6-r20i1p1f1',
       'MIROC6-r21i1p1f1', 'MIROC6-r22i1p1f1', 'MIROC6-r23i1p1f1',
       'MIROC6-r24i1p1f1', 'MIROC6-r25i1p1f1', 'MIROC6-r26i1p1f1',
       'MIROC6-r27i1p1f1', 'MIROC6-r28i1p1f1', 'MIROC6-r29i1p1f1',
       'MIROC6-r2i1p1f1', 'MIROC6-r30i1p1f1', 'MIROC6-r31i1p1f1',
       'MIROC6-r32i1p1f1', 'MIROC6-r33i1p1f1', 'MIROC6-r34i1p1f1',
       'MIROC6-r35i1p1f1', 'MIROC6-r36i1p1f1', 'MIROC6-r37i1p1f1',
       'MIROC6-r38i1p1f1', 'MIROC6-r39i1p1f1', 'MIROC6-r3i1p1f1',
       'MIROC6-r40i1p1f1', 'MIROC6-r41i1p1f1', 'MIROC6-r42i1p1f1',
       'MIROC6-r43i1p1f1', 'MIROC6-r44i1p1f1', 'MIROC6-r45i1p1f1',
       'MIROC6-r46i1p1f1', 'MIROC6-r47i1p1f1', 'MIROC6-r48i1p1f1',
       'MIROC6-r49i1p1f1', 'MIROC6-r4i1p1f1', 'MIROC6-r50i1p1f1',
       'MIROC6-r5i1p1f1', 'MIROC6-r6i1p1f1', 'MIROC6-r7i1p1f1',
       'MIROC6-r8i1p1f1', 'MIROC6-r9i1p1f1', 'MPI-ESM1-2-HR-r1i1p1f1',
       'MPI-ESM1-2-HR-r2i1p1f1', 'MPI-ESM1-2-LR-r10i1p1f1',
       'MPI-ESM1-2-LR-r1i1p1f1', 'MPI-ESM1-2-LR-r2i1p1f1',
       'MPI-ESM1-2-LR-r3i1p1f1', 'MPI-ESM1-2-LR-r4i1p1f1',
       'MPI-ESM1-2-LR-r5i1p1f1', 'MPI-ESM1-2-LR-r6i1p1f1',
       'MPI-ESM1-2-LR-r7i1p1f1', 'MPI-ESM1-2-LR-r8i1p1f1',
       'MPI-ESM1-2-LR-r9i1p1f1', 'MRI-ESM2-0-r1i1p1f1', 'MRI-ESM2-0-r1i2p1f1',
       'NESM3-r1i1p1f1', 'NESM3-r2i1p1f1', 'NorESM2-MM-r1i1p1f1',
       'TaiESM1-r1i1p1f1', 'UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
       'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2', 'UKESM1-0-LL-r8i1p1f2']


members_cmip5 = ['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1', 'CCSM4-r1i1p1', 'CCSM4-r2i1p1',
       'CCSM4-r3i1p1', 'CCSM4-r4i1p1', 'CCSM4-r5i1p1', 'CCSM4-r6i1p1',
       'CESM1-CAM5-r1i1p1', 'CESM1-CAM5-r2i1p1', 'CESM1-CAM5-r3i1p1',
       'CNRM-CM5-r10i1p1', 'CNRM-CM5-r1i1p1', 'CNRM-CM5-r2i1p1',
       'CNRM-CM5-r4i1p1', 'CNRM-CM5-r6i1p1', 'CSIRO-Mk3-6-0-r10i1p1',
       'CSIRO-Mk3-6-0-r1i1p1', 'CSIRO-Mk3-6-0-r2i1p1', 'CSIRO-Mk3-6-0-r3i1p1',
       'CSIRO-Mk3-6-0-r4i1p1', 'CSIRO-Mk3-6-0-r5i1p1', 'CSIRO-Mk3-6-0-r6i1p1',
       'CSIRO-Mk3-6-0-r7i1p1', 'CSIRO-Mk3-6-0-r8i1p1', 'CSIRO-Mk3-6-0-r9i1p1',
       'CanESM2-r1i1p1', 'CanESM2-r2i1p1', 'CanESM2-r3i1p1', 'CanESM2-r4i1p1',
       'CanESM2-r5i1p1', 'GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1',
       'GFDL-ESM2M-r1i1p1', 'GISS-E2-H-r1i1p1', 'GISS-E2-H-r1i1p2',
       'GISS-E2-H-r1i1p3', 'GISS-E2-H-r2i1p1', 'GISS-E2-H-r2i1p3',
       'GISS-E2-R-r1i1p1', 'GISS-E2-R-r1i1p2', 'GISS-E2-R-r1i1p3',
       'GISS-E2-R-r2i1p1', 'GISS-E2-R-r2i1p3', 'HadGEM2-ES-r1i1p1',
       'HadGEM2-ES-r2i1p1', 'HadGEM2-ES-r3i1p1', 'HadGEM2-ES-r4i1p1',
       'IPSL-CM5A-LR-r1i1p1', 'IPSL-CM5A-LR-r2i1p1', 'IPSL-CM5A-LR-r3i1p1',
       'IPSL-CM5A-LR-r4i1p1', 'IPSL-CM5A-MR-r1i1p1', 'IPSL-CM5B-LR-r1i1p1',
       'MIROC-ESM-r1i1p1', 'MIROC5-r1i1p1', 'MIROC5-r2i1p1', 'MIROC5-r3i1p1',
       'MPI-ESM-LR-r1i1p1', 'MPI-ESM-LR-r2i1p1', 'MPI-ESM-LR-r3i1p1',
       'MPI-ESM-MR-r1i1p1', 'MRI-CGCM3-r1i1p1', 'NorESM1-M-r1i1p1',
       'NorESM1-ME-r1i1p1', 'bcc-csm1-1-m-r1i1p1', 'bcc-csm1-1-r1i1p1',
       'inmcm4-r1i1p1']

#################################
# Performance
#################################

dir = '/**/CMIP_subselection/Data/'

dsSST6_ann = xr.open_dataset(dir + 'tos_mon_CMIP6_SSP585_g025_v2_NAWH_fix_ann.nc',use_cftime = True)
dsSST6_ann = dsSST6_ann.sortby(dsSST6_ann.member)
dsSST6_ann = dsSST6_ann.sel(member=members_cmip6)

dsSW6_ann = xr.open_dataset(dir + 'swcre_mon_CMIP6_SSP585_g025_v2_LWCLD_ann.nc',use_cftime = True)
dsSW6_ann = dsSW6_ann.sortby(dsSW6_ann.member)
dsSW6_ann = dsSW6_ann.sel(member=members_cmip6)

dsSW6_jja = xr.open_dataset(dir + 'swcre_mon_CMIP6_SSP585_g025_v2_CEU_JJA.nc',use_cftime = True)
dsSW6_jja = dsSW6_jja.sortby(dsSW6_jja.member)
dsSW6_jja = dsSW6_jja.sel(member=members_cmip6)

dsPr6_jja = xr.open_dataset(dir + 'pr_mon_CMIP6_SSP585_g025_v2_CEU_obsmask_JJA.nc',use_cftime = True)
dsPr6_jja['pr'] = dsPr6_jja.pr*86400
dsPr6_jja  = dsPr6_jja.sortby(dsPr6_jja.member)
dsPr6_jja = dsPr6_jja.sel(member=members_cmip6)

dsT6_ann = xr.open_dataset(dir + 'tas_mon_CMIP6_SSP585_g025_v2_EUR_ann.nc',use_cftime = True)
dsT6_ann = dsT6_ann-273.15
dsT6_ann = dsT6_ann.sortby(dsT6_ann.member)
dsT6_ann = dsT6_ann.sel(member=members_cmip6)

#####
dir = '/**/CMIP_subselection/Data/'

dsSST5_ann = xr.open_dataset(dir + 'tos_mon_CMIP5_rcp85_g025_v2_NAWH_fix_ann.nc',use_cftime = True)
dsSST5_ann = dsSST5_ann-273.15
dsSST5_ann = dsSST5_ann.sortby(dsSST5_ann.member)
dsSST5_ann = dsSST5_ann.sel(member=members_cmip5)

dsSW5_ann = xr.open_dataset(dir + 'swcre_mon_CMIP5_rcp85_g025_v2_LWCLD_ann.nc',use_cftime = True)
dsSW5_ann = dsSW5_ann.sortby(dsSW5_ann.member)
dsSW5_ann = dsSW5_ann.sel(member=members_cmip5)

dsSW5_jja = xr.open_dataset(dir + 'swcre_mon_CMIP5_rcp85_g025_v2_CEU_JJA.nc',use_cftime = True)
dsSW5_jja = dsSW5_jja.sortby(dsSW5_jja.member)
dsSW5_jja = dsSW5_jja.sel(member=members_cmip5)

dsPr5_jja = xr.open_dataset(dir + 'pr_mon_CMIP5_rcp85_g025_v2_CEU_obsmask_JJA.nc',use_cftime = True)
dsPr5_jja['pr'] = dsPr5_jja.pr*86400
dsPr5_jja = dsPr5_jja.sortby(dsPr5_jja.member)
dsPr5_jja = dsPr5_jja.sel(member=members_cmip5)

dsT5_ann = xr.open_dataset(dir + 'tas_mon_CMIP5_rcp85_g025_v2_EUR_ann.nc',use_cftime = True)
dsT5_ann = dsT5_ann-273.15
dsT5_ann = dsT5_ann.sortby(dsT5_ann.member)
dsT5_ann = dsT5_ann.sel(member=members_cmip5)

#####

dirobs = '/**/CMIP_subselection/Data/'
dsSSTobs_ann = xr.open_dataset(dirobs + 'tos_mon_OBS_g025_NAWH_fix_ann.nc',use_cftime = True)
dsSWobs_ann = xr.open_dataset(dirobs + 'swcre_mon_OBS_g025_LWCLD_ann.nc',use_cftime = True)
dsSWobs_jja = xr.open_dataset(dirobs + 'swcre_mon_OBS_g025_CEU_jja.nc',use_cftime = True)
dsProbs_jja = xr.open_dataset(dirobs + 'pr_mon_OBS_g025_CEU_JJA.nc',use_cftime = True)
dsTobs_ann = xr.open_dataset(dirobs + 'tas_mon_OBS_g025_EUR_ann.nc',use_cftime = True)

dsSST5_ann_base = dsSST5_ann.sel(year=slice('1995','2014')).mean('year')
dsSST6_ann_base = dsSST6_ann.sel(year=slice('1995','2014')).mean('year')
dsSSTobs_ann_base = dsSSTobs_ann.sel(year=slice('1995','2014')).mean('year')

dsSW5_ann_base = dsSW5_ann.sel(year=slice('2001','2018')).mean('year')
dsSW6_ann_base = dsSW6_ann.sel(year=slice('2001','2018')).mean('year')
dsSWobs_ann_base = dsSWobs_ann.sel(year=slice('2001','2018')).mean('year')

dsSW5_jja_base = dsSW5_jja.sel(year=slice('2001','2018')).mean('year') # data starts in march
dsSW6_jja_base = dsSW6_jja.sel(year=slice('2001','2018')).mean('year') # data starts in march
dsSWobs_jja_base = dsSWobs_jja.sel(year=slice('2001','2018')).mean('year') # data starts in march

dsPr5_jja_base = dsPr5_jja.sel(year=slice('1995','2014')).mean('year')
dsPr6_jja_base = dsPr6_jja.sel(year=slice('1995','2014')).mean('year')
dsProbs_jja_base = dsProbs_jja.sel(year=slice('1995','2014')).mean('year')

dsT5_ann_base = dsT5_ann.sel(year=slice('1995','2014')).mean('year')
dsT6_ann_base = dsT6_ann.sel(year=slice('1995','2014')).mean('year')
dsTobs_ann_base = dsTobs_ann.sel(year=slice('1995','2014')).mean('year')

dsT5_ann_his = dsT5_ann.sel(year=slice('1950','1969')).mean('year')
dsT6_ann_his = dsT6_ann.sel(year=slice('1950','1969')).mean('year')
dsTobs_ann_his = dsTobs_ann.sel(year=slice('1950','1969')).mean('year')

## Compute Delta Values

weights = [np.cos(np.deg2rad(dsSST5_ann_base.lat))]*144
weights = xr.concat(weights, "lon")
weights['lon'] = dsSST5_ann_base.lon

dsSST5_ann_base_delta = xskillscore.rmse(dsSST5_ann_base.tos,dsSSTobs_ann_base.tos,dim=['lat','lon'],weights=weights,skipna=True)
dsSW5_ann_base_delta = xskillscore.rmse(dsSW5_ann_base.swcre,dsSWobs_ann_base.swcre,dim=['lat','lon'],weights=weights,skipna=True)
dsSW5_jja_base_delta = xskillscore.rmse(dsSW5_jja_base.swcre,dsSWobs_jja_base.swcre,dim=['lat','lon'],weights=weights,skipna=True)
dsPr5_jja_base_delta = xskillscore.rmse(dsPr5_jja_base.pr,dsProbs_jja_base.pr,dim=['lat','lon'],weights=weights,skipna=True)
dsT5_ann_base_delta = xskillscore.rmse(dsT5_ann_base.tas,dsTobs_ann_base.tas,dim=['lat','lon'],weights=weights,skipna=True)
dsT5_ann_his_delta = xskillscore.rmse(dsT5_ann_his.tas,dsTobs_ann_his.tas,dim=['lat','lon'],weights=weights,skipna=True)

weights = [np.cos(np.deg2rad(dsSST6_ann_base.lat))]*144
weights = xr.concat(weights, "lon")
weights['lon'] = dsSST6_ann_base.lon

dsSST6_ann_base_delta = xskillscore.rmse(dsSST6_ann_base.tos,dsSSTobs_ann_base.tos,dim=['lat','lon'],weights=weights,skipna=True)
dsSW6_ann_base_delta = xskillscore.rmse(dsSW6_ann_base.swcre,dsSWobs_ann_base.swcre,dim=['lat','lon'],weights=weights,skipna=True)
dsSW6_jja_base_delta = xskillscore.rmse(dsSW6_jja_base.swcre,dsSWobs_jja_base.swcre,dim=['lat','lon'],weights=weights,skipna=True)
dsPr6_jja_base_delta = xskillscore.rmse(dsPr6_jja_base.pr,dsProbs_jja_base.pr,dim=['lat','lon'],weights=weights,skipna=True)
dsT6_ann_base_delta = xskillscore.rmse(dsT6_ann_base.tas,dsTobs_ann_base.tas,dim=['lat','lon'],weights=weights,skipna=True)
dsT6_ann_his_delta = xskillscore.rmse(dsT6_ann_his.tas,dsTobs_ann_his.tas,dim=['lat','lon'],weights=weights,skipna=True)

dsSST_ann_base_delta = xr.concat([dsSST5_ann_base_delta,dsSST6_ann_base_delta],dim='member')
dsSW_ann_base_delta = xr.concat([dsSW5_ann_base_delta,dsSW6_ann_base_delta],dim='member')
dsSW_jja_base_delta = xr.concat([dsSW5_jja_base_delta,dsSW6_jja_base_delta],dim='member')
dsPr_jja_base_delta = xr.concat([dsPr5_jja_base_delta,dsPr6_jja_base_delta],dim='member')
dsT_ann_base_delta = xr.concat([dsT5_ann_base_delta,dsT6_ann_base_delta],dim='member')
dsT_ann_his_delta = xr.concat([dsT5_ann_his_delta,dsT6_ann_his_delta],dim='member')

## Normalize

dsSST5_ann_base_norm = dsSST5_ann_base_delta/dsSST_ann_base_delta.mean('member')
dsSW5_ann_base_norm = dsSW5_ann_base_delta/dsSW_ann_base_delta.mean('member')
dsSW5_jja_base_norm = dsSW5_jja_base_delta/dsSW_jja_base_delta.mean('member')
dsPr5_jja_base_norm = dsPr5_jja_base_delta/dsPr_jja_base_delta.mean('member')
dsT5_ann_base_norm = dsT5_ann_base_delta/dsT_ann_base_delta.mean('member')
dsT5_ann_his_norm = dsT5_ann_his_delta/dsT_ann_his_delta.mean('member')

dsSST6_ann_base_norm = dsSST6_ann_base_delta/dsSST_ann_base_delta.mean('member')
dsSW6_ann_base_norm = dsSW6_ann_base_delta/dsSW_ann_base_delta.mean('member')
dsSW6_jja_base_norm = dsSW6_jja_base_delta/dsSW_jja_base_delta.mean('member')
dsPr6_jja_base_norm = dsPr6_jja_base_delta/dsPr_jja_base_delta.mean('member')
dsT6_ann_base_norm = dsT6_ann_base_delta/dsT_ann_base_delta.mean('member')
dsT6_ann_his_norm = dsT6_ann_his_delta/dsT_ann_his_delta.mean('member')

# Compute Predictor
dsDelta5_1c = (dsSST5_ann_base_norm + dsSW5_ann_base_norm + dsSW5_jja_base_norm + dsPr5_jja_base_norm  + dsT5_ann_his_norm + dsT5_ann_base_norm)/6
dsDelta6_1c = (dsSST6_ann_base_norm + dsSW6_ann_base_norm + dsSW6_jja_base_norm + dsPr6_jja_base_norm + dsT6_ann_his_norm + dsT6_ann_base_norm)/6

## cmip6 performance ensemble averaging
rng = np.arange(99,103)
hadgemmm = dsDelta6_1c.isel(member=rng).mean()
hadgemmm['member'] = 'HadGEM3-GC31-MM-r0i0p0f0'
rng = np.arange(174,176)
mpihr = dsDelta6_1c.isel(member=rng).mean()
mpihr['member'] = 'MPI-ESM1-2-HR-r0i0p0f0'
rng = np.arange(186,188)
mri2 = dsDelta6_1c.isel(member=rng).mean()
mri2['member'] = 'MRI-ESM2-0-r0i0p0f0'
rng = np.arange(16,19)
cesm2_waccm = dsDelta6_1c.isel(member=rng).mean()
cesm2_waccm['member'] = 'CESM2-WACCM-r0i0p0f0'
rng = np.arange(19,24)
cesm2 = dsDelta6_1c.isel(member=rng).mean()
cesm2['member'] = 'CESM2-r0i0p0f0'
rng = np.arange(33,38)
cnrm2 = dsDelta6_1c.isel(member=rng).mean()
cnrm2['member'] = 'CNRM-ESM2-1-r0i0p0f0'
rng = np.arange(27,33)
cnrm6 = dsDelta6_1c.isel(member=rng).mean()
cnrm6['member'] = 'CNRM-CM6-1-r0i0p0f0'
rng = np.arange(95,99)
hadgemll = dsDelta6_1c.isel(member=rng).mean()
hadgemll['member'] = 'HadGEM3-GC31-LL-r0i0p0f0'
rng = np.arange(0,3)
access_cm2 = dsDelta6_1c.isel(member=rng).mean()
access_cm2['member'] = 'ACCESS-CM2-r0i0p0f0'
rng = np.arange(105,111)
ipsl6a = dsDelta6_1c.isel(member=rng).mean()
ipsl6a['member'] = 'IPSL-CM6A-LR-r0i0p0f0'
rng = np.arange(3,13)
access_5 = dsDelta6_1c.isel(member=rng).mean()
access_5['member'] = 'ACCESS-ESM1-5-r0i0p0f0'
rng = np.arange(192,197)
uk = dsDelta6_1c.isel(member=rng).mean()
uk['member'] = 'UKESM1-0-LL-r0i0p0f0'
rng = np.arange(176,186)
mpi2lr = dsDelta6_1c.isel(member=rng).mean()
mpi2lr['member'] = 'MPI-ESM1-2-LR-r0i0p0f0'
rng = np.arange(38,88)
canesm5 = dsDelta6_1c.isel(member=rng).mean()
canesm5['member'] = 'CanESM5-r0i0p0f0'
rng = np.arange(114,124)
miroce = dsDelta6_1c.isel(member=rng).mean()
miroce['member'] = 'MIROC-ES2L-r0i0p0f0'
rng = np.arange(124,174)
miroc6 = dsDelta6_1c.isel(member=rng).mean()
miroc6['member'] = 'MIROC6-r0i0p0f0'
rng = np.arange(188,190)
nesm3 = dsDelta6_1c.isel(member=rng).mean()
nesm3['member'] = 'NESM3-r0i0p0f0'
rng = np.arange(90,92)
fgoalsg = dsDelta6_1c.isel(member=rng).mean()
fgoalsg['member'] = 'FGOALS-g3-r0i0p0f0'
rng = np.arange(111,113)
kace = dsDelta6_1c.isel(member=rng).mean()
kace['member'] = 'KACE-1-0-G-r0i0p0f0'
rng = np.arange(14,16)
cas = dsDelta6_1c.isel(member=rng).mean()
cas['member'] = 'CAS-ESM2-0-r0i0p0f0'

dsDelta6_1c_em = xr.concat([access_cm2,access_5,dsDelta6_1c.sel(member='AWI-CM-1-1-MR-r1i1p1f1'),cas,
cesm2_waccm,cesm2,dsDelta6_1c.sel(member='CMCC-CM2-SR5-r1i1p1f1'),dsDelta6_1c.sel(member='CMCC-ESM2-r1i1p1f1'),
dsDelta6_1c.sel(member='CNRM-CM6-1-HR-r1i1p1f2'),cnrm6,cnrm2,canesm5,dsDelta6_1c.sel(member='E3SM-1-1-r1i1p1f1'),
dsDelta6_1c.sel(member='FGOALS-f3-L-r1i1p1f1'),fgoalsg,dsDelta6_1c.sel(member='GFDL-CM4-r1i1p1f1'),
dsDelta6_1c.sel(member='GFDL-ESM4-r1i1p1f1'),dsDelta6_1c.sel(member='GISS-E2-1-G-r1i1p3f1'),hadgemll,
hadgemmm,dsDelta6_1c.sel(member='INM-CM4-8-r1i1p1f1'),dsDelta6_1c.sel(member='INM-CM5-0-r1i1p1f1'),
ipsl6a,kace,dsDelta6_1c.sel(member='KIOST-ESM-r1i1p1f1'),miroce,miroc6,mpihr,mpi2lr,mri2,nesm3,
dsDelta6_1c.sel(member='NorESM2-MM-r1i1p1f1'),dsDelta6_1c.sel(member='TaiESM1-r1i1p1f1'),uk],dim='member')

#################################
# Targets
#################################

dir = '/**/CMIP_subselection/Data/'

dsT6_jja = xr.open_dataset(dir + 'tas_mon_CMIP6_SSP585_g025_v2_CEU_JJA.nc',use_cftime = True)
dsT6_jja  = dsT6_jja-273.15
dsT6_jja  = dsT6_jja.sortby(dsT6_jja.member)
dsT6_jja = dsT6_jja.sel(member=members_cmip6)


dsPr6_ceu_jja = xr.open_dataset(dir + 'pr_mon_CMIP6_SSP585_g025_v2_CEU_JJA.nc',use_cftime = True)
dsPr6_ceu_jja['pr'] = dsPr6_ceu_jja.pr*86400
dsPr6_ceu_jja  = dsPr6_ceu_jja.sortby(dsPr6_ceu_jja.member)
dsPr6_ceu_jja = dsPr6_ceu_jja.sel(member=members_cmip6)

dsT6_jja_base = dsT6_jja.sel(year=slice('1995','2014')).mean('year')
dsT6_jja_fut = dsT6_jja.sel(year=slice('2041','2060')).mean('year')

dsT6_target = dsT6_jja_fut - dsT6_jja_base

dsPr6_ceu_jja_base = dsPr6_ceu_jja.sel(year=slice('1995','2014')).mean('year')
dsPr6_ceu_jja_fut = dsPr6_ceu_jja.sel(year=slice('2041','2060')).mean('year')

dsPr6_target = dsPr6_ceu_jja_fut - dsPr6_ceu_jja_base

def cos_lat_weighted_mean(ds):
 weights = np.cos(np.deg2rad(ds.lat))
 weights.name = "weights"
 ds_weighted = ds.weighted(weights)
 weighted_mean = ds_weighted.mean(('lon', 'lat'))
 return weighted_mean

dsT6_target_ts = cos_lat_weighted_mean(dsT6_target)
dsPr6_target_ts = cos_lat_weighted_mean(dsPr6_target)

## cmip6 ensemble averaging
rng = np.arange(99,103)
hadgemmm = dsT6_target_ts.isel(member=rng).mean('member').tas
hadgemmm['member'] = 'HadGEM3-GC31-MM-r0i0p0f0'
rng = np.arange(174,176)
mpihr = dsT6_target_ts.isel(member=rng).mean('member').tas
mpihr['member'] = 'MPI-ESM1-2-HR-r0i0p0f0'
rng = np.arange(186,188)
mri2 = dsT6_target_ts.isel(member=rng).mean('member').tas
mri2['member'] = 'MRI-ESM2-0-r0i0p0f0'
rng = np.arange(16,19)
cesm2_waccm = dsT6_target_ts.isel(member=rng).mean('member').tas
cesm2_waccm['member'] = 'CESM2-WACCM-r0i0p0f0'
rng = np.arange(19,24)
cesm2 = dsT6_target_ts.isel(member=rng).mean('member').tas
cesm2['member'] = 'CESM2-r0i0p0f0'
rng = np.arange(33,38)
cnrm2 = dsT6_target_ts.isel(member=rng).mean('member').tas
cnrm2['member'] = 'CNRM-ESM2-1-r0i0p0f0'
rng = np.arange(27,33)
cnrm6 = dsT6_target_ts.isel(member=rng).mean('member').tas
cnrm6['member'] = 'CNRM-CM6-1-r0i0p0f0'
rng = np.arange(95,99)
hadgemll = dsT6_target_ts.isel(member=rng).mean('member').tas
hadgemll['member'] = 'HadGEM3-GC31-LL-r0i0p0f0'
rng = np.arange(0,3)
access_cm2 = dsT6_target_ts.isel(member=rng).mean('member').tas
access_cm2['member'] = 'ACCESS-CM2-r0i0p0f0'
rng = np.arange(105,111)
ipsl6a = dsT6_target_ts.isel(member=rng).mean('member').tas
ipsl6a['member'] = 'IPSL-CM6A-LR-r0i0p0f0'
rng = np.arange(3,13)
access_5 = dsT6_target_ts.isel(member=rng).mean('member').tas
access_5['member'] = 'ACCESS-ESM1-5-r0i0p0f0'
rng = np.arange(192,197)
uk = dsT6_target_ts.isel(member=rng).mean('member').tas
uk['member'] = 'UKESM1-0-LL-r0i0p0f0'
rng = np.arange(176,186)
mpi2lr = dsT6_target_ts.isel(member=rng).mean('member').tas
mpi2lr['member'] = 'MPI-ESM1-2-LR-r0i0p0f0'
rng = np.arange(38,88)
canesm5 = dsT6_target_ts.isel(member=rng).mean('member').tas
canesm5['member'] = 'CanESM5-r0i0p0f0'
rng = np.arange(114,124)
miroce = dsT6_target_ts.isel(member=rng).mean('member').tas
miroce['member'] = 'MIROC-ES2L-r0i0p0f0'
rng = np.arange(124,174)
miroc6 = dsT6_target_ts.isel(member=rng).mean('member').tas
miroc6['member'] = 'MIROC6-r0i0p0f0'
rng = np.arange(188,190)
nesm3 = dsT6_target_ts.isel(member=rng).mean('member').tas
nesm3['member'] = 'NESM3-r0i0p0f0'
rng = np.arange(90,92)
fgoalsg = dsT6_target_ts.isel(member=rng).mean('member').tas
fgoalsg['member'] = 'FGOALS-g3-r0i0p0f0'
rng = np.arange(111,113)
kace = dsT6_target_ts.isel(member=rng).mean('member').tas
kace['member'] = 'KACE-1-0-G-r0i0p0f0'
rng = np.arange(14,16)
cas = dsT6_target_ts.isel(member=rng).mean('member').tas
cas['member'] = 'CAS-ESM2-0-r0i0p0f0'

dsT6_target_ts_em = xr.concat([access_cm2,access_5,dsT6_target_ts.sel(member='AWI-CM-1-1-MR-r1i1p1f1').tas,cas,
cesm2_waccm,cesm2,dsT6_target_ts.sel(member='CMCC-CM2-SR5-r1i1p1f1').tas,dsT6_target_ts.sel(member='CMCC-ESM2-r1i1p1f1').tas,
dsT6_target_ts.sel(member='CNRM-CM6-1-HR-r1i1p1f2').tas,cnrm6,cnrm2,canesm5,dsT6_target_ts.sel(member='E3SM-1-1-r1i1p1f1').tas,
dsT6_target_ts.sel(member='FGOALS-f3-L-r1i1p1f1').tas,fgoalsg,dsT6_target_ts.sel(member='GFDL-CM4-r1i1p1f1').tas,
dsT6_target_ts.sel(member='GFDL-ESM4-r1i1p1f1').tas,dsT6_target_ts.sel(member='GISS-E2-1-G-r1i1p3f1').tas,hadgemll,
hadgemmm,dsT6_target_ts.sel(member='INM-CM4-8-r1i1p1f1').tas,dsT6_target_ts.sel(member='INM-CM5-0-r1i1p1f1').tas,
ipsl6a,kace,dsT6_target_ts.sel(member='KIOST-ESM-r1i1p1f1').tas,miroce,miroc6,mpihr,mpi2lr,mri2,nesm3,
dsT6_target_ts.sel(member='NorESM2-MM-r1i1p1f1').tas,dsT6_target_ts.sel(member='TaiESM1-r1i1p1f1').tas,uk],dim='member')

## cmip6 ensemble averaging
rng = np.arange(99,103)
hadgemmm = dsPr6_target_ts.isel(member=rng).mean('member').pr
hadgemmm['member'] = 'HadGEM3-GC31-MM-r0i0p0f0'
rng = np.arange(174,176)
mpihr = dsPr6_target_ts.isel(member=rng).mean('member').pr
mpihr['member'] = 'MPI-ESM1-2-HR-r0i0p0f0'
rng = np.arange(186,188)
mri2 = dsPr6_target_ts.isel(member=rng).mean('member').pr
mri2['member'] = 'MRI-ESM2-0-r0i0p0f0'
rng = np.arange(16,19)
cesm2_waccm = dsPr6_target_ts.isel(member=rng).mean('member').pr
cesm2_waccm['member'] = 'CESM2-WACCM-r0i0p0f0'
rng = np.arange(19,24)
cesm2 = dsPr6_target_ts.isel(member=rng).mean('member').pr
cesm2['member'] = 'CESM2-r0i0p0f0'
rng = np.arange(33,38)
cnrm2 = dsPr6_target_ts.isel(member=rng).mean('member').pr
cnrm2['member'] = 'CNRM-ESM2-1-r0i0p0f0'
rng = np.arange(27,33)
cnrm6 = dsPr6_target_ts.isel(member=rng).mean('member').pr
cnrm6['member'] = 'CNRM-CM6-1-r0i0p0f0'
rng = np.arange(95,99)
hadgemll = dsPr6_target_ts.isel(member=rng).mean('member').pr
hadgemll['member'] = 'HadGEM3-GC31-LL-r0i0p0f0'
rng = np.arange(0,3)
access_cm2 = dsPr6_target_ts.isel(member=rng).mean('member').pr
access_cm2['member'] = 'ACCESS-CM2-r0i0p0f0'
rng = np.arange(105,111)
ipsl6a = dsPr6_target_ts.isel(member=rng).mean('member').pr
ipsl6a['member'] = 'IPSL-CM6A-LR-r0i0p0f0'
rng = np.arange(3,13)
access_5 = dsPr6_target_ts.isel(member=rng).mean('member').pr
access_5['member'] = 'ACCESS-ESM1-5-r0i0p0f0'
rng = np.arange(192,197)
uk = dsPr6_target_ts.isel(member=rng).mean('member').pr
uk['member'] = 'UKESM1-0-LL-r0i0p0f0'
rng = np.arange(176,186)
mpi2lr = dsPr6_target_ts.isel(member=rng).mean('member').pr
mpi2lr['member'] = 'MPI-ESM1-2-LR-r0i0p0f0'
rng = np.arange(38,88)
canesm5 = dsPr6_target_ts.isel(member=rng).mean('member').pr
canesm5['member'] = 'CanESM5-r0i0p0f0'
rng = np.arange(114,124)
miroce = dsPr6_target_ts.isel(member=rng).mean('member').pr
miroce['member'] = 'MIROC-ES2L-r0i0p0f0'
rng = np.arange(124,174)
miroc6 = dsPr6_target_ts.isel(member=rng).mean('member').pr
miroc6['member'] = 'MIROC6-r0i0p0f0'
rng = np.arange(188,190)
nesm3 = dsPr6_target_ts.isel(member=rng).mean('member').pr
nesm3['member'] = 'NESM3-r0i0p0f0'
rng = np.arange(90,92)
fgoalsg = dsPr6_target_ts.isel(member=rng).mean('member').pr
fgoalsg['member'] = 'FGOALS-g3-r0i0p0f0'
rng = np.arange(111,113)
kace = dsPr6_target_ts.isel(member=rng).mean('member').pr
kace['member'] = 'KACE-1-0-G-r0i0p0f0'
rng = np.arange(14,16)
cas = dsPr6_target_ts.isel(member=rng).mean('member').pr
cas['member'] = 'CAS-ESM2-0-r0i0p0f0'

dsPr6_target_ts_em = xr.concat([access_cm2,access_5,dsPr6_target_ts.sel(member='AWI-CM-1-1-MR-r1i1p1f1').pr,cas,
cesm2_waccm,cesm2,dsPr6_target_ts.sel(member='CMCC-CM2-SR5-r1i1p1f1').pr,dsPr6_target_ts.sel(member='CMCC-ESM2-r1i1p1f1').pr,
dsPr6_target_ts.sel(member='CNRM-CM6-1-HR-r1i1p1f2').pr,cnrm6,cnrm2,canesm5,dsPr6_target_ts.sel(member='E3SM-1-1-r1i1p1f1').pr,
dsPr6_target_ts.sel(member='FGOALS-f3-L-r1i1p1f1').pr,fgoalsg,dsPr6_target_ts.sel(member='GFDL-CM4-r1i1p1f1').pr,
dsPr6_target_ts.sel(member='GFDL-ESM4-r1i1p1f1').pr,dsPr6_target_ts.sel(member='GISS-E2-1-G-r1i1p3f1').pr,hadgemll,
hadgemmm,dsPr6_target_ts.sel(member='INM-CM4-8-r1i1p1f1').pr,dsPr6_target_ts.sel(member='INM-CM5-0-r1i1p1f1').pr,
ipsl6a,kace,dsPr6_target_ts.sel(member='KIOST-ESM-r1i1p1f1').pr,miroce,miroc6,mpihr,mpi2lr,mri2,nesm3,
dsPr6_target_ts.sel(member='NorESM2-MM-r1i1p1f1').pr,dsPr6_target_ts.sel(member='TaiESM1-r1i1p1f1').pr,uk],dim='member')

######### normalize
dsT6_target_ts_em_norm = (dsT6_target_ts_em - np.mean(dsT6_target_ts_em))/np.std(dsT6_target_ts_em)
dsPr6_target_ts_em_norm = (dsPr6_target_ts_em - np.mean(dsPr6_target_ts_em))/np.std(dsPr6_target_ts_em)

#################################
# Independence
#################################

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

# internal variability
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

# intermember spread
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
'TaiESM1-r1i1p1f1', 'UKESM1-0-LL-r1i1p1f2']).std('member') #37

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
dsT6_clim_mask = dsT6_clim_mask.sel(member=members_cmip6)

# ###################

rng = np.arange(99,103)
hadgemmm = dsT6_clim_mask.isel(member=rng).mean('member').tas
hadgemmm['member'] = 'HadGEM3-GC31-MM-r0i0p0f0'
rng = np.arange(174,176)
mpihr = dsT6_clim_mask.isel(member=rng).mean('member').tas
mpihr['member'] = 'MPI-ESM1-2-HR-r0i0p0f0'
rng = np.arange(186,188)
mri2 = dsT6_clim_mask.isel(member=rng).mean('member').tas
mri2['member'] = 'MRI-ESM2-0-r0i0p0f0'
rng = np.arange(16,19)
cesm2_waccm = dsT6_clim_mask.isel(member=rng).mean('member').tas
cesm2_waccm['member'] = 'CESM2-WACCM-r0i0p0f0'
rng = np.arange(19,24)
cesm2 = dsT6_clim_mask.isel(member=rng).mean('member').tas
cesm2['member'] = 'CESM2-r0i0p0f0'
rng = np.arange(33,38)
cnrm2 = dsT6_clim_mask.isel(member=rng).mean('member').tas
cnrm2['member'] = 'CNRM-ESM2-1-r0i0p0f0'
rng = np.arange(27,33)
cnrm6 = dsT6_clim_mask.isel(member=rng).mean('member').tas
cnrm6['member'] = 'CNRM-CM6-1-r0i0p0f0'
rng = np.arange(95,99)
hadgemll = dsT6_clim_mask.isel(member=rng).mean('member').tas
hadgemll['member'] = 'HadGEM3-GC31-LL-r0i0p0f0'
rng = np.arange(0,3)
access_cm2 = dsT6_clim_mask.isel(member=rng).mean('member').tas
access_cm2['member'] = 'ACCESS-CM2-r0i0p0f0'
rng = np.arange(105,111)
ipsl6a = dsT6_clim_mask.isel(member=rng).mean('member').tas
ipsl6a['member'] = 'IPSL-CM6A-LR-r0i0p0f0'
rng = np.arange(3,13)
access_5 = dsT6_clim_mask.isel(member=rng).mean('member').tas
access_5['member'] = 'ACCESS-ESM1-5-r0i0p0f0'
rng = np.arange(192,197)
uk = dsT6_clim_mask.isel(member=rng).mean('member').tas
uk['member'] = 'UKESM1-0-LL-r0i0p0f0'
rng = np.arange(176,186)
mpi2lr = dsT6_clim_mask.isel(member=rng).mean('member').tas
mpi2lr['member'] = 'MPI-ESM1-2-LR-r0i0p0f0'
rng = np.arange(38,88)
canesm5 = dsT6_clim_mask.isel(member=rng).mean('member').tas
canesm5['member'] = 'CanESM5-r0i0p0f0'
rng = np.arange(114,124)
miroce = dsT6_clim_mask.isel(member=rng).mean('member').tas
miroce['member'] = 'MIROC-ES2L-r0i0p0f0'
rng = np.arange(124,174)
miroc6 = dsT6_clim_mask.isel(member=rng).mean('member').tas
miroc6['member'] = 'MIROC6-r0i0p0f0'
rng = np.arange(188,190)
nesm3 = dsT6_clim_mask.isel(member=rng).mean('member').tas
nesm3['member'] = 'NESM3-r0i0p0f0'
rng = np.arange(90,92)
fgoalsg = dsT6_clim_mask.isel(member=rng).mean('member').tas
fgoalsg['member'] = 'FGOALS-g3-r0i0p0f0'
rng = np.arange(111,113)
kace = dsT6_clim_mask.isel(member=rng).mean('member').tas
kace['member'] = 'KACE-1-0-G-r0i0p0f0'
rng = np.arange(14,16)
cas = dsT6_clim_mask.isel(member=rng).mean('member').tas
cas['member'] = 'CAS-ESM2-0-r0i0p0f0'

dsT6_clim_mask_em = xr.concat([access_cm2,access_5,dsT6_clim_mask.sel(member='AWI-CM-1-1-MR-r1i1p1f1').tas,cas,
cesm2_waccm,cesm2,dsT6_clim_mask.sel(member='CMCC-CM2-SR5-r1i1p1f1').tas,dsT6_clim_mask.sel(member='CMCC-ESM2-r1i1p1f1').tas,
dsT6_clim_mask.sel(member='CNRM-CM6-1-HR-r1i1p1f2').tas,cnrm6,cnrm2,canesm5,dsT6_clim_mask.sel(member='E3SM-1-1-r1i1p1f1').tas,
dsT6_clim_mask.sel(member='FGOALS-f3-L-r1i1p1f1').tas,fgoalsg,dsT6_clim_mask.sel(member='GFDL-CM4-r1i1p1f1').tas,
dsT6_clim_mask.sel(member='GFDL-ESM4-r1i1p1f1').tas,dsT6_clim_mask.sel(member='GISS-E2-1-G-r1i1p3f1').tas,hadgemll,
hadgemmm,dsT6_clim_mask.sel(member='INM-CM4-8-r1i1p1f1').tas,dsT6_clim_mask.sel(member='INM-CM5-0-r1i1p1f1').tas,
ipsl6a,kace,dsT6_clim_mask.sel(member='KIOST-ESM-r1i1p1f1').tas,miroce,miroc6,mpihr,mpi2lr,mri2,nesm3,
dsT6_clim_mask.sel(member='NorESM2-MM-r1i1p1f1').tas,dsT6_clim_mask.sel(member='TaiESM1-r1i1p1f1').tas,uk],dim='member')

dsT6_clim_mask_em = dsT6_clim_mask_em.sel(member=dsDelta6_1c_em.member)
dsT6_clim_mask_em = dsT6_clim_mask_em.to_dataset(name='tas')

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
dsP6_clim_mask = dsP6_clim_mask.sel(member=members_cmip6)

# ####################

rng = np.arange(99,103)
hadgemmm = dsP6_clim_mask.isel(member=rng).mean('member').psl
hadgemmm['member'] = 'HadGEM3-GC31-MM-r0i0p0f0'
rng = np.arange(174,176)
mpihr = dsP6_clim_mask.isel(member=rng).mean('member').psl
mpihr['member'] = 'MPI-ESM1-2-HR-r0i0p0f0'
rng = np.arange(186,188)
mri2 = dsP6_clim_mask.isel(member=rng).mean('member').psl
mri2['member'] = 'MRI-ESM2-0-r0i0p0f0'
rng = np.arange(16,19)
cesm2_waccm = dsP6_clim_mask.isel(member=rng).mean('member').psl
cesm2_waccm['member'] = 'CESM2-WACCM-r0i0p0f0'
rng = np.arange(19,24)
cesm2 = dsP6_clim_mask.isel(member=rng).mean('member').psl
cesm2['member'] = 'CESM2-r0i0p0f0'
rng = np.arange(33,38)
cnrm2 = dsP6_clim_mask.isel(member=rng).mean('member').psl
cnrm2['member'] = 'CNRM-ESM2-1-r0i0p0f0'
rng = np.arange(27,33)
cnrm6 = dsP6_clim_mask.isel(member=rng).mean('member').psl
cnrm6['member'] = 'CNRM-CM6-1-r0i0p0f0'
rng = np.arange(95,99)
hadgemll = dsP6_clim_mask.isel(member=rng).mean('member').psl
hadgemll['member'] = 'HadGEM3-GC31-LL-r0i0p0f0'
rng = np.arange(0,3)
access_cm2 = dsP6_clim_mask.isel(member=rng).mean('member').psl
access_cm2['member'] = 'ACCESS-CM2-r0i0p0f0'
rng = np.arange(105,111)
ipsl6a = dsP6_clim_mask.isel(member=rng).mean('member').psl
ipsl6a['member'] = 'IPSL-CM6A-LR-r0i0p0f0'
rng = np.arange(3,13)
access_5 = dsP6_clim_mask.isel(member=rng).mean('member').psl
access_5['member'] = 'ACCESS-ESM1-5-r0i0p0f0'
rng = np.arange(192,197)
uk = dsP6_clim_mask.isel(member=rng).mean('member').psl
uk['member'] = 'UKESM1-0-LL-r0i0p0f0'
rng = np.arange(176,186)
mpi2lr = dsP6_clim_mask.isel(member=rng).mean('member').psl
mpi2lr['member'] = 'MPI-ESM1-2-LR-r0i0p0f0'
rng = np.arange(38,88)
canesm5 = dsP6_clim_mask.isel(member=rng).mean('member').psl
canesm5['member'] = 'CanESM5-r0i0p0f0'
rng = np.arange(114,124)
miroce = dsP6_clim_mask.isel(member=rng).mean('member').psl
miroce['member'] = 'MIROC-ES2L-r0i0p0f0'
rng = np.arange(124,174)
miroc6 = dsP6_clim_mask.isel(member=rng).mean('member').psl
miroc6['member'] = 'MIROC6-r0i0p0f0'
rng = np.arange(188,190)
nesm3 = dsP6_clim_mask.isel(member=rng).mean('member').psl
nesm3['member'] = 'NESM3-r0i0p0f0'
rng = np.arange(90,92)
fgoalsg = dsP6_clim_mask.isel(member=rng).mean('member').psl
fgoalsg['member'] = 'FGOALS-g3-r0i0p0f0'
rng = np.arange(111,113)
kace = dsP6_clim_mask.isel(member=rng).mean('member').psl
kace['member'] = 'KACE-1-0-G-r0i0p0f0'
rng = np.arange(14,16)
cas = dsP6_clim_mask.isel(member=rng).mean('member').psl
cas['member'] = 'CAS-ESM2-0-r0i0p0f0'

dsP6_clim_mask_em = xr.concat([access_cm2,access_5,dsP6_clim_mask.sel(member='AWI-CM-1-1-MR-r1i1p1f1').psl,cas,
cesm2_waccm,cesm2,dsP6_clim_mask.sel(member='CMCC-CM2-SR5-r1i1p1f1').psl,dsP6_clim_mask.sel(member='CMCC-ESM2-r1i1p1f1').psl,
dsP6_clim_mask.sel(member='CNRM-CM6-1-HR-r1i1p1f2').psl,cnrm6,cnrm2,canesm5,dsP6_clim_mask.sel(member='E3SM-1-1-r1i1p1f1').psl,
dsP6_clim_mask.sel(member='FGOALS-f3-L-r1i1p1f1').psl,fgoalsg,dsP6_clim_mask.sel(member='GFDL-CM4-r1i1p1f1').psl,
dsP6_clim_mask.sel(member='GFDL-ESM4-r1i1p1f1').psl,dsP6_clim_mask.sel(member='GISS-E2-1-G-r1i1p3f1').psl,hadgemll,
hadgemmm,dsP6_clim_mask.sel(member='INM-CM4-8-r1i1p1f1').psl,dsP6_clim_mask.sel(member='INM-CM5-0-r1i1p1f1').psl,
ipsl6a,kace,dsP6_clim_mask.sel(member='KIOST-ESM-r1i1p1f1').psl,miroce,miroc6,mpihr,mpi2lr,mri2,nesm3,
dsP6_clim_mask.sel(member='NorESM2-MM-r1i1p1f1').psl,dsP6_clim_mask.sel(member='TaiESM1-r1i1p1f1').psl,uk],dim='member')

dsP6_clim_mask_em = dsP6_clim_mask_em.sel(member=dsDelta6_1c_em.member)
dsP6_clim_mask_em = dsP6_clim_mask_em.to_dataset(name='psl')

#################

# Compute Independence Matrix

weights = [np.cos(np.deg2rad(dsT6_clim.lat))]*144
weights = xr.concat(weights, "lon")
weights['lon'] = dsT6_clim.lon

def get_error(ds,weights):
   mod_coords = ds.member.values
   nmod = len(mod_coords)
   res = xr.DataArray(np.empty(shape=(nmod, nmod)),
                       dims=("member", "member_model"), coords=dict(member=mod_coords, member_model=mod_coords))

   for mod1 in ds.transpose("member", ...):
       for mod2 in ds.transpose("member", ...):
           a = xskillscore.rmse(mod1,mod2,dim=['lat','lon'],weights=weights,skipna=True)
           res.loc[dict(member=mod1.member, member_model=mod2.member)] = a

   return res


dsT6_clim_test = get_error(dsT6_clim_mask_em.tas,weights)
dsT6_clim_test = dsT6_clim_test.where(dsT6_clim_test!=0)
dsT6_clim_test_norm = dsT6_clim_test/np.nanmean(dsT6_clim_test)

dsP6_clim_test = get_error(dsP6_clim_mask_em.psl,weights)
dsP6_clim_test = dsP6_clim_test.where(dsP6_clim_test!=0)
dsP6_clim_test_norm = dsP6_clim_test/np.nanmean(dsP6_clim_test)

dsWi6 = (dsT6_clim_test_norm + dsP6_clim_test_norm)/2 ###############

dsWi6 = dsWi6.sel(member=['AWI-CM-1-1-MR-r1i1p1f1','FGOALS-g3-r0i0p0f0',
'HadGEM3-GC31-MM-r0i0p0f0','MIROC-ES2L-r0i0p0f0','UKESM1-0-LL-r0i0p0f0'],
member_model=['AWI-CM-1-1-MR-r1i1p1f1','FGOALS-g3-r0i0p0f0',
'HadGEM3-GC31-MM-r0i0p0f0','MIROC-ES2L-r0i0p0f0','UKESM1-0-LL-r0i0p0f0'])

dsDelta6_1c_em = dsDelta6_1c_em.sel(member=['AWI-CM-1-1-MR-r1i1p1f1','FGOALS-g3-r0i0p0f0',
'HadGEM3-GC31-MM-r0i0p0f0','MIROC-ES2L-r0i0p0f0','UKESM1-0-LL-r0i0p0f0'])

### Top three panels ###
################################################
fig = plt.figure(figsize=(19,5))
ax = plt.subplot(132)
################################################
#plt.suptitle('CMIP6 Central European Summer Case, five model example')
rng = [13]
awi = dsDelta6_1c.isel(member=rng)
plt.plot([-0.5,0.5],[awi.isel(member=0),awi.isel(member=0)],linewidth=2,color='tab:orange')

rng = np.arange(99,103)
hadgemmm = dsDelta6_1c.isel(member=rng)
plt.plot([1.5,2.5],[hadgemmm.isel(member=0),hadgemmm.isel(member=0)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([1.5,2.5],[hadgemmm.isel(member=1),hadgemmm.isel(member=1)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([1.5,2.5],[hadgemmm.isel(member=2),hadgemmm.isel(member=2)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([1.5,2.5],[hadgemmm.isel(member=3),hadgemmm.isel(member=3)],linewidth=2,color='tab:red',alpha=0.3)
f = np.mean(hadgemmm)
plt.plot([2,2],[f,f],"*",markersize=9,color='w')
plt.plot([2,2],[f,f],"*",markersize=7,color='tab:red')

rng = np.arange(192,197)
uk = dsDelta6_1c.isel(member=rng)
plt.plot([3.5,4.5],[uk.isel(member=0),uk.isel(member=0)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([3.5,4.5],[uk.isel(member=1),uk.isel(member=1)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([3.5,4.5],[uk.isel(member=2),uk.isel(member=2)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([3.5,4.5],[uk.isel(member=3),uk.isel(member=3)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([3.5,4.5],[uk.isel(member=4),uk.isel(member=4)],linewidth=2,color='tab:red',alpha=0.3)
f = np.mean(uk)
plt.plot([4,4],[f,f],"*",markersize=9,color='w')
plt.plot([4,4],[f,f],"*",markersize=7,color='tab:red')

rng = np.arange(114,124)
miroce = dsDelta6_1c.isel(member=rng)
plt.plot([5.5,6.5],[miroce.isel(member=0),miroce.isel(member=0)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([5.5,6.5],[miroce.isel(member=1),miroce.isel(member=1)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([5.5,6.5],[miroce.isel(member=2),miroce.isel(member=2)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([5.5,6.5],[miroce.isel(member=3),miroce.isel(member=3)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([5.5,6.5],[miroce.isel(member=4),miroce.isel(member=4)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([5.5,6.5],[miroce.isel(member=5),miroce.isel(member=5)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([5.5,6.5],[miroce.isel(member=6),miroce.isel(member=6)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([5.5,6.5],[miroce.isel(member=7),miroce.isel(member=7)],linewidth=2,color='lightsalmon',alpha=0.3)
f = np.mean(miroce)
plt.plot([6,6],[f,f],"*",markersize=9,color='w')
plt.plot([6,6],[f,f],"*",markersize=7,color='lightsalmon')

rng = np.arange(90,92)
fgoalsg = dsDelta6_1c.isel(member=rng)
plt.plot([7.5,8.5],[fgoalsg.isel(member=0),fgoalsg.isel(member=0)],linewidth=2,color='maroon',alpha=0.3)
plt.plot([7.5,8.5],[fgoalsg.isel(member=1),fgoalsg.isel(member=1)],linewidth=2,color='maroon',alpha=0.3)
f = np.mean(fgoalsg)
plt.plot([8,8],[f,f],"*",markersize=9,color='w')
plt.plot([8,8],[f,f],"*",markersize=7,color='maroon')

plt.xlim([-1,9])
plt.ylim([0.6,1.2])
xticks = np.arange(0,9,2)
ax.set_xticks(xticks)
labels = ['1) AWI-CM-1-1-MR', '2) HadGEM3-GC31-MM',
       '3) UKESM1-0-LL','4) MIROC-ES2L','5) FGOALS-g3']
ax.set_xticklabels(labels,fontsize=9,rotation = 90)
# ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0,1.2],fontsize=12)
plt.ylabel('Aggregate Distance from Observed',fontsize=10)
plt.title('b) Performance' ,fontsize=13,fontweight='bold',loc='left')
#plt.show()

ax = plt.subplot(131)

dsWi6 = dsWi6.assign_coords({"perf": ("member", dsDelta6_1c_em)})
dsWi6 = dsWi6.assign_coords({"perf_2": ("member_model", dsDelta6_1c_em)})
a = dsWi6.sortby(['perf'],ascending=False)
b = a.sortby(['perf_2'],ascending=True)
plt.pcolor(b,vmin=0.5,vmax=2.3,cmap='viridis')
label_1 = b.member_model.data
label_1 = ['1) AWI-CM-1-1-MR', '2) HadGEM3-GC31-MM',
       '3) UKESM1-0-LL','4) MIROC-ES2L','5) FGOALS-g3']
#label_2 = b.member.data
label_2 = ['5) FGOALS-g3', '4) MIROC-ES2L','3) UKESM1-0-LL',
        '2) HadGEM3-GC31-MM','1) AWI-CM-1-1-MR']
ax.set_xticks(np.arange(0.5,5.5))
ax.set_xticklabels(label_1,fontsize=9,rotation=90)
ax.set_yticks(np.arange(0.5,5.5))
ax.set_yticklabels(label_2,fontsize=9)
plt.title('a) Independence',fontsize=13,fontweight='bold',loc='left')
cbar = plt.colorbar()
cbar.set_label('intermember distance',fontsize=10)

ax = plt.subplot(133)
dsT6_target_ts_em['perf'] = dsDelta6_1c_em
dsPr6_target_ts_em['perf']= dsDelta6_1c_em

plt.scatter(dsT6_target_ts_em_norm,dsPr6_target_ts_em_norm,c="silver",s=20,marker='.') #cmap
plt.scatter(dsT6_target_ts_em_norm.sel(member='AWI-CM-1-1-MR-r1i1p1f1'),dsPr6_target_ts_em_norm.sel(member='AWI-CM-1-1-MR-r1i1p1f1'),c="tab:orange",s=40,marker='*') #cmap
plt.scatter(dsT6_target_ts_em_norm.sel(member='FGOALS-g3-r0i0p0f0'),dsPr6_target_ts_em_norm.sel(member='FGOALS-g3-r0i0p0f0'),c="maroon",s=40,marker='*') #cmap
plt.scatter(dsT6_target_ts_em_norm.sel(member='HadGEM3-GC31-MM-r0i0p0f0'),dsPr6_target_ts_em_norm.sel(member='HadGEM3-GC31-MM-r0i0p0f0'),c="tab:red",s=40,marker='*') #cmap
plt.scatter(dsT6_target_ts_em_norm.sel(member='MIROC-ES2L-r0i0p0f0'),dsPr6_target_ts_em_norm.sel(member='MIROC-ES2L-r0i0p0f0'),c="lightsalmon",s=40,marker='*') #cmap
plt.scatter(dsT6_target_ts_em_norm.sel(member='UKESM1-0-LL-r0i0p0f0'),dsPr6_target_ts_em_norm.sel(member='UKESM1-0-LL-r0i0p0f0'),c="tab:red",s=40,marker='*') #cmap

plt.text(x=dsT6_target_ts_em_norm.sel(member='AWI-CM-1-1-MR-r1i1p1f1')+0.05,y=dsPr6_target_ts_em_norm.sel(member='AWI-CM-1-1-MR-r1i1p1f1')+0.05,s='1) AWI-CM-1-1-MR',fontdict=dict(color='k',size=9))
plt.text(x=dsT6_target_ts_em_norm.sel(member='FGOALS-g3-r0i0p0f0')+0.05,y=dsPr6_target_ts_em_norm.sel(member='FGOALS-g3-r0i0p0f0')+0.05,s='5) FGOALS-g3',fontdict=dict(color='k',size=9))
plt.text(x=dsT6_target_ts_em_norm.sel(member='HadGEM3-GC31-MM-r0i0p0f0')+0.05,y=dsPr6_target_ts_em_norm.sel(member='HadGEM3-GC31-MM-r0i0p0f0')+0.05,s='2) HadGEM3-GC31-MM',fontdict=dict(color='k',size=9))
plt.text(x=dsT6_target_ts_em_norm.sel(member='MIROC-ES2L-r0i0p0f0')+0.05,y=dsPr6_target_ts_em_norm.sel(member='MIROC-ES2L-r0i0p0f0')-0.065,s='4) MIROC-ES2L',fontdict=dict(color='k',size=9))
plt.text(x=dsT6_target_ts_em_norm.sel(member='UKESM1-0-LL-r0i0p0f0')+0.05,y=dsPr6_target_ts_em_norm.sel(member='UKESM1-0-LL-r0i0p0f0')+0.05,s='3) UKESM1-0-LL',fontdict=dict(color='k',size=9))

plt.xlabel('Normalized JJA CEU SAT Change (2041/2060 - 1995/2014)',fontsize=10)
plt.xlim([-2,3.5])
plt.ylabel('Normalized JJA CEU PR Change (2041/2060 - 1995/2014)',fontsize=10)
plt.ylim([-2.75,2.75])
plt.title('c) Spread (SAT & PR Change)',fontsize=13,fontweight='bold',loc='left')
# plt.show()
plt.savefig('Fig6_five_model_example_components.png',bbox_inches='tight',dpi=300)
