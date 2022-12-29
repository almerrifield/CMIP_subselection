# Supplementary Figure 10: performance order for all regions/seasons

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


dir = '/**/CMIP_subselection/Data/'

dsSST6_ann = xr.open_dataset(dir + 'tos_mon_CMIP6_SSP585_g025_v2_NAWH_fix_ann.nc',use_cftime = True)
dsSST6_ann = dsSST6_ann.sortby(dsSST6_ann.member)
dsSST6_ann = dsSST6_ann.sel(member=members_cmip6)

dsSW6_ann = xr.open_dataset(dir + 'swcre_mon_CMIP6_SSP585_g025_v2_LWCLD_ann.nc',use_cftime = True)
dsSW6_ann = dsSW6_ann.sortby(dsSW6_ann.member)
dsSW6_ann = dsSW6_ann.sel(member=members_cmip6)

dsPr6_djf = xr.open_dataset(dir + 'pr_mon_CMIP6_SSP585_g025_v2_NEU_obsmask_DJF.nc',use_cftime = True)
dsPr6_djf['pr'] = dsPr6_djf.pr*86400
dsPr6_djf  = dsPr6_djf.sortby(dsPr6_djf.member)
dsPr6_djf = dsPr6_djf.sel(member=members_cmip6)

dsT6_ann = xr.open_dataset(dir + 'tas_mon_CMIP6_SSP585_g025_v2_EUR_ann.nc',use_cftime = True)
dsT6_ann = dsT6_ann-273.15
dsT6_ann = dsT6_ann.sortby(dsT6_ann.member)
dsT6_ann = dsT6_ann.sel(member=members_cmip6)

dsP6_djf = xr.open_dataset(dir + 'psl_mon_CMIP6_SSP585_g025_v2_TW2_DJF.nc',use_cftime = True)
dsP6_djf = dsP6_djf/100
dsP6_djf = dsP6_djf.sortby(dsP6_djf.member)
dsP6_djf = dsP6_djf.sel(member=members_cmip6)

dir = '/**/CMIP_subselection/Data/'

dsSST5_ann = xr.open_dataset(dir + 'tos_mon_CMIP5_rcp85_g025_v2_NAWH_fix_ann.nc',use_cftime = True)
dsSST5_ann = dsSST5_ann-273.15
dsSST5_ann = dsSST5_ann.sortby(dsSST5_ann.member)
dsSST5_ann = dsSST5_ann.sel(member=members_cmip5)

dsSW5_ann = xr.open_dataset(dir + 'swcre_mon_CMIP5_rcp85_g025_v2_LWCLD_ann.nc',use_cftime = True)
dsSW5_ann = dsSW5_ann.sortby(dsSW5_ann.member)
dsSW5_ann = dsSW5_ann.sel(member=members_cmip5)

dsPr5_djf = xr.open_dataset(dir + 'pr_mon_CMIP5_rcp85_g025_v2_NEU_obsmask_DJF.nc',use_cftime = True)
dsPr5_djf['pr'] = dsPr5_djf.pr*86400
dsPr5_djf = dsPr5_djf.sortby(dsPr5_djf.member)
dsPr5_djf = dsPr5_djf.sel(member=members_cmip5)

dsT5_ann = xr.open_dataset(dir + 'tas_mon_CMIP5_rcp85_g025_v2_EUR_ann.nc',use_cftime = True)
dsT5_ann = dsT5_ann-273.15
dsT5_ann = dsT5_ann.sortby(dsT5_ann.member)
dsT5_ann = dsT5_ann.sel(member=members_cmip5)

dsP5_djf = xr.open_dataset(dir + 'psl_mon_CMIP5_rcp85_g025_v2_TW2_DJF.nc',use_cftime = True)
dsP5_djf = dsP5_djf/100
dsP5_djf = dsP5_djf.sortby(dsP5_djf.member)
dsP5_djf = dsP5_djf.sel(member=members_cmip5)

dirobs = '/**/CMIP_subselection/Data/'
dsSSTobs_ann = xr.open_dataset(dirobs + 'tos_mon_OBS_g025_NAWH_fix_ann.nc',use_cftime = True)
dsSWobs_ann = xr.open_dataset(dirobs + 'swcre_mon_OBS_g025_LWCLD_ann.nc',use_cftime = True)
dsPobs_djf = xr.open_dataset(dirobs + 'psl_mon_OBS_g025_TW2_DJF.nc',use_cftime = True)
dsPobs_djf = dsPobs_djf/100
dsProbs_djf = xr.open_dataset(dirobs + 'pr_mon_OBS_g025_NEU_DJF.nc',use_cftime = True)
dsTobs_ann = xr.open_dataset(dirobs + 'tas_mon_OBS_g025_EUR_ann.nc',use_cftime = True)

dsSST5_ann_base = dsSST5_ann.sel(year=slice('1995','2014')).mean('year')
dsSST6_ann_base = dsSST6_ann.sel(year=slice('1995','2014')).mean('year')
dsSSTobs_ann_base = dsSSTobs_ann.sel(year=slice('1995','2014')).mean('year')

dsSW5_ann_base = dsSW5_ann.sel(year=slice('2001','2018')).mean('year')
dsSW6_ann_base = dsSW6_ann.sel(year=slice('2001','2018')).mean('year')
dsSWobs_ann_base = dsSWobs_ann.sel(year=slice('2001','2018')).mean('year')

dsP5_djf_base = dsP5_djf.sel(year=slice('1950','2014')).mean('year')
dsP6_djf_base = dsP6_djf.sel(year=slice('1950','2014')).mean('year')
dsPobs_djf_base = dsPobs_djf.sel(year=slice('1950','2014')).mean('year')

dsPr5_djf_base = dsPr5_djf.sel(year=slice('1995','2014')).mean('year')
dsPr6_djf_base = dsPr6_djf.sel(year=slice('1995','2014')).mean('year')
dsProbs_djf_base = dsProbs_djf.sel(year=slice('1995','2014')).mean('year')

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
dsP5_djf_base_delta = xskillscore.rmse(dsP5_djf_base.psl,dsPobs_djf_base.psl,dim=['lat','lon'],weights=weights,skipna=True)
dsPr5_djf_base_delta = xskillscore.rmse(dsPr5_djf_base.pr,dsProbs_djf_base.pr,dim=['lat','lon'],weights=weights,skipna=True)
dsT5_ann_base_delta = xskillscore.rmse(dsT5_ann_base.tas,dsTobs_ann_base.tas,dim=['lat','lon'],weights=weights,skipna=True)
dsT5_ann_his_delta = xskillscore.rmse(dsT5_ann_his.tas,dsTobs_ann_his.tas,dim=['lat','lon'],weights=weights,skipna=True)

weights = [np.cos(np.deg2rad(dsSST6_ann_base.lat))]*144
weights = xr.concat(weights, "lon")
weights['lon'] = dsSST6_ann_base.lon

dsSST6_ann_base_delta = xskillscore.rmse(dsSST6_ann_base.tos,dsSSTobs_ann_base.tos,dim=['lat','lon'],weights=weights,skipna=True)
dsSW6_ann_base_delta = xskillscore.rmse(dsSW6_ann_base.swcre,dsSWobs_ann_base.swcre,dim=['lat','lon'],weights=weights,skipna=True)
dsP6_djf_base_delta = xskillscore.rmse(dsP6_djf_base.psl,dsPobs_djf_base.psl,dim=['lat','lon'],weights=weights,skipna=True)
dsPr6_djf_base_delta = xskillscore.rmse(dsPr6_djf_base.pr,dsProbs_djf_base.pr,dim=['lat','lon'],weights=weights,skipna=True)
dsT6_ann_base_delta = xskillscore.rmse(dsT6_ann_base.tas,dsTobs_ann_base.tas,dim=['lat','lon'],weights=weights,skipna=True)
dsT6_ann_his_delta = xskillscore.rmse(dsT6_ann_his.tas,dsTobs_ann_his.tas,dim=['lat','lon'],weights=weights,skipna=True)

dsSST_ann_base_delta = xr.concat([dsSST5_ann_base_delta,dsSST6_ann_base_delta],dim='member')
dsSW_ann_base_delta = xr.concat([dsSW5_ann_base_delta,dsSW6_ann_base_delta],dim='member')
dsP_djf_base_delta = xr.concat([dsP5_djf_base_delta,dsP6_djf_base_delta],dim='member')
dsPr_djf_base_delta = xr.concat([dsPr5_djf_base_delta,dsPr6_djf_base_delta],dim='member')
dsT_ann_base_delta = xr.concat([dsT5_ann_base_delta,dsT6_ann_base_delta],dim='member')
dsT_ann_his_delta = xr.concat([dsT5_ann_his_delta,dsT6_ann_his_delta],dim='member')

## Normalize

dsSST5_ann_base_norm = dsSST5_ann_base_delta/dsSST_ann_base_delta.mean('member')
dsSW5_ann_base_norm = dsSW5_ann_base_delta/dsSW_ann_base_delta.mean('member')
dsP5_djf_base_norm = dsP5_djf_base_delta/dsP_djf_base_delta.mean('member')
dsPr5_djf_base_norm = dsPr5_djf_base_delta/dsPr_djf_base_delta.mean('member')
dsT5_ann_base_norm = dsT5_ann_base_delta/dsT_ann_base_delta.mean('member')
dsT5_ann_his_norm = dsT5_ann_his_delta/dsT_ann_his_delta.mean('member')

dsSST6_ann_base_norm = dsSST6_ann_base_delta/dsSST_ann_base_delta.mean('member')
dsSW6_ann_base_norm = dsSW6_ann_base_delta/dsSW_ann_base_delta.mean('member')
dsP6_djf_base_norm = dsP6_djf_base_delta/dsP_djf_base_delta.mean('member')
dsPr6_djf_base_norm = dsPr6_djf_base_delta/dsPr_djf_base_delta.mean('member')
dsT6_ann_base_norm = dsT6_ann_base_delta/dsT_ann_base_delta.mean('member')
dsT6_ann_his_norm = dsT6_ann_his_delta/dsT_ann_his_delta.mean('member')

# Compute Predictor
dsDelta5_1c = (dsSST5_ann_base_norm + dsSW5_ann_base_norm + dsP5_djf_base_norm + dsPr5_djf_base_norm  + dsT5_ann_his_norm + dsT5_ann_base_norm)/6
dsDelta6_1c = (dsSST6_ann_base_norm + dsSW6_ann_base_norm + dsP6_djf_base_norm + dsPr6_djf_base_norm + dsT6_ann_his_norm + dsT6_ann_base_norm)/6

dsDelta6_1c.argsort()
dsDelta6_1c.isel(member=[14]).member

### CMIP6 ###
################################################
fig = plt.figure(figsize=(14,4))
ax = plt.subplot(121)
################################################

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

rng = [190]
noresm2mm = dsDelta6_1c.isel(member=rng)
plt.plot([3.5,4.5],[noresm2mm.isel(member=0),noresm2mm.isel(member=0)],linewidth=2,color='darkgoldenrod')


rng = np.arange(174,176)
mpihr = dsDelta6_1c.isel(member=rng)
plt.plot([5.5,6.5],[mpihr.isel(member=0),mpihr.isel(member=0)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([5.5,6.5],[mpihr.isel(member=1),mpihr.isel(member=1)],linewidth=2,color='tab:orange',alpha=0.3)
f = np.mean(mpihr)
plt.plot([6,6],[f,f],"*",markersize=9,color='w')
plt.plot([6,6],[f,f],"*",markersize=7,color='tab:orange')

rng = np.arange(111,113)
kace = dsDelta6_1c.isel(member=rng)
plt.plot([7.5,8.5],[kace.isel(member=0),kace.isel(member=0)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([7.5,8.5],[kace.isel(member=1),kace.isel(member=1)],linewidth=2,color='tab:red',alpha=0.3)
f = np.mean(kace)
plt.plot([8,8],[f,f],"*",markersize=9,color='w')
plt.plot([8,8],[f,f],"*",markersize=7,color='tab:red')

rng = np.arange(33,38)
cnrm2 = dsDelta6_1c.isel(member=rng)
plt.plot([9.5,10.5],[cnrm2.isel(member=0),cnrm2.isel(member=0)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([9.5,10.5],[cnrm2.isel(member=1),cnrm2.isel(member=1)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([9.5,10.5],[cnrm2.isel(member=2),cnrm2.isel(member=2)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([9.5,10.5],[cnrm2.isel(member=3),cnrm2.isel(member=3)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([9.5,10.5],[cnrm2.isel(member=4),cnrm2.isel(member=4)],linewidth=2,color='cornflowerblue',alpha=0.3)
f = np.mean(cnrm2)
plt.plot([10,10],[f,f],"*",markersize=9,color='w')
plt.plot([10,10],[f,f],"*",markersize=7,color='cornflowerblue')

rng = np.arange(186,188)
mri2 = dsDelta6_1c.isel(member=rng)
plt.plot([11.5,12.5],[mri2.isel(member=0),mri2.isel(member=0)],linewidth=2,color='palevioletred',alpha=0.3)
plt.plot([11.5,12.5],[mri2.isel(member=1),mri2.isel(member=1)],linewidth=2,color='palevioletred',alpha=0.3)
f = np.mean(mri2)
plt.plot([12,12],[f,f],"*",markersize=9,color='w')
plt.plot([12,12],[f,f],"*",markersize=7,color='palevioletred')

rng = np.arange(27,33)
cnrm6 = dsDelta6_1c.isel(member=rng)
plt.plot([13.5,14.5],[cnrm6.isel(member=0),cnrm6.isel(member=0)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([13.5,14.5],[cnrm6.isel(member=1),cnrm6.isel(member=1)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([13.5,14.5],[cnrm6.isel(member=2),cnrm6.isel(member=2)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([13.5,14.5],[cnrm6.isel(member=3),cnrm6.isel(member=3)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([13.5,14.5],[cnrm6.isel(member=4),cnrm6.isel(member=4)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([13.5,14.5],[cnrm6.isel(member=5),cnrm6.isel(member=5)],linewidth=2,color='cornflowerblue',alpha=0.3)
f = np.mean(cnrm6)
plt.plot([14,14],[f,f],"*",markersize=9,color='w')
plt.plot([14,14],[f,f],"*",markersize=7,color='cornflowerblue')

rng = [93]
gfdle4 = dsDelta6_1c.isel(member=rng)
plt.plot([15.5,16.5],[gfdle4.isel(member=0),gfdle4.isel(member=0)],linewidth=2,color='indigo')

rng = [191]
tai = dsDelta6_1c.isel(member=rng)
plt.plot([17.5,18.5],[tai.isel(member=0),tai.isel(member=0)],linewidth=2,color='darkgoldenrod')

rng = [92]
gfdl4 = dsDelta6_1c.isel(member=rng)
plt.plot([19.5,20.5],[gfdl4.isel(member=0),gfdl4.isel(member=0)],linewidth=2,color='indigo')

rng = [25]
cmcc2 = dsDelta6_1c.isel(member=rng)
plt.plot([21.5,22.5],[cmcc2.isel(member=0),cmcc2.isel(member=0)],linewidth=2,color='darkgoldenrod')

rng = np.arange(0,3)
access_cm2 = dsDelta6_1c.isel(member=rng)
plt.plot([23.5,24.5],[access_cm2.isel(member=0),access_cm2.isel(member=0)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([23.5,24.5],[access_cm2.isel(member=1),access_cm2.isel(member=1)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([23.5,24.5],[access_cm2.isel(member=2),access_cm2.isel(member=2)],linewidth=2,color='tab:red',alpha=0.3)
f = np.mean(access_cm2)
plt.plot([24,24],[f,f],"*",markersize=9,color='w')
plt.plot([24,24],[f,f],"*",markersize=7,color='tab:red')

rng = np.arange(95,99)
hadgemll = dsDelta6_1c.isel(member=rng)
plt.plot([25.5,26.5],[hadgemll.isel(member=0),hadgemll.isel(member=0)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([25.5,26.5],[hadgemll.isel(member=1),hadgemll.isel(member=1)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([25.5,26.5],[hadgemll.isel(member=2),hadgemll.isel(member=2)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([25.5,26.5],[hadgemll.isel(member=3),hadgemll.isel(member=3)],linewidth=2,color='tab:red',alpha=0.3)
f = np.mean(hadgemll)
plt.plot([26,26],[f,f],"*",markersize=9,color='w')
plt.plot([26,26],[f,f],"*",markersize=7,color='tab:red')

rng = np.arange(16,19)
cesm2_waccm = dsDelta6_1c.isel(member=rng)
plt.plot([27.5,28.5],[cesm2_waccm.isel(member=0),cesm2_waccm.isel(member=0)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([27.5,28.5],[cesm2_waccm.isel(member=1),cesm2_waccm.isel(member=1)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([27.5,28.5],[cesm2_waccm.isel(member=2),cesm2_waccm.isel(member=2)],linewidth=2,color='darkgoldenrod',alpha=0.3)
f = np.mean(cesm2_waccm)
plt.plot([28,28],[f,f],"*",markersize=9,color='w')
plt.plot([28,28],[f,f],"*",markersize=7,color='darkgoldenrod')

rng = [104]
inm50 = dsDelta6_1c.isel(member=rng)
plt.plot([29.5,30.5],[inm50.isel(member=0),inm50.isel(member=0)],linewidth=2,color='mediumseagreen')

rng = np.arange(176,186)
mpi2lr = dsDelta6_1c.isel(member=rng)
plt.plot([31.5,32.5],[mpi2lr.isel(member=0),mpi2lr.isel(member=0)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([31.5,32.5],[mpi2lr.isel(member=1),mpi2lr.isel(member=1)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([31.5,32.5],[mpi2lr.isel(member=2),mpi2lr.isel(member=2)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([31.5,32.5],[mpi2lr.isel(member=3),mpi2lr.isel(member=3)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([31.5,32.5],[mpi2lr.isel(member=4),mpi2lr.isel(member=4)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([31.5,32.5],[mpi2lr.isel(member=5),mpi2lr.isel(member=5)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([31.5,32.5],[mpi2lr.isel(member=6),mpi2lr.isel(member=6)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([31.5,32.5],[mpi2lr.isel(member=7),mpi2lr.isel(member=7)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([31.5,32.5],[mpi2lr.isel(member=8),mpi2lr.isel(member=8)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([31.5,32.5],[mpi2lr.isel(member=9),mpi2lr.isel(member=9)],linewidth=2,color='tab:orange',alpha=0.3)
f = np.mean(mpi2lr)
plt.plot([32,32],[f,f],"*",markersize=9,color='w')
plt.plot([32,32],[f,f],"*",markersize=7,color='tab:orange')

rng = np.arange(3,13)
access_5 = dsDelta6_1c.isel(member=rng)
plt.plot([33.5,34.5],[access_5.isel(member=0),access_5.isel(member=0)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([33.5,34.5],[access_5.isel(member=1),access_5.isel(member=1)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([33.5,34.5],[access_5.isel(member=2),access_5.isel(member=2)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([33.5,34.5],[access_5.isel(member=3),access_5.isel(member=3)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([33.5,34.5],[access_5.isel(member=4),access_5.isel(member=4)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([33.5,34.5],[access_5.isel(member=5),access_5.isel(member=5)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([33.5,34.5],[access_5.isel(member=6),access_5.isel(member=6)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([33.5,34.5],[access_5.isel(member=7),access_5.isel(member=7)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([33.5,34.5],[access_5.isel(member=8),access_5.isel(member=8)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([33.5,34.5],[access_5.isel(member=9),access_5.isel(member=9)],linewidth=2,color='tab:red',alpha=0.3)
f = np.mean(access_5)
plt.plot([34,34],[f,f],"*",markersize=9,color='w')
plt.plot([34,34],[f,f],"*",markersize=7,color='tab:red')

rng = np.arange(192,197)
uk = dsDelta6_1c.isel(member=rng)
plt.plot([35.5,36.5],[uk.isel(member=0),uk.isel(member=0)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([35.5,36.5],[uk.isel(member=1),uk.isel(member=1)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([35.5,36.5],[uk.isel(member=2),uk.isel(member=2)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([35.5,36.5],[uk.isel(member=3),uk.isel(member=3)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([35.5,36.5],[uk.isel(member=4),uk.isel(member=4)],linewidth=2,color='tab:red',alpha=0.3)
f = np.mean(uk)
plt.plot([36,36],[f,f],"*",markersize=9,color='w')
plt.plot([36,36],[f,f],"*",markersize=7,color='tab:red')

rng = [24]
cmcc5 = dsDelta6_1c.isel(member=rng)
plt.plot([37.5,38.5],[cmcc5.isel(member=0),cmcc5.isel(member=0)],linewidth=2,color='darkgoldenrod')

rng = np.arange(19,24)
cesm2 = dsDelta6_1c.isel(member=rng)
plt.plot([39.5,40.5],[cesm2.isel(member=0),cesm2.isel(member=0)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([39.5,40.5],[cesm2.isel(member=1),cesm2.isel(member=1)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([39.5,40.5],[cesm2.isel(member=2),cesm2.isel(member=2)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([39.5,40.5],[cesm2.isel(member=3),cesm2.isel(member=3)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([39.5,40.5],[cesm2.isel(member=4),cesm2.isel(member=4)],linewidth=2,color='darkgoldenrod',alpha=0.3)
f = np.mean(cesm2)
plt.plot([40,40],[f,f],"*",markersize=9,color='w')
plt.plot([40,40],[f,f],"*",markersize=7,color='darkgoldenrod')

rng = np.arange(105,111)
ipsl6a = dsDelta6_1c.isel(member=rng)
plt.plot([41.5,42.5],[ipsl6a.isel(member=0),ipsl6a.isel(member=0)],linewidth=2,color='royalblue',alpha=0.3)
plt.plot([41.5,42.5],[ipsl6a.isel(member=1),ipsl6a.isel(member=1)],linewidth=2,color='royalblue',alpha=0.3)
plt.plot([41.5,42.5],[ipsl6a.isel(member=2),ipsl6a.isel(member=2)],linewidth=2,color='royalblue',alpha=0.3)
plt.plot([41.5,42.5],[ipsl6a.isel(member=3),ipsl6a.isel(member=3)],linewidth=2,color='royalblue',alpha=0.3)
plt.plot([41.5,42.5],[ipsl6a.isel(member=4),ipsl6a.isel(member=4)],linewidth=2,color='royalblue',alpha=0.3)
plt.plot([41.5,42.5],[ipsl6a.isel(member=5),ipsl6a.isel(member=5)],linewidth=2,color='royalblue',alpha=0.3)
f = np.mean(ipsl6a)
plt.plot([42,42],[f,f],"*",markersize=9,color='w')
plt.plot([42,42],[f,f],"*",markersize=7,color='royalblue')

rng = [103]
inm48 = dsDelta6_1c.isel(member=rng)
plt.plot([43.5,44.5],[inm48.isel(member=0),inm48.isel(member=0)],linewidth=2,color='mediumseagreen')

rng = [26]
cnrm6_hr = dsDelta6_1c.isel(member=rng)
plt.plot([45.5,46.5],[cnrm6_hr.isel(member=0),cnrm6_hr.isel(member=0)],linewidth=2,color='cornflowerblue')

rng = [94]
gissg = dsDelta6_1c.isel(member=rng)
plt.plot([47.5,48.5],[gissg.isel(member=0),gissg.isel(member=0)],linewidth=2,color='blueviolet')

rng = np.arange(188,190)
nesm3 = dsDelta6_1c.isel(member=rng)
plt.plot([49.5,50.5],[nesm3.isel(member=0),nesm3.isel(member=0)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([49.5,50.5],[nesm3.isel(member=1),nesm3.isel(member=1)],linewidth=2,color='tab:orange',alpha=0.3)
f = np.mean(nesm3)
plt.plot([50,50],[f,f],"*",markersize=9,color='w')
plt.plot([50,50],[f,f],"*",markersize=7,color='tab:orange')

rng = np.arange(38,88)
canesm5 = dsDelta6_1c.isel(member=rng)
plt.plot([51.5,52.5],[canesm5.isel(member=0),canesm5.isel(member=0)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=1),canesm5.isel(member=1)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=2),canesm5.isel(member=2)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=3),canesm5.isel(member=3)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=4),canesm5.isel(member=4)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=5),canesm5.isel(member=5)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=6),canesm5.isel(member=6)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=7),canesm5.isel(member=7)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=8),canesm5.isel(member=8)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=9),canesm5.isel(member=9)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=10),canesm5.isel(member=10)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=11),canesm5.isel(member=11)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=12),canesm5.isel(member=12)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=13),canesm5.isel(member=13)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=14),canesm5.isel(member=14)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=15),canesm5.isel(member=15)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=16),canesm5.isel(member=16)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=17),canesm5.isel(member=17)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=18),canesm5.isel(member=18)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=19),canesm5.isel(member=19)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=20),canesm5.isel(member=20)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=21),canesm5.isel(member=21)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=22),canesm5.isel(member=22)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=23),canesm5.isel(member=23)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=24),canesm5.isel(member=24)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=25),canesm5.isel(member=25)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=26),canesm5.isel(member=26)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=27),canesm5.isel(member=27)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=28),canesm5.isel(member=28)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=29),canesm5.isel(member=29)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=30),canesm5.isel(member=30)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=31),canesm5.isel(member=31)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=32),canesm5.isel(member=32)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=33),canesm5.isel(member=33)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=34),canesm5.isel(member=34)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=35),canesm5.isel(member=35)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=36),canesm5.isel(member=36)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=37),canesm5.isel(member=37)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=38),canesm5.isel(member=38)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=39),canesm5.isel(member=39)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=40),canesm5.isel(member=40)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=41),canesm5.isel(member=41)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=42),canesm5.isel(member=42)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=43),canesm5.isel(member=43)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=44),canesm5.isel(member=44)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=45),canesm5.isel(member=45)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=46),canesm5.isel(member=46)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=47),canesm5.isel(member=47)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=48),canesm5.isel(member=48)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([51.5,52.5],[canesm5.isel(member=49),canesm5.isel(member=49)],linewidth=2,color='dodgerblue',alpha=0.3)
f = np.mean(canesm5)
plt.plot([52,52],[f,f],"*",markersize=9,color='w')
plt.plot([52,52],[f,f],"*",markersize=7,color='dodgerblue')

rng = [88]
esm = dsDelta6_1c.isel(member=rng)
plt.plot([53.5,54.5],[esm.isel(member=0),esm.isel(member=0)],linewidth=2,color='k')

rng = np.arange(90,92)
fgoalsg = dsDelta6_1c.isel(member=rng)
plt.plot([55.5,56.5],[fgoalsg.isel(member=0),fgoalsg.isel(member=0)],linewidth=2,color='maroon',alpha=0.3)
plt.plot([55.5,56.5],[fgoalsg.isel(member=1),fgoalsg.isel(member=1)],linewidth=2,color='maroon',alpha=0.3)
f = np.mean(fgoalsg)
plt.plot([56,56],[f,f],"*",markersize=9,color='w')
plt.plot([56,56],[f,f],"*",markersize=7,color='maroon')

rng = np.arange(114,124)
miroce = dsDelta6_1c.isel(member=rng)
plt.plot([57.5,58.5],[miroce.isel(member=0),miroce.isel(member=0)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([57.5,58.5],[miroce.isel(member=1),miroce.isel(member=1)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([57.5,58.5],[miroce.isel(member=2),miroce.isel(member=2)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([57.5,58.5],[miroce.isel(member=3),miroce.isel(member=3)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([57.5,58.5],[miroce.isel(member=4),miroce.isel(member=4)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([57.5,58.5],[miroce.isel(member=5),miroce.isel(member=5)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([57.5,58.5],[miroce.isel(member=6),miroce.isel(member=6)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([57.5,58.5],[miroce.isel(member=7),miroce.isel(member=7)],linewidth=2,color='lightsalmon',alpha=0.3)
f = np.mean(miroce)
plt.plot([58,58],[f,f],"*",markersize=9,color='w')
plt.plot([58,58],[f,f],"*",markersize=7,color='lightsalmon')


rng = np.arange(124,174)
miroc6 = dsDelta6_1c.isel(member=rng)
plt.plot([59.5,60.5],[miroc6.isel(member=0),miroc6.isel(member=0)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=1),miroc6.isel(member=1)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=2),miroc6.isel(member=2)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=3),miroc6.isel(member=3)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=4),miroc6.isel(member=4)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=5),miroc6.isel(member=5)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=6),miroc6.isel(member=6)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=7),miroc6.isel(member=7)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=8),miroc6.isel(member=8)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=9),miroc6.isel(member=9)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=10),miroc6.isel(member=10)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=11),miroc6.isel(member=11)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=12),miroc6.isel(member=12)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=13),miroc6.isel(member=13)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=14),miroc6.isel(member=14)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=15),miroc6.isel(member=15)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=16),miroc6.isel(member=16)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=17),miroc6.isel(member=17)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=18),miroc6.isel(member=18)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=19),miroc6.isel(member=19)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=20),miroc6.isel(member=20)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=21),miroc6.isel(member=21)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=22),miroc6.isel(member=22)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=23),miroc6.isel(member=23)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=24),miroc6.isel(member=24)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=25),miroc6.isel(member=25)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=26),miroc6.isel(member=26)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=27),miroc6.isel(member=27)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=28),miroc6.isel(member=28)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=29),miroc6.isel(member=29)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=30),miroc6.isel(member=30)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=31),miroc6.isel(member=31)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=32),miroc6.isel(member=32)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=33),miroc6.isel(member=33)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=34),miroc6.isel(member=34)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=35),miroc6.isel(member=35)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=36),miroc6.isel(member=36)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=37),miroc6.isel(member=37)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=38),miroc6.isel(member=38)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=39),miroc6.isel(member=39)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=40),miroc6.isel(member=40)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=41),miroc6.isel(member=41)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=42),miroc6.isel(member=42)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=43),miroc6.isel(member=43)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=44),miroc6.isel(member=44)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=45),miroc6.isel(member=45)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=46),miroc6.isel(member=46)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=47),miroc6.isel(member=47)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=48),miroc6.isel(member=48)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([59.5,60.5],[miroc6.isel(member=49),miroc6.isel(member=49)],linewidth=2,color='lightsalmon',alpha=0.3)
f = np.mean(miroc6)
plt.plot([60,60],[f,f],"*",markersize=9,color='w')
plt.plot([60,60],[f,f],"*",markersize=7,color='lightsalmon')

rng = [113]
kiost = dsDelta6_1c.isel(member=rng)
plt.plot([61.5,62.5],[kiost.isel(member=0),kiost.isel(member=0)],linewidth=2,color='darkslateblue')

rng = [89]
fgoalsf = dsDelta6_1c.isel(member=rng)
plt.plot([63.5,64.5],[fgoalsf.isel(member=0),fgoalsf.isel(member=0)],linewidth=2,color='maroon')


rng = np.arange(14,16)
cas = dsDelta6_1c.isel(member=rng)
plt.plot([65.5,66.5],[cas.isel(member=0),cas.isel(member=0)],linewidth=2,color='tab:cyan',alpha=0.3)
plt.plot([65.5,66.5],[cas.isel(member=1),cas.isel(member=1)],linewidth=2,color='tab:cyan',alpha=0.3)
f = np.mean(cas)
plt.plot([66,66],[f,f],"*",markersize=9,color='w')
plt.plot([66,66],[f,f],"*",markersize=7,color='tab:cyan')

plt.xlim([-1,67])
plt.ylim([0.5,2])
xticks = np.arange(0,67,2)
ax.set_xticks(xticks)
labels = ['AWI-CM-1-1-MR', 'HadGEM3-GC31-MM','NorESM2-MM',
       'MPI-ESM1-2-HR','KACE-1-0-G','CNRM-ESM2-1', 'MRI-ESM2-0',
       'CNRM-CM6-1', 'GFDL-ESM4', 'TaiESM1','GFDL-CM4','CMCC-ESM2',
       'ACCESS-CM2','HadGEM3-GC31-LL','CESM2-WACCM','INM-CM5-0',
       'MPI-ESM1-2-LR','ACCESS-ESM1-5','UKESM1-0-LL','CMCC-CM2-SR5',
       'CESM2','IPSL-CM6A-LR','INM-CM4-8','CNRM-CM6-1-HR','GISS-E2-1-G',
       'NESM3','CanESM5','E3SM-1-1','FGOALS-g3','MIROC-ES2L','MIROC6',
       'KIOST-ESM', 'FGOALS-f3-L',  'CAS-ESM2-0']
ax.set_xticklabels(labels,fontsize=8,rotation = 90)
ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0,1.2],fontsize=12)
plt.ylabel('Aggregate Distance from Observed',fontsize=10)
plt.title('c) CMIP6 Northern European Winter Case' ,fontsize=14,fontweight='bold',loc='left')


### CMIP5 ###
################################################
ax = plt.subplot(122)
################################################

rng = [0]
access_0 = dsDelta5_1c.isel(member=rng)
plt.plot([-0.5,0.5],[access_0.isel(member=0),access_0.isel(member=0)],linewidth=2,color='tab:red')

rng = np.arange(8,11)
cesm1 = dsDelta5_1c.isel(member=rng)
plt.plot([1.5,2.5],[cesm1.isel(member=0),cesm1.isel(member=0)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([1.5,2.5],[cesm1.isel(member=1),cesm1.isel(member=1)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([1.5,2.5],[cesm1.isel(member=2),cesm1.isel(member=2)],linewidth=2,color='darkgoldenrod',alpha=0.3)
f = np.mean(cesm1)
plt.plot([2,2],[f,f],"*",markersize=9,color='w')
plt.plot([2,2],[f,f],"*",markersize=7,color='darkgoldenrod')

rng = [31]
gfdl3 = dsDelta5_1c.isel(member=rng)
plt.plot([3.5,4.5],[gfdl3.isel(member=0),gfdl3.isel(member=0)],linewidth=2,color='indigo')

rng = np.arange(44,48)
hadgemes = dsDelta5_1c.isel(member=rng)
plt.plot([5.5,6.5],[hadgemes.isel(member=0),hadgemes.isel(member=0)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([5.5,6.5],[hadgemes.isel(member=1),hadgemes.isel(member=1)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([5.5,6.5],[hadgemes.isel(member=2),hadgemes.isel(member=2)],linewidth=2,color='tab:red',alpha=0.3)
plt.plot([5.5,6.5],[hadgemes.isel(member=3),hadgemes.isel(member=3)],linewidth=2,color='tab:red',alpha=0.3)
f = np.mean(hadgemes)
plt.plot([6,6],[f,f],"*",markersize=9,color='w')
plt.plot([6,6],[f,f],"*",markersize=7,color='tab:red')

rng = np.arange(58,61)
mpilr = dsDelta5_1c.isel(member=rng)
plt.plot([7.5,8.5],[mpilr.isel(member=0),mpilr.isel(member=0)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([7.5,8.5],[mpilr.isel(member=1),mpilr.isel(member=1)],linewidth=2,color='tab:orange',alpha=0.3)
plt.plot([7.5,8.5],[mpilr.isel(member=2),mpilr.isel(member=2)],linewidth=2,color='tab:orange',alpha=0.3)
f = np.mean(mpilr)
plt.plot([8,8],[f,f],"*",markersize=9,color='w')
plt.plot([8,8],[f,f],"*",markersize=7,color='tab:orange')

rng = [61]
mpimr = dsDelta5_1c.isel(member=rng)
plt.plot([9.5,10.5],[mpimr.isel(member=0),mpimr.isel(member=0)],linewidth=2,color='tab:orange')

rng = [1]
access_3 = dsDelta5_1c.isel(member=rng)
plt.plot([11.5,12.5],[access_3.isel(member=0),access_3.isel(member=0)],linewidth=2,color='tab:red')

rng = [33]
gfdlm = dsDelta5_1c.isel(member=rng)
plt.plot([13.5,14.5],[gfdlm.isel(member=0),gfdlm.isel(member=0)],linewidth=2,color='indigo')

rng = [64]
noresm1me = dsDelta5_1c.isel(member=rng)
plt.plot([15.5,16.5],[noresm1me.isel(member=0),noresm1me.isel(member=0)],linewidth=2,color='darkgoldenrod')

rng = np.arange(55,58)
miroc5 = dsDelta5_1c.isel(member=rng)
plt.plot([17.5,18.5],[miroc5.isel(member=0),miroc5.isel(member=0)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([17.5,18.5],[miroc5.isel(member=1),miroc5.isel(member=1)],linewidth=2,color='lightsalmon',alpha=0.3)
plt.plot([17.5,18.5],[miroc5.isel(member=2),miroc5.isel(member=2)],linewidth=2,color='lightsalmon',alpha=0.3)
f = np.mean(miroc5)
plt.plot([18,18],[f,f],"*",markersize=9,color='w')
plt.plot([18,18],[f,f],"*",markersize=7,color='lightsalmon')

rng = [63]
noresm1m = dsDelta5_1c.isel(member=rng)
plt.plot([19.5,20.5],[noresm1m.isel(member=0),noresm1m.isel(member=0)],linewidth=2,color='darkgoldenrod')

rng = [32]
gfdlg = dsDelta5_1c.isel(member=rng)
plt.plot([21.5,22.5],[gfdlg.isel(member=0),gfdlg.isel(member=0)],linewidth=2,color='indigo')

rng = [67]
inmcm = dsDelta5_1c.isel(member=rng)
plt.plot([23.5,24.5],[inmcm.isel(member=0),inmcm.isel(member=0)],linewidth=2,color='mediumseagreen')

rng = np.arange(26,31)
canesm2 = dsDelta5_1c.isel(member=rng)
plt.plot([25.5,26.5],[canesm2.isel(member=0),canesm2.isel(member=0)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([25.5,26.5],[canesm2.isel(member=1),canesm2.isel(member=1)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([25.5,26.5],[canesm2.isel(member=2),canesm2.isel(member=2)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([25.5,26.5],[canesm2.isel(member=3),canesm2.isel(member=3)],linewidth=2,color='dodgerblue',alpha=0.3)
plt.plot([25.5,26.5],[canesm2.isel(member=4),canesm2.isel(member=4)],linewidth=2,color='dodgerblue',alpha=0.3)
f = np.mean(canesm2)
plt.plot([26,26],[f,f],"*",markersize=9,color='w')
plt.plot([26,26],[f,f],"*",markersize=7,color='dodgerblue')

rng = np.arange(11,16)
cnrm5 = dsDelta5_1c.isel(member=rng)
plt.plot([27.5,28.5],[cnrm5.isel(member=0),cnrm5.isel(member=0)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([27.5,28.5],[cnrm5.isel(member=1),cnrm5.isel(member=1)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([27.5,28.5],[cnrm5.isel(member=2),cnrm5.isel(member=2)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([27.5,28.5],[cnrm5.isel(member=3),cnrm5.isel(member=3)],linewidth=2,color='cornflowerblue',alpha=0.3)
plt.plot([27.5,28.5],[cnrm5.isel(member=4),cnrm5.isel(member=4)],linewidth=2,color='cornflowerblue',alpha=0.3)
f = np.mean(cnrm5)
plt.plot([28,28],[f,f],"*",markersize=9,color='w')
plt.plot([28,28],[f,f],"*",markersize=7,color='cornflowerblue')

rng = np.arange(39,44)
gissr = dsDelta5_1c.isel(member=rng)
plt.plot([29.5,30.5],[gissr.isel(member=0),gissr.isel(member=0)],linewidth=2,color='blueviolet',alpha=0.3)
plt.plot([29.5,30.5],[gissr.isel(member=1),gissr.isel(member=1)],linewidth=2,color='blueviolet',alpha=0.3)
plt.plot([29.5,30.5],[gissr.isel(member=2),gissr.isel(member=2)],linewidth=2,color='blueviolet',alpha=0.3)
plt.plot([29.5,30.5],[gissr.isel(member=3),gissr.isel(member=3)],linewidth=2,color='blueviolet',alpha=0.3)
plt.plot([29.5,30.5],[gissr.isel(member=4),gissr.isel(member=4)],linewidth=2,color='blueviolet',alpha=0.3)
f = np.mean(gissr)
plt.plot([30,30],[f,f],"*",markersize=9,color='w')
plt.plot([30,30],[f,f],"*",markersize=7,color='blueviolet')

rng = np.arange(2,8)
ccsm4 = dsDelta5_1c.isel(member=rng)
plt.plot([31.5,32.5],[ccsm4.isel(member=0),ccsm4.isel(member=0)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([31.5,32.5],[ccsm4.isel(member=1),ccsm4.isel(member=1)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([31.5,32.5],[ccsm4.isel(member=2),ccsm4.isel(member=2)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([31.5,32.5],[ccsm4.isel(member=3),ccsm4.isel(member=3)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([31.5,32.5],[ccsm4.isel(member=4),ccsm4.isel(member=4)],linewidth=2,color='darkgoldenrod',alpha=0.3)
plt.plot([31.5,32.5],[ccsm4.isel(member=5),ccsm4.isel(member=5)],linewidth=2,color='darkgoldenrod',alpha=0.3)
f = np.mean(ccsm4)
plt.plot([32,32],[f,f],"*",markersize=9,color='w')
plt.plot([32,32],[f,f],"*",markersize=7,color='darkgoldenrod')

rng = np.arange(34,39)
gissh = dsDelta5_1c.isel(member=rng)
plt.plot([33.5,34.5],[gissh.isel(member=0),gissh.isel(member=0)],linewidth=2,color='blueviolet',alpha=0.3)
plt.plot([33.5,34.5],[gissh.isel(member=1),gissh.isel(member=1)],linewidth=2,color='blueviolet',alpha=0.3)
plt.plot([33.5,34.5],[gissh.isel(member=2),gissh.isel(member=2)],linewidth=2,color='blueviolet',alpha=0.3)
plt.plot([33.5,34.5],[gissh.isel(member=3),gissh.isel(member=3)],linewidth=2,color='blueviolet',alpha=0.3)
plt.plot([33.5,34.5],[gissh.isel(member=4),gissh.isel(member=4)],linewidth=2,color='blueviolet',alpha=0.3)
f = np.mean(gissh)
plt.plot([34,34],[f,f],"*",markersize=9,color='w')
plt.plot([34,34],[f,f],"*",markersize=7,color='blueviolet')

rng = [66]
bcc = dsDelta5_1c.isel(member=rng)
plt.plot([35.5,36.5],[bcc.isel(member=0),bcc.isel(member=0)],linewidth=2,color='silver')

rng = [65]
bccm = dsDelta5_1c.isel(member=rng)
plt.plot([37.5,38.5],[bccm.isel(member=0),bccm.isel(member=0)],linewidth=2,color='silver')

rng = np.arange(16,26)
csiro = dsDelta5_1c.isel(member=rng)
plt.plot([39.5,40.5],[csiro.isel(member=0),csiro.isel(member=0)],linewidth=2,color='deeppink',alpha=0.3)
plt.plot([39.5,40.5],[csiro.isel(member=1),csiro.isel(member=1)],linewidth=2,color='deeppink',alpha=0.3)
plt.plot([39.5,40.5],[csiro.isel(member=2),csiro.isel(member=2)],linewidth=2,color='deeppink',alpha=0.3)
plt.plot([39.5,40.5],[csiro.isel(member=3),csiro.isel(member=3)],linewidth=2,color='deeppink',alpha=0.3)
plt.plot([39.5,40.5],[csiro.isel(member=4),csiro.isel(member=4)],linewidth=2,color='deeppink',alpha=0.3)
plt.plot([39.5,40.5],[csiro.isel(member=5),csiro.isel(member=5)],linewidth=2,color='deeppink',alpha=0.3)
plt.plot([39.5,40.5],[csiro.isel(member=6),csiro.isel(member=6)],linewidth=2,color='deeppink',alpha=0.3)
plt.plot([39.5,40.5],[csiro.isel(member=7),csiro.isel(member=7)],linewidth=2,color='deeppink',alpha=0.3)
plt.plot([39.5,40.5],[csiro.isel(member=8),csiro.isel(member=8)],linewidth=2,color='deeppink',alpha=0.3)
plt.plot([39.5,40.5],[csiro.isel(member=9),csiro.isel(member=9)],linewidth=2,color='deeppink',alpha=0.3)
f = np.mean(csiro)
plt.plot([40,40],[f,f],"*",markersize=9,color='w')
plt.plot([40,40],[f,f],"*",markersize=7,color='deeppink')

rng = [52]
ipsl5am = dsDelta5_1c.isel(member=rng)
plt.plot([41.5,42.5],[ipsl5am.isel(member=0),ipsl5am.isel(member=0)],linewidth=2,color='royalblue')

rng = [54]
miroc = dsDelta5_1c.isel(member=rng)
plt.plot([43.5,44.5],[miroc.isel(member=0),miroc.isel(member=0)],linewidth=2,color='lightsalmon')

rng = [62]
mri1 = dsDelta5_1c.isel(member=rng)
plt.plot([45.5,46.5],[mri1.isel(member=0),mri1.isel(member=0)],linewidth=2,color='palevioletred')

rng = np.arange(48,52)
ipsl5a = dsDelta5_1c.isel(member=rng)
plt.plot([47.5,48.5],[ipsl5a.isel(member=0),ipsl5a.isel(member=0)],linewidth=2,color='royalblue',alpha=0.3)
plt.plot([47.5,48.5],[ipsl5a.isel(member=1),ipsl5a.isel(member=1)],linewidth=2,color='royalblue',alpha=0.3)
plt.plot([47.5,48.5],[ipsl5a.isel(member=2),ipsl5a.isel(member=2)],linewidth=2,color='royalblue',alpha=0.3)
plt.plot([47.5,48.5],[ipsl5a.isel(member=3),ipsl5a.isel(member=3)],linewidth=2,color='royalblue',alpha=0.3)
f = np.mean(ipsl5a)
plt.plot([48,48],[f,f],"*",markersize=9,color='w')
plt.plot([48,48],[f,f],"*",markersize=7,color='royalblue')

rng = [53]
ipsl5b = dsDelta5_1c.isel(member=rng)
plt.plot([49.5,50.5],[ipsl5b.isel(member=0),ipsl5b.isel(member=0)],linewidth=2,color='royalblue')

plt.xlim([-1,51])
plt.ylim([0.5,2])
xticks = np.arange(0,51,2)
ax.set_xticks(xticks)
labels = ['ACCESS1-0','CESM1-CAM5','GFDL-CM3','HadGEM2-ES','MPI-ESM-LR','MPI-ESM-MR',
'ACCESS1-3','GFDL-ESM2M','NorESM1-ME','MIROC5','NorESM1-M','GFDL-ESM2G',"inmcm4",
'CanESM2',"CNRM-CM5","GISS-E2-R",'CCSM4',"GISS-E2-H","bcc-csm1-1",'bcc-csm1-1-m',
"CSIRO-Mk3-6-0","IPSL-CM5A-MR","MIROC-ESM","MRI-CGCM3","IPSL-CM5A-LR","IPSL-CM5B-LR"]
ax.set_xticklabels(labels,fontsize=8,rotation = 90)
plt.subplots_adjust(wspace=0.05, hspace=0)
ax.set_yticklabels('')
plt.title('d) CMIP5 Northern European Winter Case' ,fontsize=14,fontweight='bold',loc='left')

plt.savefig('SupFig10_perf_order_CMIP56_DJF_NEU.png',bbox_inches='tight',dpi=300)
