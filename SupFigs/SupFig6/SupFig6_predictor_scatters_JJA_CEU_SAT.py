# Supplementary Figure 6: predictor scatters vs. JJA CEU SAT change

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

dsT6_jja = xr.open_dataset(dir + 'tas_mon_CMIP6_SSP585_g025_v2_CEU_JJA.nc',use_cftime = True)
dsT6_jja  = dsT6_jja-273.15
dsT6_jja  = dsT6_jja.sortby(dsT6_jja.member)
dsT6_jja = dsT6_jja.sel(member=members_cmip6)

dir = '/**/CMIP_subselection/Data/'

dsT5_jja = xr.open_dataset(dir + 'tas_mon_CMIP5_rcp85_g025_v2_CEU_JJA.nc',use_cftime = True)
dsT5_jja = dsT5_jja-273.15
dsT5_jja = dsT5_jja.sortby(dsT5_jja.member)
dsT5_jja = dsT5_jja.sel(member=members_cmip5)

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

###
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

dsT5_jja_base = dsT5_jja.sel(year=slice('1995','2014')).mean('year')
dsT6_jja_base = dsT6_jja.sel(year=slice('1995','2014')).mean('year')

dsT5_jja_fut = dsT5_jja.sel(year=slice('2041','2060')).mean('year')
dsT6_jja_fut = dsT6_jja.sel(year=slice('2041','2060')).mean('year')

dsT5_target = dsT5_jja_fut - dsT5_jja_base
dsT6_target = dsT6_jja_fut - dsT6_jja_base

def cos_lat_weighted_mean(ds):
  weights = np.cos(np.deg2rad(ds.lat))
  weights.name = "weights"
  ds_weighted = ds.weighted(weights)
  weighted_mean = ds_weighted.mean(('lon', 'lat'))
  return weighted_mean

dsT5_target_ts = cos_lat_weighted_mean(dsT5_target)
dsT6_target_ts = cos_lat_weighted_mean(dsT6_target)

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

# Figure
fig = plt.figure(figsize=(10,12)) #figsize=(16,8)
fig.suptitle('Predictor RMSE from Observed vs. JJA Central European SAT Change \n(2041/2060 - 1995/2014)\n', fontsize=14,fontweight='bold')
ax1 = plt.subplot(3,3,1)
plt.plot(dsT5_ann_his_delta,dsT5_target_ts.tas,'x',label='CMIP5',markersize=4)
plt.plot(dsT6_ann_his_delta,dsT6_target_ts.tas,'.',label='CMIP6',fillstyle='none')
plt.xlim([0,4])
plt.ylim([0.5,6])
plt.title('a) Annual European SAT \nClimatology (1950/1969)',fontsize=10)
plt.xlabel('RMSE (˚C)',fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.ylabel('JJA CEU SAT Change (˚C)',fontsize=11)

ax2 = plt.subplot(3,3,2)
plt.plot(dsT5_ann_base_delta,dsT5_target_ts.tas,'x',label='CMIP5',markersize=4)
plt.plot(dsT6_ann_base_delta,dsT6_target_ts.tas,'.',label='CMIP6',fillstyle='none')
plt.xlim([0,4])
plt.ylim([0.5,6])
plt.title('b) Annual European SAT \nClimatology (1995/2014)',fontsize=10)
plt.yticks(fontsize=11)
ax2.set_yticklabels('')
plt.xticks(fontsize=11)
plt.xlabel('RMSE (˚C)',fontsize=11)

ax3 = plt.subplot(3,3,3)
plt.plot(dsPr5_jja_base_delta,dsT5_target_ts.tas,'x',label='CMIP5',markersize=4)
plt.plot(dsPr6_jja_base_delta,dsT6_target_ts.tas,'.',label='CMIP6',fillstyle='none')
plt.xlim([0,2.2])
plt.ylim([0.5,6])
plt.title('c) JJA Central Europe Station PR \nClimatology (1995/2014)',fontsize=10)
plt.yticks(fontsize=11)
ax3.set_yticklabels('')
plt.xticks(fontsize=11)
plt.xlabel('RMSE (mm/day)',fontsize=11)


ax4 = plt.subplot(3,3,4)
plt.plot(dsSW5_jja_base_delta,dsT5_target_ts.tas,'x',label='CMIP5',markersize=4)
plt.plot(dsSW6_jja_base_delta,dsT6_target_ts.tas,'.',label='CMIP6',fillstyle='none')
plt.xlim([0,60])
plt.ylim([0.5,6])
plt.title('d) JJA Central European SWCRE \nClimatology (2001/2018)',fontsize=10)
plt.xlabel('RMSE (W/m^2)',fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.ylabel('JJA CEU SAT Change (˚C)',fontsize=12)


ax5 = plt.subplot(3,3,5)
plt.plot(dsSST5_ann_base_delta,dsT5_target_ts.tas,'x',label='CMIP5',markersize=4)
plt.plot(dsSST6_ann_base_delta,dsT6_target_ts.tas,'.',label='CMIP6',fillstyle='none')
plt.xlim([0,6])
plt.ylim([0.5,6])
plt.title('e) Annual North Atlantic SST \nClimatology (1995/2014)',fontsize=10)
plt.yticks(fontsize=11)
ax5.set_yticklabels('')
plt.xticks(fontsize=11)
plt.xlabel('RMSE (˚C)',fontsize=11)
#
ax6 = plt.subplot(3,3,6)
plt.plot(dsSW5_ann_base_delta,dsT5_target_ts.tas,'x',label='CMIP5',markersize=4)
plt.plot(dsSW6_ann_base_delta,dsT6_target_ts.tas,'.',label='CMIP6',fillstyle='none')
plt.xlim([0,32])
plt.ylim([0.5,6])
plt.title('f) Annual Southern Midlatitude SWCRE \nClimatology (2001/2018)',fontsize=10)
plt.yticks(fontsize=11)
ax6.set_yticklabels('')
plt.xticks(fontsize=11)
plt.xlabel('RMSE (W/m^2)',fontsize=11)


ax7 = plt.subplot(3,1,3)
plt.plot(dsDelta5_1c,dsT5_target_ts.tas,'x',label='CMIP5',markersize=4)
plt.plot(dsDelta6_1c,dsT6_target_ts.tas,'.',label='CMIP6',fillstyle='none')
plt.xlim([0.6,2.0])
plt.ylim([0.5,6])
plt.title('g) Average of normalized RMSEs; all predictors',fontsize=10)
plt.xlabel('Aggregated Distance From Observed',fontsize=11)
plt.ylabel('JJA CEU SAT Change (˚C)',fontsize=11)
plt.legend(fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.subplots_adjust(wspace=0.1, hspace=0.4)
plt.savefig('SupFig6_predictor_scatter_JJA_CEU_SAT.png',dpi=300)
plt.close()
