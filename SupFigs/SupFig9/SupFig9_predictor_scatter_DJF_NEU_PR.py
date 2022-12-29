# Supplementary Figure 9: predictor scatters vs. DJF NEU PR change

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

dsPr6_neu_djf = xr.open_dataset(dir + 'pr_mon_CMIP6_SSP585_g025_v2_NEU_DJF.nc',use_cftime = True)
dsPr6_neu_djf['pr']  = dsPr6_neu_djf.pr*86400
dsPr6_neu_djf  = dsPr6_neu_djf.sortby(dsPr6_neu_djf.member)
dsPr6_neu_djf  = dsPr6_neu_djf.sel(member=members_cmip6)

dir = '/**/CMIP_subselection/Data/'

dsPr5_neu_djf = xr.open_dataset(dir + 'pr_mon_CMIP5_rcp85_g025_v2_NEU_DJF.nc',use_cftime = True)
dsPr5_neu_djf['pr'] = dsPr5_neu_djf.pr*86400
dsPr5_neu_djf = dsPr5_neu_djf.sortby(dsPr5_neu_djf.member)
dsPr5_neu_djf = dsPr5_neu_djf.sel(member=members_cmip5)

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

dsPr5_neu_djf_base = dsPr5_neu_djf.sel(year=slice('1995','2014')).mean('year')
dsPr6_neu_djf_base = dsPr6_neu_djf.sel(year=slice('1995','2014')).mean('year')

dsPr5_neu_djf_fut = dsPr5_neu_djf.sel(year=slice('2041','2060')).mean('year')
dsPr6_neu_djf_fut = dsPr6_neu_djf.sel(year=slice('2041','2060')).mean('year')

dsPr5_target = dsPr5_neu_djf_fut - dsPr5_neu_djf_base
dsPr6_target = dsPr6_neu_djf_fut - dsPr6_neu_djf_base

def cos_lat_weighted_mean(ds):
  weights = np.cos(np.deg2rad(ds.lat))
  weights.name = "weights"
  ds_weighted = ds.weighted(weights)
  weighted_mean = ds_weighted.mean(('lon', 'lat'))
  return weighted_mean

dsPr5_target_ts = cos_lat_weighted_mean(dsPr5_target)
dsPr6_target_ts = cos_lat_weighted_mean(dsPr6_target)

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


# Figure (to be stitched together)
fig = plt.figure(figsize=(10,12)) #figsize=(16,8)
fig.suptitle('Predictor RMSE from Observed vs. DJF Northern European PR Change \n(2041/2060 - 1995/2014)\n', fontsize=14,fontweight='bold')
ax1 = plt.subplot(3,3,1)
plt.plot(dsT5_ann_his_delta,dsPr5_target_ts.pr,'x',label='CMIP5',markersize=4)
plt.plot(dsT6_ann_his_delta,dsPr6_target_ts.pr,'.',label='CMIP6',fillstyle='none')
plt.xlim([0,4])
plt.ylim([-0.3,0.8])
plt.title('a) Annual European SAT \nClimatology (1950/1969)',fontsize=10)
plt.xlabel('RMSE (˚C)',fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.ylabel('DJF NEU PR Change (mm/day)',fontsize=11)

ax2 = plt.subplot(3,3,2)
plt.plot(dsT5_ann_base_delta,dsPr5_target_ts.pr,'x',label='CMIP5',markersize=4)
plt.plot(dsT6_ann_base_delta,dsPr6_target_ts.pr,'.',label='CMIP6',fillstyle='none')
plt.xlim([0,4])
plt.ylim([-0.3,0.8])
plt.title('b) Annual European SAT \nClimatology (1995/2014)',fontsize=10)
plt.yticks(fontsize=11)
ax2.set_yticklabels('')
plt.xticks(fontsize=11)
plt.xlabel('RMSE (˚C)',fontsize=11)

ax3 = plt.subplot(3,3,3)
plt.plot(dsPr5_djf_base_delta,dsPr5_target_ts.pr,'x',label='CMIP5',markersize=4)
plt.plot(dsPr6_djf_base_delta,dsPr6_target_ts.pr,'.',label='CMIP6',fillstyle='none')
plt.xlim([0,2])
plt.ylim([-0.3,0.8])
plt.title('c) DJF Northern Europe Station PR \nClimatology (1995/2014)',fontsize=10)
plt.yticks(fontsize=11)
ax3.set_yticklabels('')
plt.xticks(fontsize=11)
plt.xlabel('RMSE (mm/day)',fontsize=11)


ax4 = plt.subplot(3,3,4)
plt.plot(dsP5_djf_base_delta,dsPr5_target_ts.pr,'x',label='CMIP5',markersize=4)
plt.plot(dsP6_djf_base_delta,dsPr6_target_ts.pr,'.',label='CMIP6',fillstyle='none')
plt.xlim([0,10])
plt.ylim([-0.3,0.8])
plt.title('d) DJF North Atlantic Sector SLP \nClimatology (1950/2014)',fontsize=10)
plt.xlabel('RMSE (hPa)',fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.ylabel('DJF NEU PR Change (mm/day)',fontsize=12)


ax5 = plt.subplot(3,3,5)
plt.plot(dsSST5_ann_base_delta,dsPr5_target_ts.pr,'x',label='CMIP5',markersize=4)
plt.plot(dsSST6_ann_base_delta,dsPr6_target_ts.pr,'.',label='CMIP6',fillstyle='none')
plt.xlim([0,6])
plt.ylim([-0.3,0.8])
plt.title('e) Annual North Atlantic SST \nClimatology (1995/2014)',fontsize=10)
plt.yticks(fontsize=11)
ax5.set_yticklabels('')
plt.xticks(fontsize=11)
plt.xlabel('RMSE (˚C)',fontsize=11)
#
ax6 = plt.subplot(3,3,6)
plt.plot(dsSW5_ann_base_delta,dsPr5_target_ts.pr,'x',label='CMIP5',markersize=4)
plt.plot(dsSW6_ann_base_delta,dsPr6_target_ts.pr,'.',label='CMIP6',fillstyle='none')
plt.xlim([0,32])
plt.ylim([-0.3,0.8])
plt.title('f) Annual Southern Midlatitude SWCRE \nClimatology (2001/2018)',fontsize=10)
plt.yticks(fontsize=11)
ax6.set_yticklabels('')
plt.xticks(fontsize=11)
plt.xlabel('RMSE (W/m^2)',fontsize=11)

ax7 = plt.subplot(3,1,3)
plt.plot(dsDelta5_1c,dsPr5_target_ts.pr,'x',label='CMIP5',markersize=4)
plt.plot(dsDelta6_1c,dsPr6_target_ts.pr,'.',label='CMIP6',fillstyle='none')
plt.xlim([0.5,2])
plt.ylim([-0.3,0.8])
plt.title('g) Average of normalized RMSEs; all predictors',fontsize=10)
plt.xlabel('Aggregated Distance From Observed',fontsize=11)
plt.ylabel('DEU NEU PR Change (mm/day)',fontsize=11)
plt.legend(fontsize=11,loc=4)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.subplots_adjust(wspace=0.1, hspace=0.4)

plt.savefig('SupFig9_predictor_scatter_DJF_NEU_PR.png',dpi=300)
plt.close()
