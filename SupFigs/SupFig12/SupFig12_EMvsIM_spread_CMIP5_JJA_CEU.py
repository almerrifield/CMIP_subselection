# Supplementary Figure 12 : Spread comparison for the CMIP5 JJA CEU case

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

## targets

dir = '/**/CMIP_subselection/Data/'

dsT5_jja = xr.open_dataset(dir + 'tas_mon_CMIP5_rcp85_g025_v2_CEU_JJA.nc',use_cftime = True)
dsT5_jja = dsT5_jja-273.15
dsT5_jja = dsT5_jja.sortby(dsT5_jja.member)
dsT5_jja = dsT5_jja.sel(member=members_cmip5)

dsPr5_ceu_jja = xr.open_dataset(dir + 'pr_mon_CMIP5_rcp85_g025_v2_CEU_JJA.nc',use_cftime = True)
dsPr5_ceu_jja['pr'] = dsPr5_ceu_jja.pr*86400
dsPr5_ceu_jja = dsPr5_ceu_jja.sortby(dsPr5_ceu_jja.member)
dsPr5_ceu_jja = dsPr5_ceu_jja.sel(member=members_cmip5)

dsT5_jja_base = dsT5_jja.sel(year=slice('1995','2014')).mean('year')
dsT5_jja_fut = dsT5_jja.sel(year=slice('2041','2060')).mean('year')

dsT5_target = dsT5_jja_fut - dsT5_jja_base

def cos_lat_weighted_mean(ds):
 weights = np.cos(np.deg2rad(ds.lat))
 weights.name = "weights"
 ds_weighted = ds.weighted(weights)
 weighted_mean = ds_weighted.mean(('lon', 'lat'))
 return weighted_mean

dsT5_target_ts = cos_lat_weighted_mean(dsT5_target)

dsPr5_ceu_jja_base = dsPr5_ceu_jja.sel(year=slice('1995','2014')).mean('year')
dsPr5_ceu_jja_fut = dsPr5_ceu_jja.sel(year=slice('2041','2060')).mean('year')

dsPr5_target = dsPr5_ceu_jja_fut - dsPr5_ceu_jja_base
dsPr5_target_ts = cos_lat_weighted_mean(dsPr5_target)

## cmip5 ensemble averaging
rng = np.arange(8,11)
cesm1 = dsT5_target_ts.isel(member=rng).mean('member').tas
cesm1['member'] = 'CESM1-CAM5-r0i0p0'
rng = np.arange(55,58)
miroc5 = dsT5_target_ts.isel(member=rng).mean('member').tas
miroc5['member'] = 'MIROC5-r0i0p0'
rng = np.arange(44,48)
hadgemes = dsT5_target_ts.isel(member=rng).mean('member').tas
hadgemes['member'] = 'HadGEM2-ES-r0i0p0'
rng = np.arange(2,8)
ccsm4 = dsT5_target_ts.isel(member=rng).mean('member').tas
ccsm4['member'] = 'CCSM4-r0i0p0'
rng = np.arange(26,31)
canesm2 = dsT5_target_ts.isel(member=rng).mean('member').tas
canesm2['member'] = 'CanESM2-r0i0p0'
rng = np.arange(58,61)
mpilr = dsT5_target_ts.isel(member=rng).mean('member').tas
mpilr['member'] = 'MPI-ESM-LR-r0i0p0'
rng = np.arange(16,26)
csiro = dsT5_target_ts.isel(member=rng).mean('member').tas
csiro['member'] = 'CSIRO-Mk3-6-0-r0i0p0'
rng = np.arange(39,44)
gissr = dsT5_target_ts.isel(member=rng).mean('member').tas
gissr['member'] = 'GISS-E2-R-r0i0p0'
rng = np.arange(34,39)
gissh = dsT5_target_ts.isel(member=rng).mean('member').tas
gissh['member'] = 'GISS-E2-H-r0i0p0'
rng = np.arange(11,16)
cnrm5 = dsT5_target_ts.isel(member=rng).mean('member').tas
cnrm5['member'] = 'CNRM-CM5-r0i0p0'
rng = np.arange(48,52)
ipsl5a = dsT5_target_ts.isel(member=rng).mean('member').tas
ipsl5a['member'] = 'IPSL-CM5A-LR-r0i0p0'

dsT5_target_ts_em = xr.concat([dsT5_target_ts.sel(member='ACCESS1-0-r1i1p1').tas,
dsT5_target_ts.sel(member='ACCESS1-3-r1i1p1').tas,ccsm4,cesm1,cnrm5,csiro,canesm2,
dsT5_target_ts.sel(member='GFDL-CM3-r1i1p1').tas,dsT5_target_ts.sel(member='GFDL-ESM2G-r1i1p1').tas,
dsT5_target_ts.sel(member='GFDL-ESM2M-r1i1p1').tas,gissh,gissr,hadgemes,ipsl5a,
dsT5_target_ts.sel(member='IPSL-CM5A-MR-r1i1p1').tas,dsT5_target_ts.sel(member='IPSL-CM5B-LR-r1i1p1').tas,
dsT5_target_ts.sel(member='MIROC-ESM-r1i1p1').tas,miroc5,mpilr,dsT5_target_ts.sel(member='MPI-ESM-MR-r1i1p1').tas,
dsT5_target_ts.sel(member='MRI-CGCM3-r1i1p1').tas,dsT5_target_ts.sel(member='NorESM1-M-r1i1p1').tas,
dsT5_target_ts.sel(member='NorESM1-ME-r1i1p1').tas,dsT5_target_ts.sel(member='bcc-csm1-1-m-r1i1p1').tas,
dsT5_target_ts.sel(member='bcc-csm1-1-r1i1p1').tas,dsT5_target_ts.sel(member='inmcm4-r1i1p1').tas],dim='member')

rng = np.arange(8,11)
cesm1 = dsPr5_target_ts.isel(member=rng).mean('member').pr
cesm1['member'] = 'CESM1-CAM5-r0i0p0'
rng = np.arange(55,58)
miroc5 = dsPr5_target_ts.isel(member=rng).mean('member').pr
miroc5['member'] = 'MIROC5-r0i0p0'
rng = np.arange(44,48)
hadgemes = dsPr5_target_ts.isel(member=rng).mean('member').pr
hadgemes['member'] = 'HadGEM2-ES-r0i0p0'
rng = np.arange(2,8)
ccsm4 = dsPr5_target_ts.isel(member=rng).mean('member').pr
ccsm4['member'] = 'CCSM4-r0i0p0'
rng = np.arange(26,31)
canesm2 = dsPr5_target_ts.isel(member=rng).mean('member').pr
canesm2['member'] = 'CanESM2-r0i0p0'
rng = np.arange(58,61)
mpilr = dsPr5_target_ts.isel(member=rng).mean('member').pr
mpilr['member'] = 'MPI-ESM-LR-r0i0p0'
rng = np.arange(16,26)
csiro = dsPr5_target_ts.isel(member=rng).mean('member').pr
csiro['member'] = 'CSIRO-Mk3-6-0-r0i0p0'
rng = np.arange(39,44)
gissr = dsPr5_target_ts.isel(member=rng).mean('member').pr
gissr['member'] = 'GISS-E2-R-r0i0p0'
rng = np.arange(34,39)
gissh = dsPr5_target_ts.isel(member=rng).mean('member').pr
gissh['member'] = 'GISS-E2-H-r0i0p0'
rng = np.arange(11,16)
cnrm5 = dsPr5_target_ts.isel(member=rng).mean('member').pr
cnrm5['member'] = 'CNRM-CM5-r0i0p0'
rng = np.arange(48,52)
ipsl5a = dsPr5_target_ts.isel(member=rng).mean('member').pr
ipsl5a['member'] = 'IPSL-CM5A-LR-r0i0p0'

dsPr5_target_ts_em = xr.concat([dsPr5_target_ts.sel(member='ACCESS1-0-r1i1p1').pr,
dsPr5_target_ts.sel(member='ACCESS1-3-r1i1p1').pr,ccsm4,cesm1,cnrm5,csiro,canesm2,
dsPr5_target_ts.sel(member='GFDL-CM3-r1i1p1').pr,dsPr5_target_ts.sel(member='GFDL-ESM2G-r1i1p1').pr,
dsPr5_target_ts.sel(member='GFDL-ESM2M-r1i1p1').pr,gissh,gissr,hadgemes,ipsl5a,
dsPr5_target_ts.sel(member='IPSL-CM5A-MR-r1i1p1').pr,dsPr5_target_ts.sel(member='IPSL-CM5B-LR-r1i1p1').pr,
dsPr5_target_ts.sel(member='MIROC-ESM-r1i1p1').pr,miroc5,mpilr,dsPr5_target_ts.sel(member='MPI-ESM-MR-r1i1p1').pr,
dsPr5_target_ts.sel(member='MRI-CGCM3-r1i1p1').pr,dsPr5_target_ts.sel(member='NorESM1-M-r1i1p1').pr,
dsPr5_target_ts.sel(member='NorESM1-ME-r1i1p1').pr,dsPr5_target_ts.sel(member='bcc-csm1-1-m-r1i1p1').pr,
dsPr5_target_ts.sel(member='bcc-csm1-1-r1i1p1').pr,dsPr5_target_ts.sel(member='inmcm4-r1i1p1').pr],dim='member')

###############################################################

#### ** Normalize ** #####

dsT5_target_ts_norm = (dsT5_target_ts  - np.mean(dsT5_target_ts))/np.std(dsT5_target_ts)
dsPr5_target_ts_norm = (dsPr5_target_ts - np.mean(dsPr5_target_ts))/np.std(dsPr5_target_ts)

#spread ensemble
dsT5_target_ts_spd_ind = dsT5_target_ts_norm.sel(member=[
       'ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1','GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1',
       'GFDL-ESM2M-r1i1p1','IPSL-CM5A-MR-r1i1p1', 'IPSL-CM5B-LR-r1i1p1',
       'MIROC-ESM-r1i1p1','MPI-ESM-MR-r1i1p1', 'MRI-CGCM3-r1i1p1', 'NorESM1-M-r1i1p1',
       'NorESM1-ME-r1i1p1', 'bcc-csm1-1-m-r1i1p1', 'bcc-csm1-1-r1i1p1',
       'inmcm4-r1i1p1']).tas
dsPr5_target_ts_spd_ind = dsPr5_target_ts_norm.sel(member=['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1',
        'GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1','GFDL-ESM2M-r1i1p1','IPSL-CM5A-MR-r1i1p1',
        'IPSL-CM5B-LR-r1i1p1','MIROC-ESM-r1i1p1','MPI-ESM-MR-r1i1p1', 'MRI-CGCM3-r1i1p1',
        'NorESM1-M-r1i1p1','NorESM1-ME-r1i1p1', 'bcc-csm1-1-m-r1i1p1', 'bcc-csm1-1-r1i1p1',
       'inmcm4-r1i1p1']).pr

#####################
# fixed ponts
keys = ['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1','GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1',
'GFDL-ESM2M-r1i1p1','IPSL-CM5A-MR-r1i1p1', 'IPSL-CM5B-LR-r1i1p1',
'MIROC-ESM-r1i1p1','MPI-ESM-MR-r1i1p1', 'MRI-CGCM3-r1i1p1', 'NorESM1-M-r1i1p1',
'NorESM1-ME-r1i1p1', 'bcc-csm1-1-m-r1i1p1', 'bcc-csm1-1-r1i1p1',
'inmcm4-r1i1p1']

dict_ind = {}
for ii in range(len(keys)):
    dict_ind[keys[ii]] = (dsT5_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr5_target_ts_norm.sel(member=keys[ii]).pr.item(0))

#########
# Model 1
keys = ['CCSM4-r1i1p1', 'CCSM4-r2i1p1',
'CCSM4-r3i1p1', 'CCSM4-r4i1p1', 'CCSM4-r5i1p1', 'CCSM4-r6i1p1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT5_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr5_target_ts_norm.sel(member=keys[ii]).pr.item(0))

## pick member
def choose_furthest_member(mem):
    min_dist = {}
    dict_dist = {}
    for key in dict_ind:
        dist = np.linalg.norm(np.array(dict_model[mem]) - np.array(dict_ind[key]))
        dict_dist[key] = dist
    min_key = min(dict_dist, key=dict_dist.get)
    min_value = min(dict_dist.values())
    print("Lowest value:",mem,min_key,min_value)

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'CCSM4-r6i1p1'
dict_ind[key_choice] = (dsT5_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr5_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 2
keys = ['CESM1-CAM5-r1i1p1', 'CESM1-CAM5-r2i1p1', 'CESM1-CAM5-r3i1p1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT5_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr5_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'CESM1-CAM5-r1i1p1'
dict_ind[key_choice] = (dsT5_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr5_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 3
keys = ['CNRM-CM5-r10i1p1', 'CNRM-CM5-r1i1p1', 'CNRM-CM5-r2i1p1','CNRM-CM5-r4i1p1', 'CNRM-CM5-r6i1p1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT5_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr5_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'CNRM-CM5-r2i1p1'
dict_ind[key_choice] = (dsT5_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr5_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 4
keys = ['CSIRO-Mk3-6-0-r10i1p1',
'CSIRO-Mk3-6-0-r1i1p1', 'CSIRO-Mk3-6-0-r2i1p1', 'CSIRO-Mk3-6-0-r3i1p1',
'CSIRO-Mk3-6-0-r4i1p1', 'CSIRO-Mk3-6-0-r5i1p1', 'CSIRO-Mk3-6-0-r6i1p1',
'CSIRO-Mk3-6-0-r7i1p1', 'CSIRO-Mk3-6-0-r8i1p1', 'CSIRO-Mk3-6-0-r9i1p1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT5_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr5_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'CSIRO-Mk3-6-0-r10i1p1'
dict_ind[key_choice] = (dsT5_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr5_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 5
keys = ['CanESM2-r1i1p1', 'CanESM2-r2i1p1', 'CanESM2-r3i1p1', 'CanESM2-r4i1p1',
'CanESM2-r5i1p1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT5_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr5_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'CanESM2-r5i1p1'
dict_ind[key_choice] = (dsT5_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr5_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 6
keys = ['GISS-E2-H-r1i1p1', 'GISS-E2-H-r1i1p2','GISS-E2-H-r1i1p3', 'GISS-E2-H-r2i1p1', 'GISS-E2-H-r2i1p3']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT5_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr5_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'GISS-E2-H-r1i1p3'
dict_ind[key_choice] = (dsT5_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr5_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 7
keys = ['GISS-E2-R-r1i1p1', 'GISS-E2-R-r1i1p2', 'GISS-E2-R-r1i1p3','GISS-E2-R-r2i1p1', 'GISS-E2-R-r2i1p3']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT5_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr5_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'GISS-E2-R-r1i1p3'
dict_ind[key_choice] = (dsT5_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr5_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 8
keys = ['HadGEM2-ES-r1i1p1','HadGEM2-ES-r2i1p1', 'HadGEM2-ES-r3i1p1', 'HadGEM2-ES-r4i1p1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT5_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr5_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'HadGEM2-ES-r4i1p1'
dict_ind[key_choice] = (dsT5_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr5_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 9
keys = ['IPSL-CM5A-LR-r1i1p1', 'IPSL-CM5A-LR-r2i1p1', 'IPSL-CM5A-LR-r3i1p1','IPSL-CM5A-LR-r4i1p1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT5_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr5_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'IPSL-CM5A-LR-r2i1p1'
dict_ind[key_choice] = (dsT5_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr5_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 10
keys = ['MIROC5-r1i1p1', 'MIROC5-r2i1p1', 'MIROC5-r3i1p1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT5_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr5_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'MIROC5-r3i1p1'
dict_ind[key_choice] = (dsT5_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr5_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 11
keys = ['MPI-ESM-LR-r1i1p1', 'MPI-ESM-LR-r2i1p1', 'MPI-ESM-LR-r3i1p1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT5_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr5_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'MPI-ESM-LR-r2i1p1'
dict_ind[key_choice] = (dsT5_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr5_target_ts_norm.sel(member=key_choice).pr.item(0))

###########################################################

dsT5_target_ts_spd_all = dsT5_target_ts.sel(member=['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1', 'CCSM4-r6i1p1',
       'CESM1-CAM5-r1i1p1','CNRM-CM5-r2i1p1', 'CSIRO-Mk3-6-0-r10i1p1',
       'CanESM2-r5i1p1', 'GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1',
       'GFDL-ESM2M-r1i1p1', 'GISS-E2-H-r1i1p3','GISS-E2-R-r1i1p3','HadGEM2-ES-r4i1p1',
       'IPSL-CM5A-LR-r1i1p1','IPSL-CM5A-MR-r1i1p1', 'IPSL-CM5B-LR-r1i1p1',
       'MIROC-ESM-r1i1p1', 'MIROC5-r3i1p1','MPI-ESM-LR-r2i1p1',
       'MPI-ESM-MR-r1i1p1', 'MRI-CGCM3-r1i1p1', 'NorESM1-M-r1i1p1',
       'NorESM1-ME-r1i1p1', 'bcc-csm1-1-m-r1i1p1', 'bcc-csm1-1-r1i1p1',
       'inmcm4-r1i1p1']).tas

dsPr5_target_ts_spd_all = dsPr5_target_ts.sel(member=['ACCESS1-0-r1i1p1', 'ACCESS1-3-r1i1p1', 'CCSM4-r6i1p1',
       'CESM1-CAM5-r1i1p1','CNRM-CM5-r2i1p1', 'CSIRO-Mk3-6-0-r10i1p1',
       'CanESM2-r5i1p1', 'GFDL-CM3-r1i1p1', 'GFDL-ESM2G-r1i1p1',
       'GFDL-ESM2M-r1i1p1', 'GISS-E2-H-r1i1p3','GISS-E2-R-r1i1p3','HadGEM2-ES-r4i1p1',
       'IPSL-CM5A-LR-r1i1p1','IPSL-CM5A-MR-r1i1p1', 'IPSL-CM5B-LR-r1i1p1',
       'MIROC-ESM-r1i1p1', 'MIROC5-r3i1p1','MPI-ESM-LR-r2i1p1',
       'MPI-ESM-MR-r1i1p1', 'MRI-CGCM3-r1i1p1', 'NorESM1-M-r1i1p1',
       'NorESM1-ME-r1i1p1', 'bcc-csm1-1-m-r1i1p1', 'bcc-csm1-1-r1i1p1',
       'inmcm4-r1i1p1']).pr

# ################################################
fig = plt.figure(figsize=(14,6))
ax = plt.subplot(121)
# ################################################

plt.axvline(dsT5_target_ts_em.median("member").data,color='k',linewidth=1,linestyle='dashed',alpha=0.2)
plt.axhline(dsPr5_target_ts_em.median("member").data,color='k',linewidth=1,linestyle='dashed',alpha=0.2)

plt.scatter(dsT5_target_ts.tas,dsPr5_target_ts.pr,s=5,c="silver",marker='.',alpha=0.5)
plt.scatter(dsT5_target_ts_em.isel(member=0),dsPr5_target_ts_em.isel(member=0),c='tab:red',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=1),dsPr5_target_ts_em.isel(member=1),c='tab:red',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=2),dsPr5_target_ts_em.isel(member=2),c='darkgoldenrod',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=3),dsPr5_target_ts_em.isel(member=3),c='darkgoldenrod',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=4),dsPr5_target_ts_em.isel(member=4),c='cornflowerblue',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=5),dsPr5_target_ts_em.isel(member=5),c='deeppink',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=6),dsPr5_target_ts_em.isel(member=6),c='dodgerblue',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=7),dsPr5_target_ts_em.isel(member=7),c='indigo',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=8),dsPr5_target_ts_em.isel(member=8),c='indigo',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=9),dsPr5_target_ts_em.isel(member=9),c='indigo',s=20,marker='^',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=10),dsPr5_target_ts_em.isel(member=10),c='blueviolet',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=11),dsPr5_target_ts_em.isel(member=11),c='blueviolet',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=12),dsPr5_target_ts_em.isel(member=12),c='tab:red',s=20,marker='^',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=13),dsPr5_target_ts_em.isel(member=13),c='royalblue',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=14),dsPr5_target_ts_em.isel(member=14),c='royalblue',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=15),dsPr5_target_ts_em.isel(member=15),c='royalblue',s=20,marker='^',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=16),dsPr5_target_ts_em.isel(member=16),c='lightsalmon',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=17),dsPr5_target_ts_em.isel(member=17),c='lightsalmon',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=18),dsPr5_target_ts_em.isel(member=18),c='tab:orange',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=19),dsPr5_target_ts_em.isel(member=19),c='tab:orange',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=20),dsPr5_target_ts_em.isel(member=20),c='palevioletred',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=21),dsPr5_target_ts_em.isel(member=21),c='darkgoldenrod',s=20,marker='^',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=22),dsPr5_target_ts_em.isel(member=22),c='darkgoldenrod',s=30,marker='*',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=23),dsPr5_target_ts_em.isel(member=23),c='silver',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=24),dsPr5_target_ts_em.isel(member=24),c='silver',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_em.isel(member=25),dsPr5_target_ts_em.isel(member=25),c='mediumseagreen',s=20,marker='x',alpha=1)

plt.xlabel('JJA CEU SAT Change (˚C; 2041/2060 - 1995/2014)')
plt.xlim([0.5,6])
plt.ylabel('JJA CEU PR Change (mm/day; 2041/2060 - 1995/2014)')
plt.ylim([-0.7,0.5])
plt.title('a) SAT-PR Change, Ensemble Means',fontsize=14,fontweight='bold',loc='left')

ax = plt.subplot(122)

plt.axvline(dsT5_target_ts_spd_all.median("member").data,color='k',linewidth=1,linestyle='dashed',alpha=0.2)
plt.axhline(dsPr5_target_ts_spd_all.median("member").data,color='k',linewidth=1,linestyle='dashed',alpha=0.2)

plt.scatter(dsT5_target_ts.tas,dsPr5_target_ts.pr,s=5,c="silver",marker='.',alpha=0.5)
plt.scatter(dsT5_target_ts_spd_all.isel(member=0),dsPr5_target_ts_spd_all.isel(member=0),c='tab:red',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=1),dsPr5_target_ts_spd_all.isel(member=1),c='tab:red',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=2),dsPr5_target_ts_spd_all.isel(member=2),c='darkgoldenrod',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=3),dsPr5_target_ts_spd_all.isel(member=3),c='darkgoldenrod',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=4),dsPr5_target_ts_spd_all.isel(member=4),c='cornflowerblue',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=5),dsPr5_target_ts_spd_all.isel(member=5),c='deeppink',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=6),dsPr5_target_ts_spd_all.isel(member=6),c='dodgerblue',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=7),dsPr5_target_ts_spd_all.isel(member=7),c='indigo',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=8),dsPr5_target_ts_spd_all.isel(member=8),c='indigo',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=9),dsPr5_target_ts_spd_all.isel(member=9),c='indigo',s=20,marker='^',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=10),dsPr5_target_ts_spd_all.isel(member=10),c='blueviolet',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=11),dsPr5_target_ts_spd_all.isel(member=11),c='blueviolet',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=12),dsPr5_target_ts_spd_all.isel(member=12),c='tab:red',s=20,marker='^',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=13),dsPr5_target_ts_spd_all.isel(member=13),c='royalblue',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=14),dsPr5_target_ts_spd_all.isel(member=14),c='royalblue',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=15),dsPr5_target_ts_spd_all.isel(member=15),c='royalblue',s=20,marker='^',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=16),dsPr5_target_ts_spd_all.isel(member=16),c='lightsalmon',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=17),dsPr5_target_ts_spd_all.isel(member=17),c='lightsalmon',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=18),dsPr5_target_ts_spd_all.isel(member=18),c='tab:orange',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=19),dsPr5_target_ts_spd_all.isel(member=19),c='tab:orange',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=20),dsPr5_target_ts_spd_all.isel(member=20),c='palevioletred',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=21),dsPr5_target_ts_spd_all.isel(member=21),c='darkgoldenrod',s=20,marker='^',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=22),dsPr5_target_ts_spd_all.isel(member=22),c='darkgoldenrod',s=30,marker='*',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=23),dsPr5_target_ts_spd_all.isel(member=23),c='silver',s=20,marker='x',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=24),dsPr5_target_ts_spd_all.isel(member=24),c='silver',s=20,marker='o',alpha=1)
plt.scatter(dsT5_target_ts_spd_all.isel(member=25),dsPr5_target_ts_spd_all.isel(member=25),c='mediumseagreen',s=20,marker='x',alpha=1)

plt.xlabel('JJA CEU SAT Change (˚C; 2041/2060 - 1995/2014)')
plt.xlim([0.5,6])
ax.set_yticklabels('')
plt.ylim([-0.7,0.5])
plt.title('b) SAT-PR Change, Individual Member',fontsize=14,fontweight='bold',loc='left')
plt.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig('SupFig12_EMvsIM_spread_CMIP5_JJA_CEU.png',bbox_inches='tight',dpi=300)
