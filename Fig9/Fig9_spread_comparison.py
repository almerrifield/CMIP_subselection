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


## targets

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

def cos_lat_weighted_mean(ds):
 weights = np.cos(np.deg2rad(ds.lat))
 weights.name = "weights"
 ds_weighted = ds.weighted(weights)
 weighted_mean = ds_weighted.mean(('lon', 'lat'))
 return weighted_mean

dsT6_target_ts = cos_lat_weighted_mean(dsT6_target)

dsPr6_ceu_jja_base = dsPr6_ceu_jja.sel(year=slice('1995','2014')).mean('year')
dsPr6_ceu_jja_fut = dsPr6_ceu_jja.sel(year=slice('2041','2060')).mean('year')

dsPr6_target = dsPr6_ceu_jja_fut - dsPr6_ceu_jja_base
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

##########################################################

#### ** Normalize ** #####

dsT6_target_ts_norm = (dsT6_target_ts  - np.mean(dsT6_target_ts))/np.std(dsT6_target_ts)
dsPr6_target_ts_norm = (dsPr6_target_ts - np.mean(dsPr6_target_ts))/np.std(dsPr6_target_ts)

# spread ensemble
dsT6_target_ts_spd_ind = dsT6_target_ts_norm.sel(member=[
       'AWI-CM-1-1-MR-r1i1p1f1', 'CMCC-CM2-SR5-r1i1p1f1',
       'CMCC-ESM2-r1i1p1f1', 'CNRM-CM6-1-HR-r1i1p1f2',
       'E3SM-1-1-r1i1p1f1', 'FGOALS-f3-L-r1i1p1f1', 'GFDL-CM4-r1i1p1f1', 'GFDL-ESM4-r1i1p1f1',
       'GISS-E2-1-G-r1i1p3f1', 'INM-CM4-8-r1i1p1f1', 'INM-CM5-0-r1i1p1f1','KIOST-ESM-r1i1p1f1',
       'NorESM2-MM-r1i1p1f1','TaiESM1-r1i1p1f1']).tas
dsPr6_target_ts_spd_ind = dsPr6_target_ts_norm.sel(member=['AWI-CM-1-1-MR-r1i1p1f1', 'CMCC-CM2-SR5-r1i1p1f1',
    'CMCC-ESM2-r1i1p1f1', 'CNRM-CM6-1-HR-r1i1p1f2','E3SM-1-1-r1i1p1f1', 'FGOALS-f3-L-r1i1p1f1',
    'GFDL-CM4-r1i1p1f1', 'GFDL-ESM4-r1i1p1f1','GISS-E2-1-G-r1i1p3f1', 'INM-CM4-8-r1i1p1f1',
    'INM-CM5-0-r1i1p1f1','KIOST-ESM-r1i1p1f1', 'NorESM2-MM-r1i1p1f1','TaiESM1-r1i1p1f1']).pr

#####################
# fixed ponts
keys = ['AWI-CM-1-1-MR-r1i1p1f1', 'CMCC-CM2-SR5-r1i1p1f1',
'CMCC-ESM2-r1i1p1f1', 'CNRM-CM6-1-HR-r1i1p1f2',
'E3SM-1-1-r1i1p1f1', 'FGOALS-f3-L-r1i1p1f1', 'GFDL-CM4-r1i1p1f1', 'GFDL-ESM4-r1i1p1f1',
'GISS-E2-1-G-r1i1p3f1', 'INM-CM4-8-r1i1p1f1', 'INM-CM5-0-r1i1p1f1','KIOST-ESM-r1i1p1f1',
'NorESM2-MM-r1i1p1f1','TaiESM1-r1i1p1f1']

dict_ind = {}
for ii in range(len(keys)):
    dict_ind[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

#########
# Model 1
keys = ['ACCESS-CM2-r1i1p1f1', 'ACCESS-CM2-r2i1p1f1', 'ACCESS-CM2-r3i1p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

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

key_choice = 'ACCESS-CM2-r2i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 2
keys = ['ACCESS-ESM1-5-r10i1p1f1', 'ACCESS-ESM1-5-r1i1p1f1',
       'ACCESS-ESM1-5-r2i1p1f1', 'ACCESS-ESM1-5-r3i1p1f1',
       'ACCESS-ESM1-5-r4i1p1f1', 'ACCESS-ESM1-5-r5i1p1f1',
       'ACCESS-ESM1-5-r6i1p1f1', 'ACCESS-ESM1-5-r7i1p1f1',
       'ACCESS-ESM1-5-r8i1p1f1', 'ACCESS-ESM1-5-r9i1p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'ACCESS-ESM1-5-r1i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 3
keys = ['CAS-ESM2-0-r1i1p1f1', 'CAS-ESM2-0-r3i1p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'CAS-ESM2-0-r1i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 4
keys = ['CESM2-WACCM-r1i1p1f1', 'CESM2-WACCM-r2i1p1f1', 'CESM2-WACCM-r3i1p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'CESM2-WACCM-r2i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 5
keys = ['CESM2-r10i1p1f1', 'CESM2-r11i1p1f1', 'CESM2-r1i1p1f1','CESM2-r2i1p1f1', 'CESM2-r4i1p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'CESM2-r11i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 6

keys = ['CNRM-CM6-1-r1i1p1f2','CNRM-CM6-1-r2i1p1f2',
'CNRM-CM6-1-r3i1p1f2', 'CNRM-CM6-1-r4i1p1f2',
'CNRM-CM6-1-r5i1p1f2', 'CNRM-CM6-1-r6i1p1f2']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'CNRM-CM6-1-r4i1p1f2'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 7

keys = ['CNRM-ESM2-1-r1i1p1f2','CNRM-ESM2-1-r2i1p1f2', 'CNRM-ESM2-1-r3i1p1f2', 'CNRM-ESM2-1-r4i1p1f2','CNRM-ESM2-1-r5i1p1f2']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'CNRM-ESM2-1-r2i1p1f2'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 8

keys = ['CanESM5-r10i1p1f1', 'CanESM5-r10i1p2f1',
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
       'CanESM5-r8i1p2f1', 'CanESM5-r9i1p1f1', 'CanESM5-r9i1p2f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'CanESM5-r16i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 9

keys = ['FGOALS-g3-r1i1p1f1','FGOALS-g3-r2i1p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'FGOALS-g3-r2i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 10

keys = ['HadGEM3-GC31-LL-r1i1p1f3','HadGEM3-GC31-LL-r2i1p1f3', 'HadGEM3-GC31-LL-r3i1p1f3','HadGEM3-GC31-LL-r4i1p1f3']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'HadGEM3-GC31-LL-r3i1p1f3'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 11

keys = ['HadGEM3-GC31-MM-r1i1p1f3','HadGEM3-GC31-MM-r2i1p1f3', 'HadGEM3-GC31-MM-r3i1p1f3','HadGEM3-GC31-MM-r4i1p1f3']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'HadGEM3-GC31-MM-r1i1p1f3'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 12

keys = ['IPSL-CM6A-LR-r14i1p1f1', 'IPSL-CM6A-LR-r1i1p1f1',
        'IPSL-CM6A-LR-r2i1p1f1', 'IPSL-CM6A-LR-r3i1p1f1',
        'IPSL-CM6A-LR-r4i1p1f1', 'IPSL-CM6A-LR-r6i1p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'IPSL-CM6A-LR-r6i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 13

keys = ['KACE-1-0-G-r2i1p1f1','KACE-1-0-G-r3i1p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'KACE-1-0-G-r3i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 14

keys = ['MIROC-ES2L-r10i1p1f2','MIROC-ES2L-r1i1p1f2', 'MIROC-ES2L-r2i1p1f2', 'MIROC-ES2L-r3i1p1f2',
'MIROC-ES2L-r4i1p1f2', 'MIROC-ES2L-r5i1p1f2', 'MIROC-ES2L-r6i1p1f2','MIROC-ES2L-r7i1p1f2', 'MIROC-ES2L-r8i1p1f2', 'MIROC-ES2L-r9i1p1f2']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'MIROC-ES2L-r1i1p1f2'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 15

keys = ['MIROC6-r10i1p1f1', 'MIROC6-r11i1p1f1', 'MIROC6-r12i1p1f1',
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
       'MIROC6-r8i1p1f1', 'MIROC6-r9i1p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'MIROC6-r15i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 16
keys = ['MPI-ESM1-2-HR-r1i1p1f1',
        'MPI-ESM1-2-HR-r2i1p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'MPI-ESM1-2-HR-r1i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 17
keys = ['MPI-ESM1-2-LR-r10i1p1f1',
       'MPI-ESM1-2-LR-r1i1p1f1', 'MPI-ESM1-2-LR-r2i1p1f1',
       'MPI-ESM1-2-LR-r3i1p1f1', 'MPI-ESM1-2-LR-r4i1p1f1',
       'MPI-ESM1-2-LR-r5i1p1f1', 'MPI-ESM1-2-LR-r6i1p1f1',
       'MPI-ESM1-2-LR-r7i1p1f1', 'MPI-ESM1-2-LR-r8i1p1f1',
       'MPI-ESM1-2-LR-r9i1p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'MPI-ESM1-2-LR-r10i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 18
keys = ['MRI-ESM2-0-r1i1p1f1', 'MRI-ESM2-0-r1i2p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'MRI-ESM2-0-r1i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 19
keys = ['NESM3-r1i1p1f1', 'NESM3-r2i1p1f1']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'NESM3-r1i1p1f1'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

#########
# Model 20
keys = ['UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
        'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2', 'UKESM1-0-LL-r8i1p1f2']

dict_model = {}
for ii in range(len(keys)):
    dict_model[keys[ii]] = (dsT6_target_ts_norm.sel(member=keys[ii]).tas.item(0),dsPr6_target_ts_norm.sel(member=keys[ii]).pr.item(0))

for ii in range(len(keys)):
    choose_furthest_member(keys[ii])

key_choice = 'UKESM1-0-LL-r1i1p1f2'
dict_ind[key_choice] = (dsT6_target_ts_norm.sel(member=key_choice).tas.item(0),dsPr6_target_ts_norm.sel(member=key_choice).pr.item(0))

################################

dsT6_target_ts_spd_all = dsT6_target_ts.sel(member=['ACCESS-CM2-r2i1p1f1','ACCESS-ESM1-5-r1i1p1f1',
       'AWI-CM-1-1-MR-r1i1p1f1', 'CAS-ESM2-0-r1i1p1f1',
       'CESM2-WACCM-r2i1p1f1','CESM2-r11i1p1f1','CMCC-CM2-SR5-r1i1p1f1',
       'CMCC-ESM2-r1i1p1f1', 'CNRM-CM6-1-HR-r1i1p1f2','CNRM-CM6-1-r4i1p1f2','CNRM-ESM2-1-r2i1p1f2',
       'CanESM5-r16i1p1f1','E3SM-1-1-r1i1p1f1', 'FGOALS-f3-L-r1i1p1f1', 'FGOALS-g3-r2i1p1f1',
       'GFDL-CM4-r1i1p1f1', 'GFDL-ESM4-r1i1p1f1','GISS-E2-1-G-r1i1p3f1',
       'HadGEM3-GC31-LL-r3i1p1f3','HadGEM3-GC31-MM-r1i1p1f3', 'INM-CM4-8-r1i1p1f1', 'INM-CM5-0-r1i1p1f1',
       'IPSL-CM6A-LR-r6i1p1f1','KACE-1-0-G-r3i1p1f1','KIOST-ESM-r1i1p1f1','MIROC-ES2L-r1i1p1f2','MIROC6-r15i1p1f1',
       'MPI-ESM1-2-HR-r1i1p1f1','MPI-ESM1-2-LR-r10i1p1f1','MRI-ESM2-0-r1i1p1f1','NESM3-r1i1p1f1',
       'NorESM2-MM-r1i1p1f1','TaiESM1-r1i1p1f1','UKESM1-0-LL-r1i1p1f2']).tas

dsPr6_target_ts_spd_all = dsPr6_target_ts.sel(member=['ACCESS-CM2-r2i1p1f1','ACCESS-ESM1-5-r1i1p1f1',
       'AWI-CM-1-1-MR-r1i1p1f1', 'CAS-ESM2-0-r1i1p1f1',
       'CESM2-WACCM-r2i1p1f1','CESM2-r11i1p1f1','CMCC-CM2-SR5-r1i1p1f1',
       'CMCC-ESM2-r1i1p1f1', 'CNRM-CM6-1-HR-r1i1p1f2','CNRM-CM6-1-r4i1p1f2','CNRM-ESM2-1-r2i1p1f2',
       'CanESM5-r16i1p1f1','E3SM-1-1-r1i1p1f1', 'FGOALS-f3-L-r1i1p1f1', 'FGOALS-g3-r2i1p1f1',
       'GFDL-CM4-r1i1p1f1', 'GFDL-ESM4-r1i1p1f1','GISS-E2-1-G-r1i1p3f1',
       'HadGEM3-GC31-LL-r3i1p1f3','HadGEM3-GC31-MM-r1i1p1f3', 'INM-CM4-8-r1i1p1f1', 'INM-CM5-0-r1i1p1f1',
       'IPSL-CM6A-LR-r6i1p1f1','KACE-1-0-G-r3i1p1f1','KIOST-ESM-r1i1p1f1','MIROC-ES2L-r1i1p1f2','MIROC6-r15i1p1f1',
       'MPI-ESM1-2-HR-r1i1p1f1','MPI-ESM1-2-LR-r10i1p1f1','MRI-ESM2-0-r1i1p1f1','NESM3-r1i1p1f1',
       'NorESM2-MM-r1i1p1f1','TaiESM1-r1i1p1f1','UKESM1-0-LL-r1i1p1f2']).pr

# ################################################
fig = plt.figure(figsize=(14,6))
ax = plt.subplot(121)
# ################################################

plt.axvline(dsT6_target_ts_em.median("member").data,color='k',linewidth=1,linestyle='dashed',alpha=0.2)
plt.axhline(dsPr6_target_ts_em.median("member").data,color='k',linewidth=1,linestyle='dashed',alpha=0.2)

c = np.arange(0,34)
plt.scatter(dsT6_target_ts.tas,dsPr6_target_ts.pr,s=5,c="silver",marker='.',alpha=0.5)
plt.scatter(dsT6_target_ts_em.isel(member=0),dsPr6_target_ts_em.isel(member=0),c='tab:red',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=2),dsPr6_target_ts_em.isel(member=2),c='tab:orange',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=3),dsPr6_target_ts_em.isel(member=3),c='tab:cyan',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=4),dsPr6_target_ts_em.isel(member=4),c='darkgoldenrod',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=5),dsPr6_target_ts_em.isel(member=5),c='darkgoldenrod',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=6),dsPr6_target_ts_em.isel(member=6),c='darkgoldenrod',s=20,marker='^',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=7),dsPr6_target_ts_em.isel(member=7),c='darkgoldenrod',s=30,marker='*',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=8),dsPr6_target_ts_em.isel(member=8),c='cornflowerblue',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=9),dsPr6_target_ts_em.isel(member=9),c='cornflowerblue',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=10),dsPr6_target_ts_em.isel(member=10),c='cornflowerblue',s=20,marker='^',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=11),dsPr6_target_ts_em.isel(member=11),c='dodgerblue',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=12),dsPr6_target_ts_em.isel(member=12),c='k',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=13),dsPr6_target_ts_em.isel(member=13),c='maroon',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=14),dsPr6_target_ts_em.isel(member=14),c='maroon',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=15),dsPr6_target_ts_em.isel(member=15),c='indigo',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=16),dsPr6_target_ts_em.isel(member=16),c='indigo',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=18),dsPr6_target_ts_em.isel(member=18),c='tab:red',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=19),dsPr6_target_ts_em.isel(member=19),c='tab:red',s=20,marker='^',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=20),dsPr6_target_ts_em.isel(member=20),c='mediumseagreen',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=21),dsPr6_target_ts_em.isel(member=21),c='mediumseagreen',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=22),dsPr6_target_ts_em.isel(member=22),c='royalblue',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=23),dsPr6_target_ts_em.isel(member=23),c='tab:red',s=30,marker='*',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=24),dsPr6_target_ts_em.isel(member=24),c='darkslateblue',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=25),dsPr6_target_ts_em.isel(member=25),c='lightsalmon',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=26),dsPr6_target_ts_em.isel(member=26),c='lightsalmon',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=27),dsPr6_target_ts_em.isel(member=27),c='tab:orange',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=28),dsPr6_target_ts_em.isel(member=28),c='tab:orange',s=20,marker='^',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=29),dsPr6_target_ts_em.isel(member=29),c='palevioletred',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=30),dsPr6_target_ts_em.isel(member=30),c='tab:orange',s=30,marker='*',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=31),dsPr6_target_ts_em.isel(member=31),c='darkgoldenrod',s=20,marker='d',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=32),dsPr6_target_ts_em.isel(member=32),c='darkgoldenrod',s=30,marker='+',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=33),dsPr6_target_ts_em.isel(member=33),c='tab:red',s=20,marker='d',alpha=1)

plt.scatter(dsT6_target_ts_em.isel(member=1),dsPr6_target_ts_em.isel(member=1),c='tab:red',s=30,marker='+',alpha=1)
plt.scatter(dsT6_target_ts_em.isel(member=17),dsPr6_target_ts_em.isel(member=17),c='blueviolet',s=20,marker='x',alpha=1)

plt.xlabel('JJA CEU SAT Change (˚C; 2041/2060 - 1995/2014)')
plt.xlim([0.5,6])
plt.ylabel('JJA CEU PR Change (mm/day; 2041/2060 - 1995/2014)')
plt.ylim([-0.7,0.5])
plt.title('a) SAT-PR Change, Ensemble Means',fontsize=14,fontweight='bold',loc='left')

################################################

ax = plt.subplot(122)

plt.axvline(dsT6_target_ts_spd_all.median("member").data,color='k',linewidth=1,linestyle='dashed',alpha=0.2)
plt.axhline(dsPr6_target_ts_spd_all.median("member").data,color='k',linewidth=1,linestyle='dashed',alpha=0.2)

c = np.arange(0,34)
plt.scatter(dsT6_target_ts.tas,dsPr6_target_ts.pr,s=5,c="silver",marker='.',alpha=0.5)
plt.scatter(dsT6_target_ts_spd_all.isel(member=0),dsPr6_target_ts_spd_all.isel(member=0),c='tab:red',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=2),dsPr6_target_ts_spd_all.isel(member=2),c='tab:orange',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=3),dsPr6_target_ts_spd_all.isel(member=3),c='tab:cyan',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=4),dsPr6_target_ts_spd_all.isel(member=4),c='darkgoldenrod',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=5),dsPr6_target_ts_spd_all.isel(member=5),c='darkgoldenrod',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=6),dsPr6_target_ts_spd_all.isel(member=6),c='darkgoldenrod',s=20,marker='^',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=7),dsPr6_target_ts_spd_all.isel(member=7),c='darkgoldenrod',s=30,marker='*',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=8),dsPr6_target_ts_spd_all.isel(member=8),c='cornflowerblue',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=9),dsPr6_target_ts_spd_all.isel(member=9),c='cornflowerblue',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=10),dsPr6_target_ts_spd_all.isel(member=10),c='cornflowerblue',s=20,marker='^',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=11),dsPr6_target_ts_spd_all.isel(member=11),c='dodgerblue',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=12),dsPr6_target_ts_spd_all.isel(member=12),c='k',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=13),dsPr6_target_ts_spd_all.isel(member=13),c='maroon',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=14),dsPr6_target_ts_spd_all.isel(member=14),c='maroon',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=15),dsPr6_target_ts_spd_all.isel(member=15),c='indigo',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=16),dsPr6_target_ts_spd_all.isel(member=16),c='indigo',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=18),dsPr6_target_ts_spd_all.isel(member=18),c='tab:red',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=19),dsPr6_target_ts_spd_all.isel(member=19),c='tab:red',s=20,marker='^',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=20),dsPr6_target_ts_spd_all.isel(member=20),c='mediumseagreen',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=21),dsPr6_target_ts_spd_all.isel(member=21),c='mediumseagreen',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=22),dsPr6_target_ts_spd_all.isel(member=22),c='royalblue',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=23),dsPr6_target_ts_spd_all.isel(member=23),c='tab:red',s=30,marker='*',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=24),dsPr6_target_ts_spd_all.isel(member=24),c='darkslateblue',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=25),dsPr6_target_ts_spd_all.isel(member=25),c='lightsalmon',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=26),dsPr6_target_ts_spd_all.isel(member=26),c='lightsalmon',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=27),dsPr6_target_ts_spd_all.isel(member=27),c='tab:orange',s=20,marker='o',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=28),dsPr6_target_ts_spd_all.isel(member=28),c='tab:orange',s=20,marker='^',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=29),dsPr6_target_ts_spd_all.isel(member=29),c='palevioletred',s=20,marker='x',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=30),dsPr6_target_ts_spd_all.isel(member=30),c='tab:orange',s=30,marker='*',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=31),dsPr6_target_ts_spd_all.isel(member=31),c='darkgoldenrod',s=20,marker='d',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=32),dsPr6_target_ts_spd_all.isel(member=32),c='darkgoldenrod',s=30,marker='+',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=33),dsPr6_target_ts_em.isel(member=33),c='tab:red',s=20,marker='d',alpha=1)

plt.scatter(dsT6_target_ts_spd_all.isel(member=1),dsPr6_target_ts_spd_all.isel(member=1),c='tab:red',s=30,marker='+',alpha=1)
plt.scatter(dsT6_target_ts_spd_all.isel(member=17),dsPr6_target_ts_spd_all.isel(member=17),c='blueviolet',s=20,marker='x',alpha=1)

plt.xlabel('JJA CEU SAT Change (˚C; 2041/2060 - 1995/2014)')
plt.xlim([0.5,6])
ax.set_yticklabels('')
plt.ylim([-0.7,0.5])
plt.title('b) SAT-PR Change, Individual Member',fontsize=14,fontweight='bold',loc='left')
plt.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig('Fig9_CMIP6_JJA_CEU_EM_IM_spread_comparison.png',bbox_inches='tight',dpi=300)
