# Supplementary Figure 1: makes CMIP5 "fingerprint" hatched map plots

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

dirT5 = '/**/CMIP_subselection/Data/'
dsT5 = xr.open_dataset(dirT5 + 'tas_mon_CMIP5_rcp85_g025_v2_ann.nc',use_cftime = True)
dsT5 = dsT5-273.15
dsT5 = dsT5.sortby(dsT5.member)

dsP5 = xr.open_dataset(dirT5 + 'psl_mon_CMIP5_rcp85_g025_v2_ann.nc',use_cftime = True)
dsP5 = dsP5/100
dsP5 = dsP5.sortby(dsP5.member)

dsT5_clim = dsT5.sel(year=slice(1905, 2005)).mean('year')
dsP5_clim = dsP5.sel(year=slice(1905, 2005)).mean('year')

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
dsT5_clim_std = dsT5_clim.std('member')

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

#############################

dsP5_clim_std = dsP5_clim.std('member')

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

#############################
fig3 = plt.figure(figsize=(8, 9))
gs = fig3.add_gridspec(3, 2)

# Annual Global Land SAT climatology; 1905-2005
f3_ax1 = fig3.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsT5_clim_one_std.tas.plot.pcolormesh(ax=f3_ax1, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=7.5,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',
                                            'fraction':0.046, 'pad':0.04})
f3_ax1.contourf(mask_obs_flip.lon,mask_obs_flip.lat,mask_obs_T5,[-1,0,1],colors='none', hatches=['++', None],transform=ccrs.PlateCarree())
f3_ax1.coastlines();
f3_ax1.set_title('a) Annual SAT Climatology, \n Inter-model $\sigma$ (˚C)',fontsize=11)

f3_ax2 = fig3.add_subplot(gs[1, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsT5_clim_iv_std_mean.tas.plot.pcolormesh(ax=f3_ax2, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=.75, #add_colorbar=False)
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '', #'ticks': [-2.5,0.0,2.5],
                                            'fraction':0.046, 'pad':0.04})
f3_ax2.contourf(mask_obs13_flip.lon,mask_obs13_flip.lat,mask_obs13_T5,[-1,0,1],colors='none', hatches=['xx', None],transform=ccrs.PlateCarree())
f3_ax2.coastlines();
f3_ax2.set_title('c) Annual SAT Climatology, \n Median Internal $\sigma$ (˚C)',fontsize=11)


f3_ax4 = fig3.add_subplot(gs[0, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP5_clim_one_std.psl.plot.pcolormesh(ax=f3_ax4, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=7.5,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',#'ticks': [-2.5,0.0,2.5],
                                            'fraction':0.046, 'pad':0.04})
f3_ax4.contourf(mask_obs_flip.lon,mask_obs_flip.lat,mask_obs_P5,[-1,0,1],colors='none', hatches=['++', None],transform=ccrs.PlateCarree())
f3_ax4.coastlines();
f3_ax4.set_title('b) Annual SLP Climatology, \n Inter-model $\sigma$ (hPa)',fontsize=11)

# Annual NH SLP climatology; 1905-2005
f3_ax5 = fig3.add_subplot(gs[1, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP5_clim_iv_std_mean.psl.plot.pcolormesh(ax=f3_ax5, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.75,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',
                                            'fraction':0.046, 'pad':0.04})
f3_ax5.contourf(mask_obs13_flip.lon,mask_obs13_flip.lat,mask_obs13_P5,[-1,0,1],colors='none', hatches=['xx', None],transform=ccrs.PlateCarree())
f3_ax5.coastlines();
f3_ax5.set_title('d) Annual SLP Climatology, \n Median Internal $\sigma$ (hPa)',fontsize=11)

# Annual Global Land SAT climatology; 1905-2005
f3_ax7 = fig3.add_subplot(gs[2, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('RdBu_r')
dsT5_clim_mask.mean('member').tas.plot.pcolormesh(ax=f3_ax7, transform=ccrs.PlateCarree(), cmap=cmap,vmin=-40,vmax=40,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',#'ticks': [-2.5,0.0,2.5],
                                            'fraction':0.046, 'pad':0.04})
f3_ax7.contourf(mask_obs_flip.lon,mask_obs_flip.lat,mask_obs_T5,[-1,0,1],colors='none', hatches=['++', None],transform=ccrs.PlateCarree())
f3_ax7.contourf(mask_obs13_flip.lon,mask_obs13_flip.lat,mask_obs13_T5,[-1,0,1],colors='none', hatches=['xx', None],transform=ccrs.PlateCarree())
f3_ax7.coastlines();
f3_ax7.set_title('e) Annual SAT Climatology, \n Ensemble Mean "Fingerprint" (˚C)',fontsize=11)

# Annual NH SLP climatology; 1905-2005
f3_ax8 = fig3.add_subplot(gs[2, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('RdBu_r')
dsP5_clim_mask.mean('member').psl.plot.pcolormesh(ax=f3_ax8, transform=ccrs.PlateCarree(), cmap=cmap,vmin=980,vmax=1030,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': '',
                                            'fraction':0.046, 'pad':0.04})
f3_ax8.contourf(mask_obs_flip.lon,mask_obs_flip.lat,mask_obs_P5,[-1,0,1],colors='none', hatches=['++', None],transform=ccrs.PlateCarree())
f3_ax8.contourf(mask_obs13_flip.lon,mask_obs13_flip.lat,mask_obs13_P5,[-1,0,1],colors='none', hatches=['xx', None],transform=ccrs.PlateCarree())
f3_ax8.coastlines();
f3_ax8.set_title('f) Annual SLP Climatology, \n Ensemble Mean "Fingerprint" (hPa)',fontsize=11)

plt.savefig('SupFig1_CMIP5_fingerprints_1905_2005.png',bbox_inches='tight',dpi=300)
