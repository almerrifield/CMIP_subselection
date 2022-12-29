# Supplementary Figure 3: makes internal variability across SMILEs plots for psl

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

dsP6 = xr.open_dataset(dirT6 + 'psl_mon_CMIP6_SSP585_g025_v2_ann.nc',use_cftime = True)
dsP6 = dsP6/100
dsP6 = dsP6.sortby(dsP6.member)

dsP6_clim = dsP6.sel(year=slice(1905, 2005)).mean('year')

dsP6_clim = dsP6_clim.drop_sel(member=['BCC-CSM2-MR-r1i1p1f1','CIESM-r1i1p1f1','FIO-ESM-2-0-r1i1p1f1','FIO-ESM-2-0-r2i1p1f1', 'FIO-ESM-2-0-r3i1p1f1','EC-Earth3-Veg-r6i1p1f1'])#,'MCM-UA-1-0-r1i1p1f2'])


#####################################################

dirT5 = '/**/CMIP_subselection/Data/'

dsP5 = xr.open_dataset(dirT5 + 'psl_mon_CMIP5_rcp85_g025_v2_ann.nc',use_cftime = True)
dsP5 = dsP5/100
dsP5 = dsP5.sortby(dsP5.member)

dsP5_clim = dsP5.sel(year=slice(1905, 2005)).mean('year')

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


#########################################################

fig3 = plt.figure(figsize=(8, 12))
gs = fig3.add_gridspec(6, 3)

#########################################################

f3_ax1 = fig3.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_access_std.psl.plot.pcolormesh(ax=f3_ax1, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax1.coastlines();
f3_ax1.set_title('a) ACCESS-ESM1-5 (10) $\sigma$',fontsize=11)


f3_ax2 = fig3.add_subplot(gs[0, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_can_std.psl.plot.pcolormesh(ax=f3_ax2, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax2.coastlines();
f3_ax2.set_title('b) CanESM5 (25) $\sigma$',fontsize=11)

f3_ax3 = fig3.add_subplot(gs[0, 2],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_cesm_std.psl.plot.pcolormesh(ax=f3_ax3, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax3.coastlines();
f3_ax3.set_title(' c) CESM2 (5) $\sigma$',fontsize=11)

f3_ax4 = fig3.add_subplot(gs[1, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_cnrm_std.psl.plot.pcolormesh(ax=f3_ax4, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax4.coastlines();
f3_ax4.set_title('d) CNRM-CM6-1 (6) $\sigma$',fontsize=11)

f3_ax5 = fig3.add_subplot(gs[1, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_cnrm_e_std.psl.plot.pcolormesh(ax=f3_ax5, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax5.coastlines();
f3_ax5.set_title('e) CNRM-ESM2-1 (5) $\sigma$',fontsize=11)

f3_ax6 = fig3.add_subplot(gs[1, 2],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_earth_std.psl.plot.pcolormesh(ax=f3_ax6, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax6.coastlines();
f3_ax6.set_title('f) EC-Earth3 (8) $\sigma$',fontsize=11)

f3_ax7 = fig3.add_subplot(gs[2, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_giss_std.psl.plot.pcolormesh(ax=f3_ax7, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax7.coastlines();
f3_ax7.set_title('g) GISS-E2-1-G (5) $\sigma$',fontsize=11)

f3_ax8 = fig3.add_subplot(gs[2, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_ipsl_std.psl.plot.pcolormesh(ax=f3_ax8, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax8.coastlines();
f3_ax8.set_title('h) IPSL-CM6A-LR (6) $\sigma$',fontsize=11)

f3_ax9 = fig3.add_subplot(gs[2, 2],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_miroc_e_std.psl.plot.pcolormesh(ax=f3_ax9, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax9.coastlines();
f3_ax9.set_title('i) MIROC-ES2L (10) $\sigma$',fontsize=11)

f3_ax10 = fig3.add_subplot(gs[3, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_miroc_std.psl.plot.pcolormesh(ax=f3_ax10, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax10.coastlines();
f3_ax10.set_title('j) MIROC6 (50) $\sigma$',fontsize=11)


f3_ax11 = fig3.add_subplot(gs[3, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP6_clim_mpi_std.psl.plot.pcolormesh(ax=f3_ax11, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax11.coastlines();
f3_ax11.set_title('k) MPI-ESM1-2-LR (10) $\sigma$',fontsize=11)


f3_ax12 = fig3.add_subplot(gs[3, 2],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
pcm = dsP6_clim_ukesm_std.psl.plot.pcolormesh(ax=f3_ax12, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax12.coastlines();
f3_ax12.set_title('l) UKESM1-0-LL (5) $\sigma$',fontsize=11)

##################
f3_ax13 = fig3.add_subplot(gs[4, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP5_clim_can_std.psl.plot.pcolormesh(ax=f3_ax13, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax13.coastlines();
f3_ax13.set_title('m) CanESM2 (5) $\sigma$',fontsize=11)


f3_ax14 = fig3.add_subplot(gs[4, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP5_clim_ccsm4_std.psl.plot.pcolormesh(ax=f3_ax14, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax14.coastlines();
f3_ax14.set_title('n) CCSM4 (6) $\sigma$',fontsize=11)

f3_ax15 = fig3.add_subplot(gs[4, 2],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP5_clim_cnrm_std.psl.plot.pcolormesh(ax=f3_ax15, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax15.coastlines();
f3_ax15.set_title('o) CNRM-CM5 (5) $\sigma$',fontsize=11)

f3_ax16 = fig3.add_subplot(gs[5, 0],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
dsP5_clim_csiro_std.psl.plot.pcolormesh(ax=f3_ax16, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax16.coastlines();
f3_ax16.set_title('p) CSIRO-Mk3-6-0 (10) $\sigma$',fontsize=11)

f3_ax17 = fig3.add_subplot(gs[5, 1],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('BuPu')
pcm = dsP5_clim_earth_std.psl.plot.pcolormesh(ax=f3_ax17, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=0.6,add_colorbar=False)
f3_ax17.coastlines();
f3_ax17.set_title('q) EC-EARTH (5) $\sigma$',fontsize=11)

axes = [f3_ax1,f3_ax2,f3_ax3,f3_ax4,f3_ax5,f3_ax6,f3_ax7,f3_ax8,f3_ax9,
f3_ax10,f3_ax11,f3_ax12,f3_ax13,f3_ax14,f3_ax15,f3_ax16,f3_ax17]
cbar = fig3.colorbar(pcm,ax=axes,location='bottom',
aspect=40,pad=0.025,shrink=0.6)
cbar.ax.tick_params(labelsize=11)
cbar.set_label(label='SLP Standard Deviation (hPa)',size=11)

plt.savefig('SupFig3_IV_SLP_1905_2005.png',bbox_inches='tight',dpi=300)
