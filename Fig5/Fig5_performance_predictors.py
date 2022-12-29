# Figure 5: makes the JJA/DJF predictor plot
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

dirobs = '/**/CMIP_subselection/Data/'
dsSSTobs_ann = xr.open_dataset(dirobs + 'tos_mon_OBS_g025_NAWH_fix_ann.nc',use_cftime = True)
dsSWobs_ann = xr.open_dataset(dirobs + 'swcre_mon_OBS_g025_LWCLD_ann.nc',use_cftime = True)
dsSWobs_jja = xr.open_dataset(dirobs + 'swcre_mon_OBS_g025_CEU_jja.nc',use_cftime = True)
dsTobs_ann = xr.open_dataset(dirobs + 'tas_mon_OBS_g025_EUR_ann.nc',use_cftime = True)
dsProbs_CEU_jja = xr.open_dataset(dirobs + 'pr_mon_OBS_g025_CEU_JJA.nc',use_cftime = True)
dsPobs_djf = xr.open_dataset(dirobs + 'psl_mon_OBS_g025_TW2_DJF.nc',use_cftime = True)
dsPobs_djf = dsPobs_djf/100
dsProbs_djf = xr.open_dataset(dirobs + 'pr_mon_OBS_g025_NEU_DJF.nc',use_cftime = True)

## Compute Time Periods

dsSSTobs_ann_base = dsSSTobs_ann.sel(year=slice('1995','2014')).mean('year')
dsSWobs_ann_base = dsSWobs_ann.sel(year=slice('2001','2018')).mean('year')
dsSWobs_jja_base = dsSWobs_jja.sel(year=slice('2001','2018')).mean('year')
dsPobs_djf_base = dsPobs_djf.sel(year=slice('1950','2014')).mean('year')
dsTobs_ann_his = dsTobs_ann.sel(year=slice('1950','1969')).mean('year')
dsTobs_ann_base = dsTobs_ann.sel(year=slice('1995','2014')).mean('year')
dsProbs_CEU_jja_base = dsProbs_CEU_jja.sel(year=slice('1995','2014')).mean('year')
dsProbs_djf_base = dsProbs_djf.sel(year=slice('1995','2014')).mean('year')


import matplotlib.gridspec as gridspec

fig3 = plt.figure(figsize=(14, 6))
gs = fig3.add_gridspec(2, 10)
# Annual European SAT climatology; 1950-1969
f3_ax1 = fig3.add_subplot(gs[0, 0:2],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('YlOrRd')
dsTobs_ann_his.tas.plot.pcolormesh(ax=f3_ax1, transform=ccrs.PlateCarree(), cmap=cmap,vmin=-5,vmax=25,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': 'a) Annual SAT Climatology;\n 1950-1969 (˚C)',
                                            'fraction':0.046, 'pad':0.04})
f3_ax1.coastlines();
f3_ax1.set_extent([-20,42,28,76], crs=ccrs.PlateCarree());

# Annual European SAT climatology; 1995-2014
f3_ax2 = fig3.add_subplot(gs[0, 2:4],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('YlOrRd')
dsTobs_ann_base.tas.plot.pcolormesh(ax=f3_ax2, transform=ccrs.PlateCarree(), cmap=cmap,vmin=-5,vmax=25,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': 'b) Annual SAT Climatology;\n 1995-2014 (˚C)',
                                            'fraction':0.046, 'pad':0.04})
f3_ax2.coastlines();
f3_ax2.set_extent([-20,42,28,76], crs=ccrs.PlateCarree());
f3_ax2.set_title('Base Set',fontweight='bold')

# Annual NATL SST climatology; 1995-2014
f3_ax5 = fig3.add_subplot(gs[0, 4:6],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('YlOrRd')
dsSSTobs_ann_base.tos.plot.pcolormesh(ax=f3_ax5, transform=ccrs.PlateCarree(), cmap=cmap,vmin=-5,vmax=25,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': 'c) Annual SST Climatology;\n 1995-2014 (˚C)',
                                            'fraction':0.046, 'pad':0.04})
f3_ax5.coastlines();
f3_ax5.set_extent([-63,-1,28,76], crs=ccrs.PlateCarree());

##
f3_ax3 = fig3.add_subplot(gs[0, 6:8],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('YlGnBu')
dsProbs_CEU_jja_base.pr.plot.pcolormesh(ax=f3_ax3, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=6,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': 'd) JJA PR Climatology;\n 1995-2014 (mm/day)',
                                            'fraction':0.046, 'pad':0.04})
f3_ax3.coastlines();
f3_ax3.set_extent([-20,42,28,76], crs=ccrs.PlateCarree());
f3_ax3.set_title('For JJA CEU',fontweight='bold')

# JJA CEU SW climatology; 2001-2018
f3_ax4 = fig3.add_subplot(gs[1, 6:8],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('magma_r')
dsSWobs_jja_base.swcre.plot.pcolormesh(ax=f3_ax4, transform=ccrs.PlateCarree(), cmap=cmap,vmin=-110,vmax=-20,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': 'g) JJA SWCRE Climatology;\n 2001-2018 (W/m^2)',
                                            'fraction':0.046, 'pad':0.04})
f3_ax4.coastlines();
f3_ax4.set_extent([-20,42,28,76], crs=ccrs.PlateCarree());

f3_ax7 = fig3.add_subplot(gs[0, 8:10],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('YlGnBu')
dsProbs_djf_base.pr.plot.pcolormesh(ax=f3_ax7, transform=ccrs.PlateCarree(), cmap=cmap,vmin=0,vmax=6,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': 'e) DJF PR Climatology;\n 1995-2014 (mm/day)',
                                            'fraction':0.046, 'pad':0.04})
f3_ax7.coastlines();
f3_ax7.set_extent([-20,42,28,76], crs=ccrs.PlateCarree());
f3_ax7.set_title('For DJF NEU',fontweight='bold')

# JJA CEU SW climatology; 2001-2018
f3_ax8 = fig3.add_subplot(gs[1, 8:10],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('RdYlBu_r')
dsPobs_djf_base.psl.plot.pcolormesh(ax=f3_ax8, transform=ccrs.PlateCarree(), cmap=cmap,vmin=995,vmax=1025,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': 'h) DJF SLP Climatology;\n 1950-2014 (hPa)',
                                            'fraction':0.046, 'pad':0.04})
f3_ax8.coastlines();
f3_ax8.set_extent([-36,26,30,78], crs=ccrs.PlateCarree());



# Annual SH SW climatology; 2001-2018
f3_ax6 = fig3.add_subplot(gs[1, 0:6],projection=ccrs.PlateCarree())
cmap = plt.get_cmap('magma_r')
dsSWobs_ann_base.swcre.plot.pcolormesh(ax=f3_ax6, transform=ccrs.PlateCarree(), cmap=cmap,vmin=-110,vmax=-20,
                                            cbar_kwargs={'orientation': 'horizontal',
                                            'label': 'f) Annual SWCRE Climatology;\n 2001-2018 (W/m^2)',
                                            'fraction':0.046, 'pad':0.04})
f3_ax6.coastlines();
f3_ax6.set_extent([-180,180,-87,0], crs=ccrs.PlateCarree());
plt.savefig('Fig5_all_predictor_fields.png',bbox_inches='tight',dpi=300)
