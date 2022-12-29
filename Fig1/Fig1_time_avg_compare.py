# Figure 1: comparing averaging period

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

##################################

dirT6 = '/**/CMIP_subselection/Data/'
dsT6 = xr.open_dataset(dirT6 + 'tas_mon_CMIP6_SSP585_g025_v2_ann.nc',use_cftime = True)
dsT6 = dsT6-273.15
dsT6 = dsT6.sortby(dsT6.member)

dsP6 = xr.open_dataset(dirT6 + 'psl_mon_CMIP6_SSP585_g025_v2_ann.nc',use_cftime = True)
dsP6 = dsP6/100
dsP6 = dsP6.sortby(dsP6.member)

dsT6_clim = dsT6.sel(year=slice(1905, 2005)).mean('year')
dsP6_clim = dsP6.sel(year=slice(1905, 2005)).mean('year')

dsT6_clim = dsT6_clim.drop_sel(member=['NorESM2-LM-r1i1p1f1','FIO-ESM-2-0-r1i1p1f1','FIO-ESM-2-0-r2i1p1f1', 'FIO-ESM-2-0-r3i1p1f1'])#,'MCM-UA-1-0-r1i1p1f2'])
dsP6_clim = dsP6_clim.drop_sel(member=['BCC-CSM2-MR-r1i1p1f1','CIESM-r1i1p1f1','FIO-ESM-2-0-r1i1p1f1','FIO-ESM-2-0-r2i1p1f1', 'FIO-ESM-2-0-r3i1p1f1','EC-Earth3-Veg-r6i1p1f1'])#,'MCM-UA-1-0-r1i1p1f2'])

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

dsT6_clim_test = get_error(dsT6_clim.tas,weights)
dsT6_clim_test = dsT6_clim_test.where(dsT6_clim_test!=0)
dsT6_clim_test_norm = dsT6_clim_test/np.nanmean(dsT6_clim_test)

dsP6_clim_test = get_error(dsP6_clim.psl,weights)
dsP6_clim_test = dsP6_clim_test.where(dsP6_clim_test!=0)
dsP6_clim_test_norm = dsP6_clim_test/np.nanmean(dsP6_clim_test)

dsWi = (dsT6_clim_test_norm + dsP6_clim_test_norm)/2 ###############

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


######################################
nesm3_fam = ['NESM3-r1i1p1f1','NESM3-r2i1p1f1']

dsWi_nesm3_fam = dsWi.sel(member1=nesm3_fam,member2=nesm3_fam)
nesm3_fam_all = np.unique(dsWi_nesm3_fam)
nesm3_fam_all = nesm3_fam_all[~np.isnan(nesm3_fam_all)]

dsWi_nesm3_rest = dsWi.sel(member1=nesm3_fam).drop_sel(member2=nesm3_fam)
nesm3_rest_all = np.unique(dsWi_nesm3_rest)
nesm3_rest_all = nesm3_rest_all[~np.isnan(nesm3_rest_all)]

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

had_ll_fam = ['HadGEM3-GC31-LL-r1i1p1f3',
'HadGEM3-GC31-LL-r2i1p1f3', 'HadGEM3-GC31-LL-r3i1p1f3',
'HadGEM3-GC31-LL-r4i1p1f3']

dsWi_had_ll_fam = dsWi.sel(member1=had_ll_fam,member2=had_ll_fam)
had_ll_fam_all = np.unique(dsWi_had_ll_fam)
had_ll_fam_all = had_ll_fam_all[~np.isnan(had_ll_fam_all)]

dsWi_had_ll_rest = dsWi.sel(member1=had_ll_fam).drop_sel(member2=had_ll_fam)
had_ll_rest_all = np.unique(dsWi_had_ll_rest)
had_ll_rest_all = had_ll_rest_all[~np.isnan(had_ll_rest_all)]

had_mm_fam = ['HadGEM3-GC31-MM-r1i1p1f3',
'HadGEM3-GC31-MM-r2i1p1f3', 'HadGEM3-GC31-MM-r3i1p1f3',
'HadGEM3-GC31-MM-r4i1p1f3']

dsWi_had_mm_fam = dsWi.sel(member1=had_mm_fam,member2=had_mm_fam)
had_mm_fam_all = np.unique(dsWi_had_mm_fam)
had_mm_fam_all = had_mm_fam_all[~np.isnan(had_mm_fam_all)]

dsWi_had_mm_rest = dsWi.sel(member1=had_mm_fam).drop_sel(member2=had_mm_fam)
had_mm_rest_all = np.unique(dsWi_had_mm_rest)
had_mm_rest_all = had_mm_rest_all[~np.isnan(had_mm_rest_all)]

kace_fam = ['KACE-1-0-G-r1i1p1f1','KACE-1-0-G-r2i1p1f1', 'KACE-1-0-G-r3i1p1f1']

dsWi_kace_fam = dsWi.sel(member1=kace_fam,member2=kace_fam)
kace_fam_all = np.unique(dsWi_kace_fam)
kace_fam_all = kace_fam_all[~np.isnan(kace_fam_all)]

dsWi_kace_rest = dsWi.sel(member1=kace_fam).drop_sel(member2=kace_fam)
kace_rest_all = np.unique(dsWi_kace_rest)
kace_rest_all = kace_rest_all[~np.isnan(kace_rest_all)]

ukesm1_fam = ['UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2','UKESM1-0-LL-r8i1p1f2']

dsWi_ukesm1_fam = dsWi.sel(member1=ukesm1_fam,member2=ukesm1_fam)
ukesm1_fam_all = np.unique(dsWi_ukesm1_fam)
ukesm1_fam_all = ukesm1_fam_all[~np.isnan(ukesm1_fam_all)]

dsWi_ukesm1_rest = dsWi.sel(member1=ukesm1_fam).drop_sel(member2=ukesm1_fam)
ukesm1_rest_all = np.unique(dsWi_ukesm1_rest)
ukesm1_rest_all = ukesm1_rest_all[~np.isnan(ukesm1_rest_all)]


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


cesm_fam = ['CESM2-r10i1p1f1', 'CESM2-r11i1p1f1',
'CESM2-r1i1p1f1', 'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1']

# closest relative: 'AWI-CM-1-1-MR-r1i1p1f1'

dsWi_cesm_fam = dsWi.sel(member1=cesm_fam,member2=cesm_fam)
cesm_fam_all = np.unique(dsWi_cesm_fam)
cesm_fam_all = cesm_fam_all[~np.isnan(cesm_fam_all)]

dsWi_cesm_rest = dsWi.sel(member1=cesm_fam).drop_sel(member2=cesm_fam)
cesm_rest_all = np.unique(dsWi_cesm_rest)
cesm_rest_all = cesm_rest_all[~np.isnan(cesm_rest_all)]

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


#####################################

ec_earth_veg_fam = ['EC-Earth3-Veg-r1i1p1f1', 'EC-Earth3-Veg-r2i1p1f1',
'EC-Earth3-Veg-r3i1p1f1', 'EC-Earth3-Veg-r4i1p1f1']

dsWi_earth_veg_fam = dsWi.sel(member1=ec_earth_veg_fam,member2=ec_earth_veg_fam)
earth_veg_fam_all = np.unique(dsWi_earth_veg_fam)
earth_veg_fam_all = earth_veg_fam_all[~np.isnan(earth_veg_fam_all)]

dsWi_earth_veg_rest = dsWi.sel(member1=ec_earth_veg_fam).drop_sel(member2=ec_earth_veg_fam)
earth_veg_rest_all = np.unique(dsWi_earth_veg_rest)
earth_veg_rest_all = earth_veg_rest_all[~np.isnan(earth_veg_rest_all)]

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

X17 = dsWi.sel(member1='TaiESM1-r1i1p1f1')
Y17 = X17.sortby(X17)

dsWi_taiesm1_rest = dsWi.sel(member1='TaiESM1-r1i1p1f1').drop_sel(member2='TaiESM1-r1i1p1f1')
taiesm1_rest_all = np.unique(dsWi_taiesm1_rest)
taiesm1_rest_all = taiesm1_rest_all[~np.isnan(taiesm1_rest_all)]

X18 = dsWi.sel(member1='MCM-UA-1-0-r1i1p1f2')
Y18 = X18.sortby(X18)

dsWi_mcm_rest = dsWi.sel(member1='MCM-UA-1-0-r1i1p1f2').drop_sel(member2='MCM-UA-1-0-r1i1p1f2')
mcm_rest_all = np.unique(dsWi_mcm_rest)
mcm_rest_all = mcm_rest_all[~np.isnan(mcm_rest_all)]

X19 = dsWi.sel(member1='KIOST-ESM-r1i1p1f1')
Y19 = X19.sortby(X19)

dsWi_kiost_rest = dsWi.sel(member1='KIOST-ESM-r1i1p1f1').drop_sel(member2='KIOST-ESM-r1i1p1f1')
kiost_rest_all = np.unique(dsWi_kiost_rest)
kiost_rest_all = kiost_rest_all[~np.isnan(kiost_rest_all)]

X20 = dsWi.sel(member1='FGOALS-f3-L-r1i1p1f1')
Y20 = X20.sortby(X20)

dsWi_fgoals_rest = dsWi.sel(member1='FGOALS-f3-L-r1i1p1f1').drop_sel(member2='FGOALS-f3-L-r1i1p1f1')
fgoals_rest_all = np.unique(dsWi_fgoals_rest)
fgoals_rest_all = fgoals_rest_all[~np.isnan(fgoals_rest_all)]

X21 = dsWi.sel(member1='E3SM-1-1-r1i1p1f1')
Y21 = X21.sortby(X21)

dsWi_e3sm_rest = dsWi.sel(member1='E3SM-1-1-r1i1p1f1').drop_sel(member2='E3SM-1-1-r1i1p1f1')
e3sm_rest_all = np.unique(dsWi_e3sm_rest)
e3sm_rest_all = e3sm_rest_all[~np.isnan(e3sm_rest_all)]

X22 = dsWi.sel(member1='CNRM-CM6-1-HR-r1i1p1f2')
Y22 = X22.sortby(X22)

dsWi_cnrm_hr_rest = dsWi.sel(member1='CNRM-CM6-1-HR-r1i1p1f2').drop_sel(member2='CNRM-CM6-1-HR-r1i1p1f2')
cnrm_hr_rest_all = np.unique(dsWi_cnrm_hr_rest)
cnrm_hr_rest_all = cnrm_hr_rest_all[~np.isnan(cnrm_hr_rest_all)]

X23 = dsWi.sel(member1='AWI-CM-1-1-MR-r1i1p1f1')
Y23 = X23.sortby(X23)

dsWi_awi_rest = dsWi.sel(member1='AWI-CM-1-1-MR-r1i1p1f1').drop_sel(member2='AWI-CM-1-1-MR-r1i1p1f1')
awi_rest_all = np.unique(dsWi_awi_rest)
awi_rest_all = awi_rest_all[~np.isnan(awi_rest_all)]

X24 = dsWi.sel(member1='CMCC-CM2-SR5-r1i1p1f1')
Y24 = X24.sortby(X24)

dsWi_cmcc_5_rest = dsWi.sel(member1='CMCC-CM2-SR5-r1i1p1f1').drop_sel(member2='CMCC-CM2-SR5-r1i1p1f1')
cmcc_5_rest_all = np.unique(dsWi_cmcc_5_rest)
cmcc_5_rest_all = cmcc_5_rest_all[~np.isnan(cmcc_5_rest_all)]

X25 = dsWi.sel(member1='CMCC-ESM2-r1i1p1f1')
Y25 = X25.sortby(X25)

dsWi_cmcc_rest = dsWi.sel(member1='CMCC-ESM2-r1i1p1f1').drop_sel(member2='CMCC-ESM2-r1i1p1f1')
cmcc_rest_all = np.unique(dsWi_cmcc_rest)
cmcc_rest_all = cmcc_rest_all[~np.isnan(cmcc_rest_all)]

X26 = dsWi.sel(member1='NorESM2-MM-r1i1p1f1')
Y26 = X26.sortby(X26)

dsWi_noresm_rest = dsWi.sel(member1='NorESM2-MM-r1i1p1f1').drop_sel(member2='NorESM2-MM-r1i1p1f1')
noresm_rest_all = np.unique(dsWi_noresm_rest)
noresm_rest_all = noresm_rest_all[~np.isnan(noresm_rest_all)]

X27 = dsWi.sel(member1='GFDL-CM4-r1i1p1f1')
Y27 = X27.sortby(X27)

dsWi_gfdl_rest = dsWi.sel(member1='GFDL-CM4-r1i1p1f1').drop_sel(member2='GFDL-CM4-r1i1p1f1')
gfdl_rest_all = np.unique(dsWi_gfdl_rest)
gfdl_rest_all = gfdl_rest_all[~np.isnan(gfdl_rest_all)]

X28 = dsWi.sel(member1='GFDL-ESM4-r1i1p1f1')
Y28 = X28.sortby(X28)

dsWi_gfdl_e_rest = dsWi.sel(member1='GFDL-ESM4-r1i1p1f1').drop_sel(member2='GFDL-ESM4-r1i1p1f1')
gfdl_e_rest_all = np.unique(dsWi_gfdl_e_rest)
gfdl_e_rest_all = gfdl_e_rest_all[~np.isnan(gfdl_e_rest_all)]

X30 = dsWi.sel(member1='INM-CM4-8-r1i1p1f1')
Y30 = X30.sortby(X30)

dsWi_inm_4_rest = dsWi.sel(member1='INM-CM4-8-r1i1p1f1').drop_sel(member2='INM-CM4-8-r1i1p1f1')
inm_4_rest_all = np.unique(dsWi_inm_4_rest)
inm_4_rest_all = inm_4_rest_all[~np.isnan(inm_4_rest_all)]

X31 = dsWi.sel(member1='INM-CM5-0-r1i1p1f1')
Y31 = X31.sortby(X31)

dsWi_inm_5_rest = dsWi.sel(member1='INM-CM5-0-r1i1p1f1').drop_sel(member2='INM-CM5-0-r1i1p1f1')
inm_5_rest_all = np.unique(dsWi_inm_5_rest)
inm_5_rest_all = inm_5_rest_all[~np.isnan(inm_5_rest_all)]

########################

fig = plt.figure(figsize=(8,14))
ax = plt.subplot(211)
plt.plot(access_fam_all,0*np.ones(np.size(access_fam_all)),'|',color='tab:red')
plt.plot(access_rest_all,0*np.ones(np.size(access_rest_all)),'|',color='silver')

plt.plot(had_mm_fam_all,1*np.ones(np.size(had_mm_fam_all)),'|',color='tab:red')
plt.plot(had_mm_rest_all,1*np.ones(np.size(had_mm_rest_all)),'|',color='silver')

plt.plot(kace_fam_all,2*np.ones(np.size(kace_fam_all)),'|',color='tab:red')
plt.plot(kace_rest_all,2*np.ones(np.size(kace_rest_all)),'|',color='silver')

plt.plot(access_2_fam_all,3*np.ones(np.size(access_2_fam_all)),'|',color='tab:red')
plt.plot(access_2_rest_all,3*np.ones(np.size(access_2_rest_all)),'|',color='silver')

plt.plot(had_ll_fam_all,4*np.ones(np.size(had_ll_fam_all)),'|',color='tab:red')
plt.plot(had_ll_rest_all,4*np.ones(np.size(had_ll_rest_all)),'|',color='silver')

plt.plot(ukesm1_fam_all,5*np.ones(np.size(ukesm1_fam_all)),'|',color='tab:red')
plt.plot(ukesm1_rest_all,5*np.ones(np.size(ukesm1_rest_all)),'|',color='silver')

plt.axhline(6,color='silver')
#############

plt.plot(taiesm1_rest_all,7*np.ones(np.size(taiesm1_rest_all)),'|',color='silver')

plt.plot(cmcc_rest_all,8*np.ones(np.size(cmcc_rest_all)),'|',color='silver')

plt.plot(cmcc_5_rest_all,9*np.ones(np.size(cmcc_5_rest_all)),'|',color='silver')

plt.plot(noresm_rest_all,10*np.ones(np.size(noresm_rest_all)),'|',color='silver')

plt.plot(cesm_waccm_fam_all,11*np.ones(np.size(cesm_waccm_fam_all)),'|',color='darkgoldenrod')
plt.plot(cesm_waccm_rest_all,11*np.ones(np.size(cesm_waccm_rest_all)),'|',color='silver')

plt.plot(cesm_fam_all,12*np.ones(np.size(cesm_fam_all)),'|',color='darkgoldenrod')
plt.plot(cesm_rest_all,12*np.ones(np.size(cesm_rest_all)),'|',color='silver')

plt.axhline(13,color='silver')
#############

plt.plot(cnrm_hr_rest_all,14*np.ones(np.size(cnrm_hr_rest_all)),'|',color='silver')

plt.plot(cnrm_esm_fam_all,15*np.ones(np.size(cnrm_esm_fam_all)),'|',color='cornflowerblue')
plt.plot(cnrm_esm_rest_all,15*np.ones(np.size(cnrm_esm_rest_all)),'|',color='silver')

plt.plot(ipsl_fam_all,16*np.ones(np.size(ipsl_fam_all)),'|',color='royalblue')
plt.plot(ipsl_rest_all,16*np.ones(np.size(ipsl_rest_all)),'|',color='silver')

plt.plot(cnrm_fam_all,17*np.ones(np.size(cnrm_fam_all)),'|',color='cornflowerblue')
plt.plot(cnrm_rest_all,17*np.ones(np.size(cnrm_rest_all)),'|',color='silver')

plt.axhline(18,color='silver')
################

plt.plot(awi_rest_all,19*np.ones(np.size(awi_rest_all)),'|',color='silver')

plt.plot(nesm3_fam_all,20*np.ones(np.size(nesm3_fam_all)),'|',color='tab:orange')
plt.plot(nesm3_rest_all,20*np.ones(np.size(nesm3_rest_all)),'|',color='silver')

plt.plot(mpi_fam_all,21*np.ones(np.size(mpi_fam_all)),'|',color='tab:orange')
plt.plot(mpi_rest_all,21*np.ones(np.size(mpi_rest_all)),'|',color='silver')

plt.plot(mpi_hr_fam_all,22*np.ones(np.size(mpi_hr_fam_all)),'|',color='tab:orange')
plt.plot(mpi_hr_rest_all,22*np.ones(np.size(mpi_hr_rest_all)),'|',color='silver')

plt.axhline(23,color='silver')
################

plt.plot(gfdl_rest_all,24*np.ones(np.size(gfdl_rest_all)),'|',color='silver')
plt.plot(gfdl_e_rest_all,25*np.ones(np.size(gfdl_e_rest_all)),'|',color='silver')

plt.axhline(26,color='silver')
################

plt.plot(earth_rest_all,27*np.ones(np.size(earth_rest_all)),'|',color='silver')
plt.plot(earth_fam_all,27*np.ones(np.size(earth_fam_all)),'|',color='darkgreen')

plt.plot(earth_veg_rest_all,28*np.ones(np.size(earth_veg_rest_all)),'|',color='silver')
plt.plot(earth_veg_fam_all,28*np.ones(np.size(earth_veg_fam_all)),'|',color='darkgreen')

plt.axhline(29,color='silver')
################

plt.plot(fgoals_rest_all,30*np.ones(np.size(fgoals_rest_all)),'|',color='silver')

plt.plot(fgoals_g_fam_all,31*np.ones(np.size(fgoals_g_fam_all)),'|',color='maroon')
plt.plot(fgoals_g_rest_all,31*np.ones(np.size(fgoals_g_rest_all)),'|',color='silver')

plt.axhline(32,color='silver')
################

plt.plot(inm_4_rest_all,33*np.ones(np.size(inm_4_rest_all)),'|',color='silver')
plt.plot(inm_5_rest_all,34*np.ones(np.size(inm_5_rest_all)),'|',color='silver')

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
labels = labels = ['1) ACCESS-ESM1-5','2) HadGEM3-GC31-MM','3) KACE-1-0-G','4) ACCESS-CM2','5) HadGEM3-GC31-LL','6) UKESM1-0-LL','',
'7) TaiESM1','8) CMCC-ESM2','9) CMCC-CM2-SR5','10) NorESM2-MM','11) CESM2-WACCM','12) CESM2','',
'13) CNRM-CM6-1-HR','14) CNRM-ESM2-1','15) IPSL-CM6A-LR','16) CNRM-CM6-1','',
'17) AWI-CM-1-1-MR','18) NESM3','19) MPI-ESM1-2-LR','20) MPI-ESM1-2-HR','',
'21) GFDL-CM4','22) GFDL-ESM4','','23) EC-Earth3','24) EC-Earth3-Veg','',
'25) FGOALS-f3-L','26) FGOALS-g3','','27) INM-CM4-8','28) INM-CM5-0','','29) MIROC6','30) MIROC-ES2L','','31) MRI-ESM2-0',''
'32) E3SM-1-1','33) CanESM5','34) CAS-ESM2-0','35) GISS-E2-1-G','36) MCM-UA-1-0','37) KIOST-ESM']
xlabels = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2]
ax.set_yticklabels(labels,fontsize=9,rotation = 0)
ax.set_xticklabels('',fontsize=9)
ax.set_axisbelow(True)
ax.invert_yaxis()
plt.title('a) CMIP6 Global SAT & SLP (1905-2005)',fontsize=12,fontweight='bold',loc='left')


##################################

dirT6 = '/**/CMIP_subselection/Data/'
dsT6 = xr.open_dataset(dirT6 + 'tas_mon_CMIP6_SSP585_g025_v2_ann.nc',use_cftime = True)
dsT6 = dsT6-273.15
dsT6 = dsT6.sortby(dsT6.member)

dsP6 = xr.open_dataset(dirT6 + 'psl_mon_CMIP6_SSP585_g025_v2_ann.nc',use_cftime = True)
dsP6 = dsP6/100
dsP6 = dsP6.sortby(dsP6.member)

dsT6_clim = dsT6.sel(year=slice(1980, 2014)).mean('year')
dsP6_clim = dsP6.sel(year=slice(1980, 2014)).mean('year')

dsT6_clim = dsT6_clim.drop_sel(member=['NorESM2-LM-r1i1p1f1','FIO-ESM-2-0-r1i1p1f1','FIO-ESM-2-0-r2i1p1f1', 'FIO-ESM-2-0-r3i1p1f1'])#,'MCM-UA-1-0-r1i1p1f2'])
dsP6_clim = dsP6_clim.drop_sel(member=['BCC-CSM2-MR-r1i1p1f1','CIESM-r1i1p1f1','FIO-ESM-2-0-r1i1p1f1','FIO-ESM-2-0-r2i1p1f1', 'FIO-ESM-2-0-r3i1p1f1','EC-Earth3-Veg-r6i1p1f1'])#,'MCM-UA-1-0-r1i1p1f2'])

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

dsT6_clim_test = get_error(dsT6_clim.tas,weights)
dsT6_clim_test = dsT6_clim_test.where(dsT6_clim_test!=0)
dsT6_clim_test_norm = dsT6_clim_test/np.nanmean(dsT6_clim_test)

dsP6_clim_test = get_error(dsP6_clim.psl,weights)
dsP6_clim_test = dsP6_clim_test.where(dsP6_clim_test!=0)
dsP6_clim_test_norm = dsP6_clim_test/np.nanmean(dsP6_clim_test)

dsWi = (dsT6_clim_test_norm + dsP6_clim_test_norm)/2 ###############

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


######################################
nesm3_fam = ['NESM3-r1i1p1f1','NESM3-r2i1p1f1']

dsWi_nesm3_fam = dsWi.sel(member1=nesm3_fam,member2=nesm3_fam)
nesm3_fam_all = np.unique(dsWi_nesm3_fam)
nesm3_fam_all = nesm3_fam_all[~np.isnan(nesm3_fam_all)]

dsWi_nesm3_rest = dsWi.sel(member1=nesm3_fam).drop_sel(member2=nesm3_fam)
nesm3_rest_all = np.unique(dsWi_nesm3_rest)
nesm3_rest_all = nesm3_rest_all[~np.isnan(nesm3_rest_all)]

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

had_ll_fam = ['HadGEM3-GC31-LL-r1i1p1f3',
'HadGEM3-GC31-LL-r2i1p1f3', 'HadGEM3-GC31-LL-r3i1p1f3',
'HadGEM3-GC31-LL-r4i1p1f3']

dsWi_had_ll_fam = dsWi.sel(member1=had_ll_fam,member2=had_ll_fam)
had_ll_fam_all = np.unique(dsWi_had_ll_fam)
had_ll_fam_all = had_ll_fam_all[~np.isnan(had_ll_fam_all)]

dsWi_had_ll_rest = dsWi.sel(member1=had_ll_fam).drop_sel(member2=had_ll_fam)
had_ll_rest_all = np.unique(dsWi_had_ll_rest)
had_ll_rest_all = had_ll_rest_all[~np.isnan(had_ll_rest_all)]

had_mm_fam = ['HadGEM3-GC31-MM-r1i1p1f3',
'HadGEM3-GC31-MM-r2i1p1f3', 'HadGEM3-GC31-MM-r3i1p1f3',
'HadGEM3-GC31-MM-r4i1p1f3']

dsWi_had_mm_fam = dsWi.sel(member1=had_mm_fam,member2=had_mm_fam)
had_mm_fam_all = np.unique(dsWi_had_mm_fam)
had_mm_fam_all = had_mm_fam_all[~np.isnan(had_mm_fam_all)]

dsWi_had_mm_rest = dsWi.sel(member1=had_mm_fam).drop_sel(member2=had_mm_fam)
had_mm_rest_all = np.unique(dsWi_had_mm_rest)
had_mm_rest_all = had_mm_rest_all[~np.isnan(had_mm_rest_all)]

kace_fam = ['KACE-1-0-G-r1i1p1f1','KACE-1-0-G-r2i1p1f1', 'KACE-1-0-G-r3i1p1f1']

dsWi_kace_fam = dsWi.sel(member1=kace_fam,member2=kace_fam)
kace_fam_all = np.unique(dsWi_kace_fam)
kace_fam_all = kace_fam_all[~np.isnan(kace_fam_all)]

dsWi_kace_rest = dsWi.sel(member1=kace_fam).drop_sel(member2=kace_fam)
kace_rest_all = np.unique(dsWi_kace_rest)
kace_rest_all = kace_rest_all[~np.isnan(kace_rest_all)]

ukesm1_fam = ['UKESM1-0-LL-r1i1p1f2', 'UKESM1-0-LL-r2i1p1f2',
'UKESM1-0-LL-r3i1p1f2', 'UKESM1-0-LL-r4i1p1f2','UKESM1-0-LL-r8i1p1f2']

dsWi_ukesm1_fam = dsWi.sel(member1=ukesm1_fam,member2=ukesm1_fam)
ukesm1_fam_all = np.unique(dsWi_ukesm1_fam)
ukesm1_fam_all = ukesm1_fam_all[~np.isnan(ukesm1_fam_all)]

dsWi_ukesm1_rest = dsWi.sel(member1=ukesm1_fam).drop_sel(member2=ukesm1_fam)
ukesm1_rest_all = np.unique(dsWi_ukesm1_rest)
ukesm1_rest_all = ukesm1_rest_all[~np.isnan(ukesm1_rest_all)]


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


cesm_fam = ['CESM2-r10i1p1f1', 'CESM2-r11i1p1f1',
'CESM2-r1i1p1f1', 'CESM2-r2i1p1f1', 'CESM2-r4i1p1f1']

# closest relative: 'AWI-CM-1-1-MR-r1i1p1f1'

dsWi_cesm_fam = dsWi.sel(member1=cesm_fam,member2=cesm_fam)
cesm_fam_all = np.unique(dsWi_cesm_fam)
cesm_fam_all = cesm_fam_all[~np.isnan(cesm_fam_all)]

dsWi_cesm_rest = dsWi.sel(member1=cesm_fam).drop_sel(member2=cesm_fam)
cesm_rest_all = np.unique(dsWi_cesm_rest)
cesm_rest_all = cesm_rest_all[~np.isnan(cesm_rest_all)]

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


#####################################

ec_earth_veg_fam = ['EC-Earth3-Veg-r1i1p1f1', 'EC-Earth3-Veg-r2i1p1f1',
'EC-Earth3-Veg-r3i1p1f1', 'EC-Earth3-Veg-r4i1p1f1']

dsWi_earth_veg_fam = dsWi.sel(member1=ec_earth_veg_fam,member2=ec_earth_veg_fam)
earth_veg_fam_all = np.unique(dsWi_earth_veg_fam)
earth_veg_fam_all = earth_veg_fam_all[~np.isnan(earth_veg_fam_all)]

dsWi_earth_veg_rest = dsWi.sel(member1=ec_earth_veg_fam).drop_sel(member2=ec_earth_veg_fam)
earth_veg_rest_all = np.unique(dsWi_earth_veg_rest)
earth_veg_rest_all = earth_veg_rest_all[~np.isnan(earth_veg_rest_all)]

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

X17 = dsWi.sel(member1='TaiESM1-r1i1p1f1')
Y17 = X17.sortby(X17)

dsWi_taiesm1_rest = dsWi.sel(member1='TaiESM1-r1i1p1f1').drop_sel(member2='TaiESM1-r1i1p1f1')
taiesm1_rest_all = np.unique(dsWi_taiesm1_rest)
taiesm1_rest_all = taiesm1_rest_all[~np.isnan(taiesm1_rest_all)]

X18 = dsWi.sel(member1='MCM-UA-1-0-r1i1p1f2')
Y18 = X18.sortby(X18)

dsWi_mcm_rest = dsWi.sel(member1='MCM-UA-1-0-r1i1p1f2').drop_sel(member2='MCM-UA-1-0-r1i1p1f2')
mcm_rest_all = np.unique(dsWi_mcm_rest)
mcm_rest_all = mcm_rest_all[~np.isnan(mcm_rest_all)]

X19 = dsWi.sel(member1='KIOST-ESM-r1i1p1f1')
Y19 = X19.sortby(X19)

dsWi_kiost_rest = dsWi.sel(member1='KIOST-ESM-r1i1p1f1').drop_sel(member2='KIOST-ESM-r1i1p1f1')
kiost_rest_all = np.unique(dsWi_kiost_rest)
kiost_rest_all = kiost_rest_all[~np.isnan(kiost_rest_all)]

X20 = dsWi.sel(member1='FGOALS-f3-L-r1i1p1f1')
Y20 = X20.sortby(X20)

dsWi_fgoals_rest = dsWi.sel(member1='FGOALS-f3-L-r1i1p1f1').drop_sel(member2='FGOALS-f3-L-r1i1p1f1')
fgoals_rest_all = np.unique(dsWi_fgoals_rest)
fgoals_rest_all = fgoals_rest_all[~np.isnan(fgoals_rest_all)]

X21 = dsWi.sel(member1='E3SM-1-1-r1i1p1f1')
Y21 = X21.sortby(X21)

dsWi_e3sm_rest = dsWi.sel(member1='E3SM-1-1-r1i1p1f1').drop_sel(member2='E3SM-1-1-r1i1p1f1')
e3sm_rest_all = np.unique(dsWi_e3sm_rest)
e3sm_rest_all = e3sm_rest_all[~np.isnan(e3sm_rest_all)]

X22 = dsWi.sel(member1='CNRM-CM6-1-HR-r1i1p1f2')
Y22 = X22.sortby(X22)

dsWi_cnrm_hr_rest = dsWi.sel(member1='CNRM-CM6-1-HR-r1i1p1f2').drop_sel(member2='CNRM-CM6-1-HR-r1i1p1f2')
cnrm_hr_rest_all = np.unique(dsWi_cnrm_hr_rest)
cnrm_hr_rest_all = cnrm_hr_rest_all[~np.isnan(cnrm_hr_rest_all)]


X23 = dsWi.sel(member1='AWI-CM-1-1-MR-r1i1p1f1')
Y23 = X23.sortby(X23)

dsWi_awi_rest = dsWi.sel(member1='AWI-CM-1-1-MR-r1i1p1f1').drop_sel(member2='AWI-CM-1-1-MR-r1i1p1f1')
awi_rest_all = np.unique(dsWi_awi_rest)
awi_rest_all = awi_rest_all[~np.isnan(awi_rest_all)]


X24 = dsWi.sel(member1='CMCC-CM2-SR5-r1i1p1f1')
Y24 = X24.sortby(X24)

dsWi_cmcc_5_rest = dsWi.sel(member1='CMCC-CM2-SR5-r1i1p1f1').drop_sel(member2='CMCC-CM2-SR5-r1i1p1f1')
cmcc_5_rest_all = np.unique(dsWi_cmcc_5_rest)
cmcc_5_rest_all = cmcc_5_rest_all[~np.isnan(cmcc_5_rest_all)]

X25 = dsWi.sel(member1='CMCC-ESM2-r1i1p1f1')
Y25 = X25.sortby(X25)

dsWi_cmcc_rest = dsWi.sel(member1='CMCC-ESM2-r1i1p1f1').drop_sel(member2='CMCC-ESM2-r1i1p1f1')
cmcc_rest_all = np.unique(dsWi_cmcc_rest)
cmcc_rest_all = cmcc_rest_all[~np.isnan(cmcc_rest_all)]

X26 = dsWi.sel(member1='NorESM2-MM-r1i1p1f1')
Y26 = X26.sortby(X26)

dsWi_noresm_rest = dsWi.sel(member1='NorESM2-MM-r1i1p1f1').drop_sel(member2='NorESM2-MM-r1i1p1f1')
noresm_rest_all = np.unique(dsWi_noresm_rest)
noresm_rest_all = noresm_rest_all[~np.isnan(noresm_rest_all)]

X27 = dsWi.sel(member1='GFDL-CM4-r1i1p1f1')
Y27 = X27.sortby(X27)

dsWi_gfdl_rest = dsWi.sel(member1='GFDL-CM4-r1i1p1f1').drop_sel(member2='GFDL-CM4-r1i1p1f1')
gfdl_rest_all = np.unique(dsWi_gfdl_rest)
gfdl_rest_all = gfdl_rest_all[~np.isnan(gfdl_rest_all)]

X28 = dsWi.sel(member1='GFDL-ESM4-r1i1p1f1')
Y28 = X28.sortby(X28)

dsWi_gfdl_e_rest = dsWi.sel(member1='GFDL-ESM4-r1i1p1f1').drop_sel(member2='GFDL-ESM4-r1i1p1f1')
gfdl_e_rest_all = np.unique(dsWi_gfdl_e_rest)
gfdl_e_rest_all = gfdl_e_rest_all[~np.isnan(gfdl_e_rest_all)]

X30 = dsWi.sel(member1='INM-CM4-8-r1i1p1f1')
Y30 = X30.sortby(X30)

dsWi_inm_4_rest = dsWi.sel(member1='INM-CM4-8-r1i1p1f1').drop_sel(member2='INM-CM4-8-r1i1p1f1')
inm_4_rest_all = np.unique(dsWi_inm_4_rest)
inm_4_rest_all = inm_4_rest_all[~np.isnan(inm_4_rest_all)]

X31 = dsWi.sel(member1='INM-CM5-0-r1i1p1f1')
Y31 = X31.sortby(X31)

dsWi_inm_5_rest = dsWi.sel(member1='INM-CM5-0-r1i1p1f1').drop_sel(member2='INM-CM5-0-r1i1p1f1')
inm_5_rest_all = np.unique(dsWi_inm_5_rest)
inm_5_rest_all = inm_5_rest_all[~np.isnan(inm_5_rest_all)]

########################

ax = plt.subplot(212)
plt.plot(access_fam_all,0*np.ones(np.size(access_fam_all)),'|',color='tab:red')
plt.plot(access_rest_all,0*np.ones(np.size(access_rest_all)),'|',color='silver')

plt.plot(had_mm_fam_all,1*np.ones(np.size(had_mm_fam_all)),'|',color='tab:red')
plt.plot(had_mm_rest_all,1*np.ones(np.size(had_mm_rest_all)),'|',color='silver')

plt.plot(kace_fam_all,2*np.ones(np.size(kace_fam_all)),'|',color='tab:red')
plt.plot(kace_rest_all,2*np.ones(np.size(kace_rest_all)),'|',color='silver')

plt.plot(access_2_fam_all,3*np.ones(np.size(access_2_fam_all)),'|',color='tab:red')
plt.plot(access_2_rest_all,3*np.ones(np.size(access_2_rest_all)),'|',color='silver')

plt.plot(had_ll_fam_all,4*np.ones(np.size(had_ll_fam_all)),'|',color='tab:red')
plt.plot(had_ll_rest_all,4*np.ones(np.size(had_ll_rest_all)),'|',color='silver')

plt.plot(ukesm1_fam_all,5*np.ones(np.size(ukesm1_fam_all)),'|',color='tab:red')
plt.plot(ukesm1_rest_all,5*np.ones(np.size(ukesm1_rest_all)),'|',color='silver')

plt.axhline(6,color='silver')
#############

plt.plot(taiesm1_rest_all,7*np.ones(np.size(taiesm1_rest_all)),'|',color='silver')

plt.plot(cmcc_rest_all,8*np.ones(np.size(cmcc_rest_all)),'|',color='silver')

plt.plot(cmcc_5_rest_all,9*np.ones(np.size(cmcc_5_rest_all)),'|',color='silver')

plt.plot(noresm_rest_all,10*np.ones(np.size(noresm_rest_all)),'|',color='silver')

plt.plot(cesm_waccm_fam_all,11*np.ones(np.size(cesm_waccm_fam_all)),'|',color='darkgoldenrod')
plt.plot(cesm_waccm_rest_all,11*np.ones(np.size(cesm_waccm_rest_all)),'|',color='silver')

plt.plot(cesm_fam_all,12*np.ones(np.size(cesm_fam_all)),'|',color='darkgoldenrod')
plt.plot(cesm_rest_all,12*np.ones(np.size(cesm_rest_all)),'|',color='silver')

plt.axhline(13,color='silver')
#############

plt.plot(cnrm_hr_rest_all,14*np.ones(np.size(cnrm_hr_rest_all)),'|',color='silver')

plt.plot(cnrm_esm_fam_all,15*np.ones(np.size(cnrm_esm_fam_all)),'|',color='cornflowerblue')
plt.plot(cnrm_esm_rest_all,15*np.ones(np.size(cnrm_esm_rest_all)),'|',color='silver')

plt.plot(ipsl_fam_all,16*np.ones(np.size(ipsl_fam_all)),'|',color='royalblue')
plt.plot(ipsl_rest_all,16*np.ones(np.size(ipsl_rest_all)),'|',color='silver')

plt.plot(cnrm_fam_all,17*np.ones(np.size(cnrm_fam_all)),'|',color='cornflowerblue')
plt.plot(cnrm_rest_all,17*np.ones(np.size(cnrm_rest_all)),'|',color='silver')

plt.axhline(18,color='silver')
################

plt.plot(awi_rest_all,19*np.ones(np.size(awi_rest_all)),'|',color='silver')

plt.plot(nesm3_fam_all,20*np.ones(np.size(nesm3_fam_all)),'|',color='tab:orange')
plt.plot(nesm3_rest_all,20*np.ones(np.size(nesm3_rest_all)),'|',color='silver')

plt.plot(mpi_fam_all,21*np.ones(np.size(mpi_fam_all)),'|',color='tab:orange')
plt.plot(mpi_rest_all,21*np.ones(np.size(mpi_rest_all)),'|',color='silver')

plt.plot(mpi_hr_fam_all,22*np.ones(np.size(mpi_hr_fam_all)),'|',color='tab:orange')
plt.plot(mpi_hr_rest_all,22*np.ones(np.size(mpi_hr_rest_all)),'|',color='silver')

plt.axhline(23,color='silver')
################

plt.plot(gfdl_rest_all,24*np.ones(np.size(gfdl_rest_all)),'|',color='silver')
plt.plot(gfdl_e_rest_all,25*np.ones(np.size(gfdl_e_rest_all)),'|',color='silver')

plt.axhline(26,color='silver')
################

plt.plot(earth_rest_all,27*np.ones(np.size(earth_rest_all)),'|',color='silver')
plt.plot(earth_fam_all,27*np.ones(np.size(earth_fam_all)),'|',color='darkgreen')

plt.plot(earth_veg_rest_all,28*np.ones(np.size(earth_veg_rest_all)),'|',color='silver')
plt.plot(earth_veg_fam_all,28*np.ones(np.size(earth_veg_fam_all)),'|',color='darkgreen')

plt.axhline(29,color='silver')
################

plt.plot(fgoals_rest_all,30*np.ones(np.size(fgoals_rest_all)),'|',color='silver')

plt.plot(fgoals_g_fam_all,31*np.ones(np.size(fgoals_g_fam_all)),'|',color='maroon')
plt.plot(fgoals_g_rest_all,31*np.ones(np.size(fgoals_g_rest_all)),'|',color='silver')

plt.axhline(32,color='silver')
################

plt.plot(inm_4_rest_all,33*np.ones(np.size(inm_4_rest_all)),'|',color='silver')
plt.plot(inm_5_rest_all,34*np.ones(np.size(inm_5_rest_all)),'|',color='silver')

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
plt.xlabel('intermember distance',fontsize=13)
ax.set_axisbelow(True)
ax.invert_yaxis()
plt.subplots_adjust(wspace=0, hspace=0.1)
plt.title('b) CMIP6 Global SAT & SLP (1980-2014)',fontsize=12,fontweight='bold',loc='left')#,pad=14)
plt.savefig('Fig1_intermember_CMIP6_compare_time_averaging.png',bbox_inches='tight',dpi=300)
