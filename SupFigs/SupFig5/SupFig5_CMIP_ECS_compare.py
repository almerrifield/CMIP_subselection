# Supplementary Figure 5: makes ECS supplement figure (discrepency between reported values)

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

# ECS likely range: 1.5–4.5 K, with a central estimate of about 3 K, 3.0 [2.0–5.0]°C
# TCR very likely range: 1.8 [1.2 – 2.4]°C

#######################################################
# Meehl et al. 2020: The ECS calculated by the Gregory method is derived from a fully coupled Earth system model and does
# not require equilibrium to actually be achieved. In the Gregory method, CO2 is instantaneously quadrupled in a fully coupled Earth
# system model and run for 150 years. As the surface temperature asymptotes toward equilibrium, the slope of the time-evolving
# curve of the net top-of-atmosphere radiance against the surface temperature is calculated to extrapolate the eventual temperature
# increase at equilibrium some time far in the future for a doubling of CO2, assuming that there is a roughly linear response that is
# half of the warming from a quadrupling of CO2

CMIP5_ECS_Meehl = {
'ACCESS1-0':3.8,
'ACCESS1-3':3.5,
'HadGEM2-ES':4.6,
'NorESM1-ME':None,
'NorESM1-M':2.8,
'CCSM4':2.9,
'CESM1-CAM5':None,
'IPSL-CM5B-LR':2.6,
'IPSL-CM5A-MR':None,
'IPSL-CM5A-LR':4.1,
"EC-EARTH":None,
'CNRM-CM5':3.3,
'MPI-ESM-MR':3.5,
'MPI-ESM-LR':3.6,
'GFDL-ESM2G':2.4,
'GFDL-ESM2M':2.4,
'GFDL-CM3':4.0,
'MIROC5':2.7,
'MIROC-ESM':4.7,
'GISS-E2-H':2.3,
'GISS-E2-R':2.1,
'bcc-csm1-1':2.8,
'bcc-csm1-1-m':2.9,
"BNU-ESM":3.9,
'inmcm4':2.1,
'CanESM2':3.7,
'MRI-CGCM3':2.6,
'CSIRO-Mk3-6-0':4.1,
"FGOALS-g2":3.4}

# CONVERT DICTIONARY TO XARRAY
coords, values = zip(* list(CMIP5_ECS_Meehl.items()))
ds_ECS_Meehl = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# missing: 'CESM1-BGC', 'CESM1-CAM5', FIO-ESM','GISS-E2-H-CC',
# 'GISS-E2-R-CC','HadGEM2-AO','HadGEM2-CC','IPSL-CM5A-MR','MIROC-ESM-CHEM',
# 'MRI-ESM1',

# Seland et al. 2020: in general ECS is more commonly estimated from the relationship 380 between surface temperature and RESTOM from the abrupt4×CO2 experiment using the so-called Gregory
# method (Gregory et al., 2004). The numbers in table 1 are calculated using years 1–150 from the simulations shown in Fig. 3, and are divided by 2 to get the number for CO2 doubling instead of quadrupling.

CMIP5_ECS_Seland = {
'ACCESS1-0':None,
'ACCESS1-3':None,
'HadGEM2-ES':None,
'NorESM1-ME':2.99,
'NorESM1-M':2.86,
'CCSM4':None,
'CESM1-CAM5':None,
'IPSL-CM5B-LR':None,
'IPSL-CM5A-MR':None,
'IPSL-CM5A-LR':None,
"EC-EARTH":None,
'CNRM-CM5':None,
'MPI-ESM-MR':None,
'MPI-ESM-LR':None,
'GFDL-ESM2G':None,
'GFDL-ESM2M':None,
'GFDL-CM3':None,
'MIROC5':None,
'MIROC-ESM':None,
'GISS-E2-H':None,
'GISS-E2-R':None,
'bcc-csm1-1':None,
'bcc-csm1-1-m':None,
"BNU-ESM":None,
'inmcm4':None,
'CanESM2':None,
'MRI-CGCM3':None,
'CSIRO-Mk3-6-0':None,
"FGOALS-g2":None}

coords, values = zip(* list(CMIP5_ECS_Seland.items()))
ds_ECS_Seland = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))


# Femke et al. 2020 - "Mean values are reported for models with multiple realisations. The values of F2 , ECS and are computed using the Gregory method (Gregory, 2004)."
# CMIP5_ECS_Femke = {'ACCESS1-0':3.90,'ACCESS1-3':3.63,"BNU-ESM":4.07,'CCSM4':None,'CESM1-CAM5':None,
# 'CNRM-CM5':3.28,'CSIRO-Mk3-6-0':4.36,'CanESM2':3.71,"EC-EARTH":None,"FGOALS-g2":None,'GFDL-CM3':4.03,'GFDL-ESM2G':2.34,
# 'GFDL-ESM2M':2.46,'GISS-E2-H':2.43,'GISS-E2-R':2.28,'HadGEM2-ES':4.64,'IPSL-CM5A-LR':4.05,'IPSL-CM5A-MR':None,
# 'IPSL-CM5B-LR':2.64,'MIROC-ESM':4.75,'MIROC5':2.70,'MPI-ESM-LR':3.66,'MPI-ESM-MR':3.51,'MRI-CGCM3':2.61,
# 'NorESM1-ME':None,'NorESM1-M':2.93,'bcc-csm1-1-m':2.91,'bcc-csm1-1':2.91,'inmcm4':2.05}

CMIP5_ECS_Femke = {
'ACCESS1-0':3.90,
'ACCESS1-3':3.63,
'HadGEM2-ES':4.64,
'NorESM1-ME':None,
'NorESM1-M':2.93,
'CCSM4':None,
'CESM1-CAM5':None,
'IPSL-CM5B-LR':2.64,
'IPSL-CM5A-MR':None,
'IPSL-CM5A-LR':4.05,
"EC-EARTH":None,
'CNRM-CM5':3.28,
'MPI-ESM-MR':3.51,
'MPI-ESM-LR':3.66,
'GFDL-ESM2G':2.34,
'GFDL-ESM2M':2.46,
'GFDL-CM3':4.03,
'MIROC5':2.70,
'MIROC-ESM':4.75,
'GISS-E2-H':2.43,
'GISS-E2-R':2.28,
'bcc-csm1-1':2.91,
'bcc-csm1-1-m':2.91,
"BNU-ESM":4.07,
'inmcm4':2.05,
'CanESM2':3.71,
'MRI-CGCM3':2.61,
'CSIRO-Mk3-6-0':4.36,
"FGOALS-g2":None}

# missing: 'CCSM4','CESM1-BGC', 'CESM1-CAM5', FIO-ESM','GISS-E2-H-CC',
# 'GISS-E2-R-CC','HadGEM2-AO','HadGEM2-CC','IPSL-CM5A-MR','MIROC-ESM-CHEM',
# 'MRI-ESM1','NorESM1-ME'

coords, values = zip(* list(CMIP5_ECS_Femke.items()))
ds_ECS_Femke = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))


# Flynn et al. 2020: These resulting anomalies are linearly regressed against each other, following the Gregory
# method (Gregory et al., 2004), to obtain the ECS value as one-half of the x-intercept
# CMIP5_ECS_Flynn = {'ACCESS1-0':3.76,'ACCESS1-3':None,"BNU-ESM":3.98,'CCSM4':2.90,'CESM1-CAM5':None,
# 'CNRM-CM5':3.21,'CSIRO-Mk3-6-0':4.05,'CanESM2':3.71,"EC-EARTH":None,"FGOALS-g2":3.39,'GFDL-CM3':3.85,'GFDL-ESM2G':2.30,
# 'GFDL-ESM2M':2.33,'GISS-E2-H':2.33,'GISS-E2-R':2.06,'HadGEM2-ES':3.96,'IPSL-CM5A-LR':3.97,'IPSL-CM5A-MR':4.03,
# 'IPSL-CM5B-LR':2.58,'MIROC-ESM':4.68,'MIROC5':2.70,'MPI-ESM-LR':3.48,'MPI-ESM-MR':3.31,'MRI-CGCM3':2.65,
# 'NorESM1-ME':None,'NorESM1-M':2.75,'bcc-csm1-1-m':2.77,'bcc-csm1-1':2.81,'inmcm4':2.01}

CMIP5_ECS_Flynn = {
'ACCESS1-0':3.76,
'ACCESS1-3':None,
'HadGEM2-ES':3.96,
'NorESM1-ME':None,
'NorESM1-M':2.75,
'CCSM4':2.90,
'CESM1-CAM5':None,
'IPSL-CM5B-LR':2.58,
'IPSL-CM5A-MR':4.03,
'IPSL-CM5A-LR':3.97,
"EC-EARTH":None,
'CNRM-CM5':3.21,
'MPI-ESM-MR':3.31,
'MPI-ESM-LR':3.48,
'GFDL-ESM2G':2.30,
'GFDL-ESM2M':2.33,
'GFDL-CM3':3.85,
'MIROC5':2.70,
'MIROC-ESM':4.68,
'GISS-E2-H':2.33,
'GISS-E2-R':2.06,
'bcc-csm1-1':2.81,
'bcc-csm1-1-m':2.77,
"BNU-ESM":3.98,
'inmcm4':2.01,
'CanESM2':3.71,
'MRI-CGCM3':2.65,
'CSIRO-Mk3-6-0':4.05,
"FGOALS-g2":3.39}

# missing: 'ACCESS1-3','CESM1-BGC','CESM1-CAM5', FIO-ESM','GISS-E2-H-CC',
# 'GISS-E2-R-CC','HadGEM2-AO','HadGEM2-CC','IPSL-CM5A-MR','MIROC-ESM-CHEM',
# 'MRI-ESM1','NorESM1-ME'

coords, values = zip(* list(CMIP5_ECS_Flynn.items()))
ds_ECS_Flynn = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))


# Schlund et al. 2020: ECS is typically approximated by a so-called “effective climate sensitivity”,
# which is derived from the first 150 years that 85 follow an instantaneous quadrupling of the atmospheric
# CO2 concentration (4xCO2 run). Since the ESMs are not in radiative equilibrium during these 150 years,
# a regression of the top-of-atmosphere net downward radiation 𝑁 versus the global mean surface air temperature
# change Δ𝑇 extrapolated to 𝑁=0 gives an estimate of the equilibrium warming (Gregory et al., 2004).
# In this paper, we use the term “ECS” to denote this effective climate sensitivity derived from the Gregory
# regression method.

# CMIP5_ECS_Schlund = {'ACCESS1-0':3.83,'ACCESS1-3':3.53,"BNU-ESM":3.92,'CCSM4':2.94,'CESM1-CAM5':None,
# 'CNRM-CM5':3.25,'CSIRO-Mk3-6-0':4.08,'CanESM2':3.69,"EC-EARTH":None,"FGOALS-g2":3.38,'GFDL-CM3':3.97,'GFDL-ESM2G':2.39,
# 'GFDL-ESM2M':2.44,'GISS-E2-H':2.31,'GISS-E2-R':2.11,'HadGEM2-ES':4.61,'IPSL-CM5A-LR':4.13,'IPSL-CM5A-MR':4.12,
# 'IPSL-CM5B-LR':2.60,'MIROC-ESM':4.67,'MIROC5':2.72,'MPI-ESM-LR':3.63,'MPI-ESM-MR':3.46,'MRI-CGCM3':2.60,
# 'NorESM1-ME':None,'NorESM1-M':2.80,'bcc-csm1-1-m':2.86,'bcc-csm1-1':2.83,'inmcm4':2.08}

CMIP5_ECS_Schlund = {
'ACCESS1-0':3.83,
'ACCESS1-3':3.53,
'HadGEM2-ES':4.61,
'NorESM1-ME':None,
'NorESM1-M':2.80,
'CCSM4':2.94,
'CESM1-CAM5':None,
'IPSL-CM5B-LR':2.60,
'IPSL-CM5A-MR':4.12,
'IPSL-CM5A-LR':4.13,
"EC-EARTH":None,
'CNRM-CM5':3.25,
'MPI-ESM-MR':3.46,
'MPI-ESM-LR':3.63,
'GFDL-ESM2G':2.39,
'GFDL-ESM2M':2.44,
'GFDL-CM3':3.97,
'MIROC5':2.72,
'MIROC-ESM':4.67,
'GISS-E2-H':2.31,
'GISS-E2-R':2.11,
'bcc-csm1-1':2.83,
'bcc-csm1-1-m':2.86,
"BNU-ESM":3.92,
'inmcm4':2.08,
'CanESM2':3.69,
'MRI-CGCM3':2.60,
'CSIRO-Mk3-6-0':4.08,
"FGOALS-g2":3.38}

# missing: 'CESM1-BGC','CESM1-CAM5', FIO-ESM','GISS-E2-H-CC',
# 'GISS-E2-R-CC','HadGEM2-AO','HadGEM2-CC','MIROC-ESM-CHEM',
# 'MRI-ESM1','NorESM1-ME'

coords, values = zip(* list(CMIP5_ECS_Schlund.items()))
ds_ECS_Schlund = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# Bacmeister et al. 2020: The iECS approach was applied to 150-yr 4xCO2 AOGCM/ESM simulations to derive the published ECS values for CMIP5 (Flato et al., 2014).
# CMIP5_ECS_Bacmeister = {'ACCESS1-0':None,'ACCESS1-3':None,"BNU-ESM":None,'CCSM4':None,'CESM1-CAM5':3.4,
# 'CNRM-CM5':None,'CSIRO-Mk3-6-0':None,'CanESM2':None,"EC-EARTH":None,"FGOALS-g2":None,'GFDL-CM3':None,'GFDL-ESM2G':None,
# 'GFDL-ESM2M':None,'GISS-E2-H':None,'GISS-E2-R':None,'HadGEM2-ES':None,'IPSL-CM5A-LR':None,'IPSL-CM5A-MR':None,
# 'IPSL-CM5B-LR':None,'MIROC-ESM':None,'MIROC5':None,'MPI-ESM-LR':None,'MPI-ESM-MR':None,'MRI-CGCM3':None,
# 'NorESM1-ME':None,'NorESM1-M':None,'bcc-csm1-1-m':None,'bcc-csm1-1':None,'inmcm4':None}

CMIP5_ECS_Bacmeister = {
'ACCESS1-0':None,
'ACCESS1-3':None,
'HadGEM2-ES':None,
'NorESM1-ME':None,
'NorESM1-M':None,
'CCSM4':None,
'CESM1-CAM5':3.4,
'IPSL-CM5B-LR':None,
'IPSL-CM5A-MR':None,
'IPSL-CM5A-LR':None,
"EC-EARTH":None,
'CNRM-CM5':None,
'MPI-ESM-MR':None,
'MPI-ESM-LR':None,
'GFDL-ESM2G':None,
'GFDL-ESM2M':None,
'GFDL-CM3':None,
'MIROC5':None,
'MIROC-ESM':None,
'GISS-E2-H':None,
'GISS-E2-R':None,
'bcc-csm1-1':None,
'bcc-csm1-1-m':None,
"BNU-ESM":None,
'inmcm4':None,
'CanESM2':None,
'MRI-CGCM3':None,
'CSIRO-Mk3-6-0':None,
"FGOALS-g2":None}

coords, values = zip(* list(CMIP5_ECS_Bacmeister.items()))
ds_ECS_Bacmeister = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# ECS is estimated from a 150-year integration of an abrupt4xCO2 experiment, subtracting the parallel piControl integration from the same period, regressing the
# modelled top-of-atmosphere energy imbalance Δ𝑁 against the modelled GSAT change Δ𝑇, and taking Δ𝑇/2 at the point where the regression slope crosses Δ𝑁 = 0 (Gregory et al., 2004).
# CMIP5_ECS_IPCC = {'ACCESS1-0':3.83,'ACCESS1-3':3.53,"BNU-ESM":3.92,'CCSM4':2.94,'CESM1-CAM5':None,
# 'CNRM-CM5':3.25,'CSIRO-Mk3-6-0':4.08,'CanESM2':3.69,"EC-EARTH":None,"FGOALS-g2":3.38,'GFDL-CM3':3.97,'GFDL-ESM2G':2.39,
# 'GFDL-ESM2M':2.44,'GISS-E2-H':2.31,'GISS-E2-R':2.11,'HadGEM2-ES':4.61,'IPSL-CM5A-LR':4.13,'IPSL-CM5A-MR':4.12,
# 'IPSL-CM5B-LR':2.60,'MIROC-ESM':4.67,'MIROC5':2.72,'MPI-ESM-LR':3.63,'MPI-ESM-MR':3.46,'MRI-CGCM3':2.60,
# 'NorESM1-ME':None,'NorESM1-M':2.80,'bcc-csm1-1-m':2.86,'bcc-csm1-1':2.83,'inmcm4':2.08}

CMIP5_ECS_IPCC = {
'ACCESS1-0':3.83,
'ACCESS1-3':3.53,
'HadGEM2-ES':4.61,
'NorESM1-ME':None,
'NorESM1-M':2.80,
'CCSM4':2.94,
'CESM1-CAM5':None,
'IPSL-CM5B-LR':2.60,
'IPSL-CM5A-MR':4.12,
'IPSL-CM5A-LR':4.13,
"EC-EARTH":None,
'CNRM-CM5':3.25,
'MPI-ESM-MR':3.46,
'MPI-ESM-LR':3.63,
'GFDL-ESM2G':2.39,
'GFDL-ESM2M':2.44,
'GFDL-CM3':3.97,
'MIROC5':2.72,
'MIROC-ESM':4.67,
'GISS-E2-H':2.31,
'GISS-E2-R':2.11,
'bcc-csm1-1':2.83,
'bcc-csm1-1-m':2.86,
"BNU-ESM":3.92,
'inmcm4':2.08,
'CanESM2':3.69,
'MRI-CGCM3':2.60,
'CSIRO-Mk3-6-0':4.08,
"FGOALS-g2":3.38}

# missing: 'CESM1-BGC','CESM1-CAM5', FIO-ESM','GISS-E2-H-CC',
# 'GISS-E2-R-CC','HadGEM2-AO','HadGEM2-CC','MIROC-ESM-CHEM',
# 'MRI-ESM1','NorESM1-ME'

coords, values = zip(* list(CMIP5_ECS_IPCC.items()))
ds_ECS_IPCC = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))


# Zelinka et al. 2020: effective 2xCO2 radiative forcing, and radiative feedbacks for all CMIP5 and CMIP6 models
# that have published output from abrupt CO2 quadrupling experiments. Two tables contain the "flagship" model variants
# for CMIP5 and CMIP6. These are typically but not always the 'r1i1p1' variant (CMIP5) or the 'r1i1p1f1' variant (CMIP6),
# and are updated from Tables S1 and S2 in Zelinka et al. (2020).
# CMIP5_ECS_Zelinka = {'ACCESS1-0':3.85,'ACCESS1-3':3.55,"BNU-ESM":4.04,'CCSM4':2.94,'CESM1-CAM5':None,
# 'CNRM-CM5':3.25,'CSIRO-Mk3-6-0':4.09,'CanESM2':3.70,"EC-EARTH":None,"FGOALS-g2":3.37,'GFDL-CM3':3.95,'GFDL-ESM2G':2.43,
# 'GFDL-ESM2M':2.44,'GISS-E2-H':2.31,'GISS-E2-R':2.12,'HadGEM2-ES':4.60,'IPSL-CM5A-LR':4.13,'IPSL-CM5A-MR':4.11,
# 'IPSL-CM5B-LR':2.61,'MIROC-ESM':4.65,'MIROC5':2.71,'MPI-ESM-LR':3.63,'MPI-ESM-MR':3.45,'MRI-CGCM3':2.61,
# 'NorESM1-ME':2.98,'NorESM1-M':2.87,'bcc-csm1-1-m':2.89,'bcc-csm1-1':2.82,'inmcm4':2.08}

CMIP5_ECS_Zelinka = {
'ACCESS1-0':3.85,
'ACCESS1-3':3.55,
'HadGEM2-ES':4.60,
'NorESM1-ME':2.98,
'NorESM1-M':2.87,
'CCSM4':2.94,
'CESM1-CAM5':None,
'IPSL-CM5B-LR':2.61,
'IPSL-CM5A-MR':4.11,
'IPSL-CM5A-LR':4.13,
"EC-EARTH":None,
'CNRM-CM5':3.25,
'MPI-ESM-MR':3.45,
'MPI-ESM-LR':3.63,
'GFDL-ESM2G':2.43,
'GFDL-ESM2M':2.44,
'GFDL-CM3':3.95,
'MIROC5':2.71,
'MIROC-ESM':4.65,
'GISS-E2-H':2.31,
'GISS-E2-R':2.12,
'bcc-csm1-1':2.82,
'bcc-csm1-1-m':2.89,
"BNU-ESM":4.04,
'inmcm4':2.08,
'CanESM2':3.70,
'MRI-CGCM3':2.61,
'CSIRO-Mk3-6-0':4.09,
"FGOALS-g2":3.37}

# missing: 'CESM1-BGC','CESM1-CAM5', FIO-ESM','GISS-E2-H-CC',
# 'GISS-E2-R-CC','HadGEM2-AO','HadGEM2-CC','MIROC-ESM-CHEM',
# 'MRI-ESM1','NorESM1-ME'

coords, values = zip(* list(CMIP5_ECS_Zelinka.items()))
ds_ECS_Zelinka = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# Wyser et al. 2020: For this reason, modellers often apply the method
# proposed by Gregory et al. (2004) that has also been used to
# estimate ECS in CMIP5 (IPCC 2013, Andrews et al., 2012)
# and CMIP6 models (e.g. Andrews et al., 2019; Voldoire et
# al., 2019; Zelinka et al., 2020). Here, we apply the Gregory
# method to the 4xCO2 experiments for CMIP6 which are only
# 150 years long (Eyring et al., 2016).

CMIP5_ECS_Wyser = {
'ACCESS1-0':None,
'ACCESS1-3':None,
'HadGEM2-ES':None,
'NorESM1-ME':None,
'NorESM1-M':None,
'CCSM4':None,
'CESM1-CAM5':None,
'IPSL-CM5B-LR':None,
'IPSL-CM5A-MR':None,
'IPSL-CM5A-LR':None,
"EC-EARTH":3.34,
'CNRM-CM5':None,
'MPI-ESM-MR':None,
'MPI-ESM-LR':None,
'GFDL-ESM2G':None,
'GFDL-ESM2M':None,
'GFDL-CM3':None,
'MIROC5':None,
'MIROC-ESM':None,
'GISS-E2-H':None,
'GISS-E2-R':None,
'bcc-csm1-1':None,
'bcc-csm1-1-m':None,
"BNU-ESM":None,
'inmcm4':None,
'CanESM2':None,
'MRI-CGCM3':None,
'CSIRO-Mk3-6-0':None,
"FGOALS-g2":None}

coords, values = zip(* list(CMIP5_ECS_Wyser.items()))
ds_ECS_Wyser = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))


fig = plt.figure(figsize=(13,6))
ax = plt.subplot(122)
plt.axhspan(1.5,4.5,alpha=0.1)
plt.plot(ds_ECS_Meehl,'s',label="Meehl et al. 2020")
plt.plot(ds_ECS_Seland,'d',label="Seland et al. 2020")
plt.plot(ds_ECS_Femke,'x',label="Nijsse et al. 2020")
plt.plot(ds_ECS_Flynn,'*',label="Flynn et al. 2020")
plt.plot(ds_ECS_Bacmeister,'^',label="Bacmeister et al. 2020")
plt.plot(ds_ECS_Schlund,'.',label="Schlund et al. 2020")
plt.plot(ds_ECS_Zelinka,'+',label="Zelinka et al. 2020")
plt.plot(ds_ECS_Wyser,'1',label="Wyser et al. 2020")
plt.plot(ds_ECS_IPCC,'k_',label="IPCC Table 7.SM.5")
plt.ylim([0,7])
plt.xlim([-1,29])
xticks = np.arange(0,29,1)
ax.set_xticks(xticks)
labels = ['ACCESS1-0','ACCESS1-3','HadGEM2-ES',
'NorESM1-ME','NorESM1-M','CCSM4','CESM1-CAM5',
'IPSL-CM5B-LR','IPSL-CM5A-MR','IPSL-CM5A-LR',
"EC-EARTH",'CNRM-CM5',
'MPI-ESM-MR','MPI-ESM-LR',
'GFDL-ESM2G','GFDL-ESM2M','GFDL-CM3',
'MIROC5','MIROC-ESM',
'GISS-E2-H','GISS-E2-R',
'bcc-csm1-1','bcc-csm1-1-m',"BNU-ESM",
'inmcm4','CanESM2','MRI-CGCM3','CSIRO-Mk3-6-0',"FGOALS-g2"]
ax.set_xticklabels(labels,fontsize=8,rotation = 90)
ax.set_yticklabels('',fontsize=8,rotation = 90)
plt.legend(fontsize=8,ncol=2,loc=3)
plt.grid(axis='x')
plt.title('b) CMIP5 Reported Equilibrium Climate Sensitivity',fontsize=11,fontweight='bold',loc='left')


#######

# CMIP6_ECS_Meehl = {'ACCESS-CM2':4.7,'ACCESS-ESM1-5':3.9,'AWI-CM-1-1-MR':3.2,'CAS-ESM2-0':None,'CESM2-WACCM':4.8,'CESM2':5.2,
# 'CMCC-CM2-SR5':None,'CMCC-ESM2':None,'CNRM-CM6-1-HR':4.3,'CNRM-CM6-1':4.8,'CNRM-ESM2-1':4.8,'CanESM5':5.6,
# 'E3SM-1-1':None,"EC-Earth3-Veg":4.3,"EC-Earth3":4.3,'FGOALS-f3-L':3.0,'FGOALS-g3':None,'GFDL-CM4':3.9,'GFDL-ESM4':2.6,'GISS-E2-1-G':2.7,'HadGEM3-GC31-LL':5.6,
# 'HadGEM3-GC31-MM':5.4,'INM-CM4-8':1.8,'INM-CM5-0':1.9,'IPSL-CM6A-LR':4.6,"KACE-1-0-G":4.5,'KIOST-ESM':None,"MCM-UA-1-0":3.7,'MIROC-ES2L':2.7,'MIROC6':2.6,'MPI-ESM1-2-HR':3.0,'MPI-ESM1-2-LR':3.0,
# 'MRI-ESM2-0':3.2,'NESM3':4.7,'NorESM2-MM':None,'TaiESM1':None,'UKESM1-0-LL':5.3}
#missing: 'CAS-ESM2-0', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'CanESM5-CanOE', 'E3SM-1-1', 'FGOALS-g3','FIO-ESM-2-0','NorESM2-MM','TaiESM1'

CMIP6_ECS_Meehl = {
'ACCESS-ESM1-5':3.9,
'HadGEM3-GC31-MM':5.4,
"KACE-1-0-G":4.5,
'ACCESS-CM2':4.7,
'HadGEM3-GC31-LL':5.6,
'UKESM1-0-LL':5.3,
'TaiESM1':None,
'CMCC-ESM2':None,
'CMCC-CM2-SR5':None,
'NorESM2-MM':None,
'CESM2-WACCM':4.8,
'CESM2':5.2,
'CNRM-CM6-1-HR':4.3,
'CNRM-ESM2-1':4.8,
'IPSL-CM6A-LR':4.6,
'CNRM-CM6-1':4.8,
'AWI-CM-1-1-MR':3.2,
'NESM3':4.7,
'MPI-ESM1-2-LR':3.0,
'MPI-ESM1-2-HR':3.0,
'GFDL-CM4':3.9,
'GFDL-ESM4':2.6,
"EC-Earth3":4.3,
"EC-Earth3-Veg":4.3,
'FGOALS-f3-L':3.0,
'FGOALS-g3':None,
'INM-CM4-8':1.8,
'INM-CM5-0':1.9,
'MIROC6':2.6,
'MIROC-ES2L':2.7,
'MRI-ESM2-0':3.2,
'E3SM-1-1':None,
'CanESM5':5.6,
'CAS-ESM2-0':None,
'GISS-E2-1-G':2.7,
"MCM-UA-1-0":3.7,
'KIOST-ESM':None}

coords, values = zip(* list(CMIP6_ECS_Meehl.items()))
ds6_ECS_Meehl = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# CMIP6_ECS_Seland = {'ACCESS-CM2':None,'ACCESS-ESM1-5':None,'AWI-CM-1-1-MR':None,'CAS-ESM2-0':None,'CESM2-WACCM':None,'CESM2':None,
# 'CMCC-CM2-SR5':None,'CMCC-ESM2':None,'CNRM-CM6-1-HR':None,'CNRM-CM6-1':None,'CNRM-ESM2-1':None,'CanESM5':None,
# 'E3SM-1-1':None,"EC-Earth3-Veg":None,"EC-Earth3":None,'FGOALS-f3-L':None,'FGOALS-g3':None,'GFDL-CM4':None,'GFDL-ESM4':None,'GISS-E2-1-G':None,'HadGEM3-GC31-LL':None,
# 'HadGEM3-GC31-MM':None,'INM-CM4-8':None,'INM-CM5-0':None,'IPSL-CM6A-LR':None,"KACE-1-0-G":None,'KIOST-ESM':None,"MCM-UA-1-0":None,'MIROC-ES2L':None,'MIROC6':None,'MPI-ESM1-2-HR':None,'MPI-ESM1-2-LR':None,
# 'MRI-ESM2-0':None,'NESM3':None,'NorESM2-MM':2.5,'TaiESM1':None,'UKESM1-0-LL':None}


CMIP6_ECS_Seland = {
'ACCESS-ESM1-5':None,
'HadGEM3-GC31-MM':None,
"KACE-1-0-G":None,
'ACCESS-CM2':None,
'HadGEM3-GC31-LL':None,
'UKESM1-0-LL':None,
'TaiESM1':None,
'CMCC-ESM2':None,
'CMCC-CM2-SR5':None,
'NorESM2-MM':2.5,
'CESM2-WACCM':None,
'CESM2':None,
'CNRM-CM6-1-HR':None,
'CNRM-ESM2-1':None,
'IPSL-CM6A-LR':None,
'CNRM-CM6-1':None,
'AWI-CM-1-1-MR':None,
'NESM3':None,
'MPI-ESM1-2-LR':None,
'MPI-ESM1-2-HR':None,
'GFDL-CM4':None,
'GFDL-ESM4':None,
"EC-Earth3":None,
"EC-Earth3-Veg":None,
'FGOALS-f3-L':None,
'FGOALS-g3':None,
'INM-CM4-8':None,
'INM-CM5-0':None,
'MIROC6':None,
'MIROC-ES2L':None,
'MRI-ESM2-0':None,
'E3SM-1-1':None,
'CanESM5':None,
'CAS-ESM2-0':None,
'GISS-E2-1-G':None,
"MCM-UA-1-0":None,
'KIOST-ESM':None}

coords, values = zip(* list(CMIP6_ECS_Seland.items()))
ds6_ECS_Seland = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# CMIP6_ECS_Femke = {'ACCESS-CM2':4.81,'ACCESS-ESM1-5':3.97,'AWI-CM-1-1-MR':None,'CAS-ESM2-0':None,'CESM2-WACCM':4.90,'CESM2':5.30,
# 'CMCC-CM2-SR5':None,'CMCC-ESM2':None,'CNRM-CM6-1-HR':None,'CNRM-CM6-1':4.94,'CNRM-ESM2-1':4.66,'CanESM5':5.66,
# 'E3SM-1-1':None,"EC-Earth3-Veg":4.34,"EC-Earth3":4.22,'FGOALS-f3-L':3.03,'FGOALS-g3':None,'GFDL-CM4':4.09,'GFDL-ESM4':2.68,'GISS-E2-1-G':2.71,'HadGEM3-GC31-LL':5.62,
# 'HadGEM3-GC31-MM':5.52,'INM-CM4-8':1.84,'INM-CM5-0':1.93,'IPSL-CM6A-LR':4.63,"KACE-1-0-G":None,'KIOST-ESM':None,"MCM-UA-1-0":None,'MIROC-ES2L':None,'MIROC6':2.56,'MPI-ESM1-2-HR':2.99,'MPI-ESM1-2-LR':None,
# 'MRI-ESM2-0':3.14,'NESM3':4.76,'NorESM2-MM':None,'TaiESM1':None,'UKESM1-0-LL':5.41}

CMIP6_ECS_Femke = {
'ACCESS-ESM1-5':3.97,
'HadGEM3-GC31-MM':5.52,
"KACE-1-0-G":None,
'ACCESS-CM2':4.81,
'HadGEM3-GC31-LL':5.62,
'UKESM1-0-LL':5.41,
'TaiESM1':None,
'CMCC-ESM2':None,
'CMCC-CM2-SR5':None,
'NorESM2-MM':None,
'CESM2-WACCM':4.90,
'CESM2':5.30,
'CNRM-CM6-1-HR':None,
'CNRM-ESM2-1':4.66,
'IPSL-CM6A-LR':4.63,
'CNRM-CM6-1':4.94,
'AWI-CM-1-1-MR':None,
'NESM3':4.76,
'MPI-ESM1-2-LR':None,
'MPI-ESM1-2-HR':2.99,
'GFDL-CM4':4.09,
'GFDL-ESM4':2.68,
"EC-Earth3":4.22,
"EC-Earth3-Veg":4.34,
'FGOALS-f3-L':3.03,
'FGOALS-g3':None,
'INM-CM4-8':1.84,
'INM-CM5-0':1.93,
'MIROC6':2.56,
'MIROC-ES2L':None,
'MRI-ESM2-0':3.14,
'E3SM-1-1':None,
'CanESM5':5.66,
'CAS-ESM2-0':None,
'GISS-E2-1-G':2.71,
"MCM-UA-1-0":None,
'KIOST-ESM':None}

#missing: 'AWI-CM-1-1-MR','CAS-ESM2-0', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'CNRM-CM6-1-HR',
# 'CanESM5-CanOE', 'E3SM-1-1', 'FGOALS-g3','FIO-ESM-2-0','MIROC-ES2L','MPI-ESM1-2-LR','NorESM2-MM','TaiESM1'

coords, values = zip(* list(CMIP6_ECS_Femke.items()))
ds6_ECS_Femke = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# CMIP6_ECS_Flynn = {'ACCESS-CM2':None,'ACCESS-ESM1-5':None,'AWI-CM-1-1-MR':None,'CAS-ESM2-0':None,'CESM2-WACCM':4.65,'CESM2':5.15,
# 'CMCC-CM2-SR5':None,'CMCC-ESM2':None,'CNRM-CM6-1-HR':None,'CNRM-CM6-1':4.81,'CNRM-ESM2-1':4.75,'CanESM5':5.58,
# 'E3SM-1-1':None,"EC-Earth3-Veg":3.93,"EC-Earth3":None,'FGOALS-f3-L':None,'FGOALS-g3':None,'GFDL-CM4':3.79,'GFDL-ESM4':2.56,'GISS-E2-1-G':2.60,'HadGEM3-GC31-LL':5.46,
# 'HadGEM3-GC31-MM':None,'INM-CM4-8':1.81,'INM-CM5-0':None,'IPSL-CM6A-LR':4.50,"KACE-1-0-G":None,'KIOST-ESM':None,"MCM-UA-1-0":None,'MIROC-ES2L':2.66,'MIROC6':2.60,'MPI-ESM1-2-HR':2.84,'MPI-ESM1-2-LR':None,
# 'MRI-ESM2-0':3.11,'NESM3':4.50,'NorESM2-MM':None,'TaiESM1':None,'UKESM1-0-LL':5.31}

CMIP6_ECS_Flynn = {
'ACCESS-ESM1-5':None,
'HadGEM3-GC31-MM':None,
"KACE-1-0-G":None,
'ACCESS-CM2':None,
'HadGEM3-GC31-LL':5.46,
'UKESM1-0-LL':5.31,
'TaiESM1':None,
'CMCC-ESM2':None,
'CMCC-CM2-SR5':None,
'NorESM2-MM':None,
'CESM2-WACCM':4.65,
'CESM2':5.15,
'CNRM-CM6-1-HR':None,
'CNRM-ESM2-1':4.75,
'IPSL-CM6A-LR':4.50,
'CNRM-CM6-1':4.81,
'AWI-CM-1-1-MR':None,
'NESM3':4.50,
'MPI-ESM1-2-LR':None,
'MPI-ESM1-2-HR':2.84,
'GFDL-CM4':3.79,
'GFDL-ESM4':2.56,
"EC-Earth3":None,
"EC-Earth3-Veg":3.93,
'FGOALS-f3-L':None,
'FGOALS-g3':None,
'INM-CM4-8':1.81,
'INM-CM5-0':None,
'MIROC6':2.60,
'MIROC-ES2L':2.66,
'MRI-ESM2-0':3.11,
'E3SM-1-1':None,
'CanESM5':5.58,
'CAS-ESM2-0':None,
'GISS-E2-1-G':2.60,
"MCM-UA-1-0":None,
'KIOST-ESM':None}

#missing: 'ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'CAS-ESM2-0', 'CNRM-CM6-1-HR','CMCC-CM2-SR5','CMCC-ESM2',
# 'CanESM5-CanOE', 'E3SM-1-1','FGOALS-f3-L','FGOALS-g3','FIO-ESM-2-0','HadGEM3-GC31-MM','INM-CM5-0','MPI-ESM1-2-LR','NorESM2-MM','TaiESM1'

coords, values = zip(* list(CMIP6_ECS_Flynn.items()))
ds6_ECS_Flynn = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# CMIP6_ECS_Schlund = {'ACCESS-CM2':4.72,'ACCESS-ESM1-5':3.87,'AWI-CM-1-1-MR':3.16,'CAS-ESM2-0':3.51,'CESM2-WACCM':4.75,'CESM2':5.16,
# 'CMCC-CM2-SR5':3.52,'CMCC-ESM2':None,'CNRM-CM6-1-HR':4.28,'CNRM-CM6-1':4.83,'CNRM-ESM2-1':4.76,'CanESM5':5.62,
# 'E3SM-1-1':None,"EC-Earth3-Veg":4.31,"EC-Earth3":None,'FGOALS-f3-L':3.00,'FGOALS-g3':2.88,'GFDL-CM4':None,'GFDL-ESM4':None,'GISS-E2-1-G':2.72,'HadGEM3-GC31-LL':5.55,
# 'HadGEM3-GC31-MM':5.42,'INM-CM4-8':1.83,'INM-CM5-0':1.92,'IPSL-CM6A-LR':4.56,"KACE-1-0-G":4.48,'KIOST-ESM':None,"MCM-UA-1-0":3.65,'MIROC-ES2L':2.68,'MIROC6':2.61,'MPI-ESM1-2-HR':2.98,'MPI-ESM1-2-LR':3.00,
# 'MRI-ESM2-0':3.15,'NESM3':4.72,'NorESM2-MM':2.50,'TaiESM1':4.31,'UKESM1-0-LL':5.34}

CMIP6_ECS_Schlund = {
'ACCESS-ESM1-5':3.87,
'HadGEM3-GC31-MM':5.42,
"KACE-1-0-G":4.48,
'ACCESS-CM2':4.72,
'HadGEM3-GC31-LL':5.55,
'UKESM1-0-LL':5.34,
'TaiESM1':4.31,
'CMCC-ESM2':None,
'CMCC-CM2-SR5':3.52,
'NorESM2-MM':2.50,
'CESM2-WACCM':4.75,
'CESM2':5.16,
'CNRM-CM6-1-HR':4.28,
'CNRM-ESM2-1':4.76,
'IPSL-CM6A-LR':4.56,
'CNRM-CM6-1':4.83,
'AWI-CM-1-1-MR':3.16,
'NESM3':4.72,
'MPI-ESM1-2-LR':3.00,
'MPI-ESM1-2-HR':2.98,
'GFDL-CM4':None,
'GFDL-ESM4':None,
"EC-Earth3":None,
"EC-Earth3-Veg":4.31,
'FGOALS-f3-L':3.00,
'FGOALS-g3':2.88,
'INM-CM4-8':1.83,
'INM-CM5-0':1.92,
'MIROC6':2.61,
'MIROC-ES2L':2.68,
'MRI-ESM2-0':3.15,
'E3SM-1-1':None,
'CanESM5':5.62,
'CAS-ESM2-0':3.51,
'GISS-E2-1-G':2.72,
"MCM-UA-1-0":3.65,
'KIOST-ESM':None}

#missing: 'CMCC-ESM2','CanESM5-CanOE', 'E3SM-1-1','FIO-ESM-2-0','GFDL-CM4','GFDL-ESM4','MPI-ESM1-2-LR'

coords, values = zip(* list(CMIP6_ECS_Schlund.items()))
ds6_ECS_Schlund = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# CMIP6_ECS_IPCC = {'ACCESS-CM2':4.72,'ACCESS-ESM1-5':3.87,'AWI-CM-1-1-MR':3.16,'CAS-ESM2-0':3.51,'CESM2-WACCM':4.75,'CESM2':5.16,
# 'CMCC-CM2-SR5':3.52,'CMCC-ESM2':None,'CNRM-CM6-1-HR':4.28,'CNRM-CM6-1':4.83,'CNRM-ESM2-1':4.76,'CanESM5':5.62,
# 'E3SM-1-1':None,"EC-Earth3-Veg":4.31,"EC-Earth3":None,'FGOALS-f3-L':3.00,'FGOALS-g3':2.88,'GFDL-CM4':None,'GFDL-ESM4':None,'GISS-E2-1-G':2.72,'HadGEM3-GC31-LL':5.55,
# 'HadGEM3-GC31-MM':5.42,'INM-CM4-8':1.83,'INM-CM5-0':1.92,'IPSL-CM6A-LR':4.56,"KACE-1-0-G":4.48,'KIOST-ESM':None,"MCM-UA-1-0":3.65,'MIROC-ES2L':2.68,'MIROC6':2.61,'MPI-ESM1-2-HR':2.98,'MPI-ESM1-2-LR':3.00,
# 'MRI-ESM2-0':3.15,'NESM3':4.72,'NorESM2-MM':2.50,'TaiESM1':4.31,'UKESM1-0-LL':5.34}

CMIP6_ECS_IPCC = {
'ACCESS-ESM1-5':3.87,
'HadGEM3-GC31-MM':5.42,
"KACE-1-0-G":4.48,
'ACCESS-CM2':4.72,
'HadGEM3-GC31-LL':5.55,
'UKESM1-0-LL':5.34,
'TaiESM1':4.31,
'CMCC-ESM2':None,
'CMCC-CM2-SR5':3.52,
'NorESM2-MM':2.50,
'CESM2-WACCM':4.75,
'CESM2':5.16,
'CNRM-CM6-1-HR':4.28,
'CNRM-ESM2-1':4.76,
'IPSL-CM6A-LR':4.56,
'CNRM-CM6-1':4.83,
'AWI-CM-1-1-MR':3.16,
'NESM3':4.72,
'MPI-ESM1-2-LR':3.00,
'MPI-ESM1-2-HR':2.98,
'GFDL-CM4':None,
'GFDL-ESM4':None,
"EC-Earth3":None,
"EC-Earth3-Veg":4.31,
'FGOALS-f3-L':3.00,
'FGOALS-g3':2.88,
'INM-CM4-8':1.83,
'INM-CM5-0':1.92,
'MIROC6':2.61,
'MIROC-ES2L':2.68,
'MRI-ESM2-0':3.15,
'E3SM-1-1':None,
'CanESM5':5.62,
'CAS-ESM2-0':3.51,
'GISS-E2-1-G':2.72,
"MCM-UA-1-0":3.65,
'KIOST-ESM':None}

#missing: 'CMCC-ESM2','CanESM5-CanOE', 'E3SM-1-1','FIO-ESM-2-0','GFDL-CM4','GFDL-ESM4'

coords, values = zip(* list(CMIP6_ECS_IPCC.items()))
ds6_ECS_IPCC = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# CMIP6_ECS_Zelinka = {'ACCESS-CM2':4.66,'ACCESS-ESM1-5':3.88,'AWI-CM-1-1-MR':3.16,'CAS-ESM2-0':None,'CESM2-WACCM':4.68,'CESM2':5.15,
# 'CMCC-CM2-SR5':3.55,'CMCC-ESM2':3.58,'CNRM-CM6-1-HR':4.33,'CNRM-CM6-1':4.90,'CNRM-ESM2-1':4.79,'CanESM5':5.64,
# 'E3SM-1-1':None,"EC-Earth3-Veg":4.33,"EC-Earth3":4.26,'FGOALS-f3-L':2.98,'FGOALS-g3':2.87,'GFDL-CM4':3.89,'GFDL-ESM4':2.65,'GISS-E2-1-G':2.71,'HadGEM3-GC31-LL':5.55,
# 'HadGEM3-GC31-MM':5.44,'INM-CM4-8':1.83,'INM-CM5-0':1.92,'IPSL-CM6A-LR':4.70,"KACE-1-0-G":4.93,'KIOST-ESM':None,"MCM-UA-1-0":None,'MIROC-ES2L':2.66,'MIROC6':2.60,'MPI-ESM1-2-HR':2.98,'MPI-ESM1-2-LR':3.03,
# 'MRI-ESM2-0':3.13,'NESM3':4.76,'NorESM2-MM':2.49,'TaiESM1':4.36,'UKESM1-0-LL':5.36}

##
CMIP6_ECS_Zelinka = {
'ACCESS-ESM1-5':3.88,
'HadGEM3-GC31-MM':5.44,
"KACE-1-0-G":4.93,
'ACCESS-CM2':4.66,
'HadGEM3-GC31-LL':5.55,
'UKESM1-0-LL':5.36,
'TaiESM1':4.36,
'CMCC-ESM2':3.58,
'CMCC-CM2-SR5':3.55,
'NorESM2-MM':2.49,
'CESM2-WACCM':4.68,
'CESM2':5.15,
'CNRM-CM6-1-HR':4.33,
'CNRM-ESM2-1':4.79,
'IPSL-CM6A-LR':4.70,
'CNRM-CM6-1':4.90,
'AWI-CM-1-1-MR':3.16,
'NESM3':4.76,
'MPI-ESM1-2-LR':3.03,
'MPI-ESM1-2-HR':2.98,
'GFDL-CM4':3.89,
'GFDL-ESM4':2.65,
"EC-Earth3":4.26,
"EC-Earth3-Veg":4.33,
'FGOALS-f3-L':2.98,
'FGOALS-g3':2.87,
'INM-CM4-8':1.83,
'INM-CM5-0':1.92,
'MIROC6':2.60,
'MIROC-ES2L':2.66,
'MRI-ESM2-0':3.13,
'E3SM-1-1':None,
'CanESM5':5.64,
'CAS-ESM2-0':None,
'GISS-E2-1-G':2.71,
"MCM-UA-1-0":None,
'KIOST-ESM':None}

#missing: 'CAS-ESM2-0','CanESM5-CanOE', 'E3SM-1-1','FIO-ESM-2-0'

coords, values = zip(* list(CMIP6_ECS_Zelinka.items()))
ds6_ECS_Zelinka = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# Golaz et al. 2019
# CMIP6_ECS_Golaz = {'ACCESS-CM2':None,'ACCESS-ESM1-5':None,'AWI-CM-1-1-MR':None,'CAS-ESM2-0':None,'CESM2-WACCM':None,'CESM2':None,
# 'CMCC-CM2-SR5':None,'CMCC-ESM2':None,'CNRM-CM6-1-HR':None,'CNRM-CM6-1':None,'CNRM-ESM2-1':None,'CanESM5':None,
# 'E3SM-1-1':5.30,"EC-Earth3-Veg":None,"EC-Earth3":None,'FGOALS-f3-L':None,'FGOALS-g3':None,'GFDL-CM4':None,'GFDL-ESM4':None,'GISS-E2-1-G':None,'HadGEM3-GC31-LL':None,
# 'HadGEM3-GC31-MM':None,'INM-CM4-8':None,'INM-CM5-0':None,'IPSL-CM6A-LR':None,"KACE-1-0-G":None,'KIOST-ESM':None,"MCM-UA-1-0":None,'MIROC-ES2L':None,'MIROC6':None,'MPI-ESM1-2-HR':None,'MPI-ESM1-2-LR':None,
# 'MRI-ESM2-0':None,'NESM3':None,'NorESM2-MM':None,'TaiESM1':None,'UKESM1-0-LL':None}

CMIP6_ECS_Golaz = {
'ACCESS-ESM1-5':None,
'HadGEM3-GC31-MM':None,
"KACE-1-0-G":None,
'ACCESS-CM2':None,
'HadGEM3-GC31-LL':None,
'UKESM1-0-LL':None,
'TaiESM1':None,
'CMCC-ESM2':None,
'CMCC-CM2-SR5':None,
'NorESM2-MM':None,
'CESM2-WACCM':None,
'CESM2':None,
'CNRM-CM6-1-HR':None,
'CNRM-ESM2-1':None,
'IPSL-CM6A-LR':None,
'CNRM-CM6-1':None,
'AWI-CM-1-1-MR':None,
'NESM3':None,
'MPI-ESM1-2-LR':None,
'MPI-ESM1-2-HR':None,
'GFDL-CM4':None,
'GFDL-ESM4':None,
"EC-Earth3":None,
"EC-Earth3-Veg":None,
'FGOALS-f3-L':None,
'FGOALS-g3':None,
'INM-CM4-8':None,
'INM-CM5-0':None,
'MIROC6':None,
'MIROC-ES2L':None,
'MRI-ESM2-0':None,
'E3SM-1-1':5.30,
'CanESM5':None,
'CAS-ESM2-0':None,
'GISS-E2-1-G':None,
"MCM-UA-1-0":None,
'KIOST-ESM':None}

coords, values = zip(* list(CMIP6_ECS_Golaz.items()))
ds6_ECS_Golaz = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# Torkarska et al. 2020
# CMIP6_ECS_Torkarska = {'ACCESS-CM2':None,'ACCESS-ESM1-5':None,'AWI-CM-1-1-MR':None,'CAS-ESM2-0':None,'CESM2-WACCM':4.70,'CESM2':5.19,
# 'CMCC-CM2-SR5':None,'CMCC-ESM2':None,'CNRM-CM6-1-HR':4.28,'CNRM-CM6-1':4.83,'CNRM-ESM2-1':4.70,'CanESM5':5.62,
# 'E3SM-1-1':None,"EC-Earth3-Veg":4.30,"EC-Earth3":4.20,'FGOALS-f3-L':2.99,'FGOALS-g3':None,'GFDL-CM4':3.87,'GFDL-ESM4':2.62,'GISS-E2-1-G':2.72,'HadGEM3-GC31-LL':5.50,
# 'HadGEM3-GC31-MM':None,'INM-CM4-8':1.83,'INM-CM5-0':1.92,'IPSL-CM6A-LR':4.52,"KACE-1-0-G":None,'KIOST-ESM':None,"MCM-UA-1-0":None,'MIROC-ES2L':2.68,'MIROC6':2.57,'MPI-ESM1-2-HR':2.97,'MPI-ESM1-2-LR':None,
# 'MRI-ESM2-0':3.14,'NESM3':4.68,'NorESM2-MM':None,'TaiESM1':None,'UKESM1-0-LL':5.34}

CMIP6_ECS_Torkarska = {
'ACCESS-ESM1-5':None,
'HadGEM3-GC31-MM':None,
"KACE-1-0-G":None,
'ACCESS-CM2':None,
'HadGEM3-GC31-LL':5.50,
'UKESM1-0-LL':5.34,
'TaiESM1':None,
'CMCC-ESM2':None,
'CMCC-CM2-SR5':None,
'NorESM2-MM':None,
'CESM2-WACCM':4.70,
'CESM2':5.19,
'CNRM-CM6-1-HR':4.28,
'CNRM-ESM2-1':4.70,
'IPSL-CM6A-LR':4.52,
'CNRM-CM6-1':4.83,
'AWI-CM-1-1-MR':None,
'NESM3':4.68,
'MPI-ESM1-2-LR':None,
'MPI-ESM1-2-HR':2.97,
'GFDL-CM4':3.87,
'GFDL-ESM4':2.62,
"EC-Earth3":4.20,
"EC-Earth3-Veg":4.30,
'FGOALS-f3-L':2.99,
'FGOALS-g3':None,
'INM-CM4-8':1.83,
'INM-CM5-0':1.92,
'MIROC6':2.57,
'MIROC-ES2L':2.68,
'MRI-ESM2-0':3.14,
'E3SM-1-1':None,
'CanESM5':5.62,
'CAS-ESM2-0':None,
'GISS-E2-1-G':2.72,
"MCM-UA-1-0":None,
'KIOST-ESM':None}

#missing: 'ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'CAS-ESM2-0','CMCC-CM2-SR5','CMCC-ESM2',
# 'CanESM5-CanOE', 'E3SM-1-1', 'FGOALS-g3','FIO-ESM-2-0','HadGEM3-GC31-MM','MPI-ESM1-2-LR','NorESM2-MM','TaiESM1'

coords, values = zip(* list(CMIP6_ECS_Torkarska.items()))
ds6_ECS_Torkarska = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

# Pak et al. 2021
# CMIP6_ECS_Pak = {'ACCESS-CM2':None,'ACCESS-ESM1-5':None,'AWI-CM-1-1-MR':None,'CAS-ESM2-0':None,'CESM2-WACCM':None,'CESM2':None,
# 'CMCC-CM2-SR5':None,'CMCC-ESM2':None,'CNRM-CM6-1-HR':None,'CNRM-CM6-1':None,'CNRM-ESM2-1':None,'CanESM5':None,
# 'E3SM-1-1':None,"EC-Earth3-Veg":None,"EC-Earth3":None,'FGOALS-f3-L':None,'FGOALS-g3':None,'GFDL-CM4':None,'GFDL-ESM4':None,'GISS-E2-1-G':None,'HadGEM3-GC31-LL':None,
# 'HadGEM3-GC31-MM':None,'INM-CM4-8':None,'INM-CM5-0':None,'IPSL-CM6A-LR':None,"KACE-1-0-G":None,'KIOST-ESM':3.36,"MCM-UA-1-0":None,'MIROC-ES2L':None,'MIROC6':None,'MPI-ESM1-2-HR':None,'MPI-ESM1-2-LR':None,
# 'MRI-ESM2-0':None,'NESM3':None,'NorESM2-MM':None,'TaiESM1':None,'UKESM1-0-LL':None}

CMIP6_ECS_Pak = {
'ACCESS-ESM1-5':None,
'HadGEM3-GC31-MM':None,
"KACE-1-0-G":None,
'ACCESS-CM2':None,
'HadGEM3-GC31-LL':None,
'UKESM1-0-LL':None,
'TaiESM1':None,
'CMCC-ESM2':None,
'CMCC-CM2-SR5':None,
'NorESM2-MM':None,
'CESM2-WACCM':None,
'CESM2':None,
'CNRM-CM6-1-HR':None,
'CNRM-ESM2-1':None,
'IPSL-CM6A-LR':None,
'CNRM-CM6-1':None,
'AWI-CM-1-1-MR':None,
'NESM3':None,
'MPI-ESM1-2-LR':None,
'MPI-ESM1-2-HR':None,
'GFDL-CM4':None,
'GFDL-ESM4':None,
"EC-Earth3":None,
"EC-Earth3-Veg":None,
'FGOALS-f3-L':None,
'FGOALS-g3':None,
'INM-CM4-8':None,
'INM-CM5-0':None,
'MIROC6':None,
'MIROC-ES2L':None,
'MRI-ESM2-0':None,
'E3SM-1-1':None,
'CanESM5':None,
'CAS-ESM2-0':None,
'GISS-E2-1-G':None,
"MCM-UA-1-0":None,
'KIOST-ESM':3.36}

#missing: 'ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'CAS-ESM2-0','CMCC-CM2-SR5','CMCC-ESM2',
# 'CanESM5-CanOE', 'E3SM-1-1', 'FGOALS-g3','FIO-ESM-2-0','HadGEM3-GC31-MM','MPI-ESM1-2-LR','NorESM2-MM','TaiESM1'

coords, values = zip(* list(CMIP6_ECS_Pak.items()))
ds6_ECS_Pak = xr.DataArray(np.array(values),dims=['model'],coords=dict(model=list(coords)))

##########

ax = plt.subplot(121)
plt.axhspan(1.5,4.5,alpha=0.1)
plt.plot(ds6_ECS_Meehl,'s',label="Meehl et al. 2020")
plt.plot(ds6_ECS_Seland,'d',label="Seland et al. 2020")
plt.plot(ds6_ECS_Femke,'x',label="Nijsse et al. 2020")
plt.plot(ds6_ECS_Flynn,'*',label="Flynn et al. 2020")
plt.plot(ds6_ECS_Golaz,'^',label="Golaz et al. 2019")
plt.plot(ds6_ECS_Schlund,'.',label="Schlund et al. 2020")
plt.plot(ds6_ECS_Zelinka,'+',label="Zelinka et al. 2020")
plt.plot(ds6_ECS_Torkarska,'1',color='tab:cyan',label="Torkarska et al. 2020")
plt.plot(ds6_ECS_Pak,'2',color='tab:olive',label="Pak et al. 2021")
plt.plot(ds6_ECS_IPCC,'k_',label="IPCC Table 7.SM.5")
plt.ylim([0,7])
plt.xlim([-1,37])
plt.ylabel('ECS (K; Gregory et al. 2004 Method)')
xticks = np.arange(0,37,1)
ax.set_xticks(xticks)
labels = ['ACCESS-ESM1-5','HadGEM3-GC31-MM',"KACE-1-0-G",'ACCESS-CM2','HadGEM3-GC31-LL','UKESM1-0-LL',
'TaiESM1','CMCC-ESM2','CMCC-CM2-SR5','NorESM2-MM','CESM2-WACCM','CESM2',
'CNRM-CM6-1-HR','CNRM-ESM2-1','IPSL-CM6A-LR','CNRM-CM6-1',
'AWI-CM-1-1-MR','NESM3','MPI-ESM1-2-LR','MPI-ESM1-2-HR',
'GFDL-CM4','GFDL-ESM4',
"EC-Earth3","EC-Earth3-Veg",
'FGOALS-f3-L','FGOALS-g3',
'INM-CM4-8','INM-CM5-0',
'MIROC6','MIROC-ES2L',
'MRI-ESM2-0','E3SM-1-1','CanESM5','CAS-ESM2-0','GISS-E2-1-G',"MCM-UA-1-0",'KIOST-ESM']
ax.set_xticklabels(labels,fontsize=8,rotation = 90)
#ax.set_yticklabels('')
plt.legend(fontsize=8,ncol=2,loc=3)
plt.grid(axis='x')
plt.title('a) CMIP6 Reported Equilibrium Climate Sensitivity',fontsize=11,fontweight='bold',loc='left')
plt.subplots_adjust(wspace=0.05, hspace=0)
# plt.show()
plt.savefig('SupFig5_CMIP56_ECS_all.png',bbox_inches='tight',dpi=300)
# plt.show()
