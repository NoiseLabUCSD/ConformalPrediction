#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:54:41 2023

@author: ikhurjekar
"""
##This code uses a Python wrapper for the Acoustics toolbox (https://oalib-acoustics.org/AcousticsToolbox/index_at.html). 
## The wrapper can be cloned and installed from https://github.com/hunterakins/pyat 

import numpy as np
import scipy.stats as st
import sys
sys.path.append("../")
from os import system
#from matplotlib import pyplot as plt
#import pyat
from pyat.env import Source, Dom, Pos
from pyat.readwrite import (SSPraw, SSP, HS, BotBndry, TopBndry, Bndry,
                                 write_env, write_fieldflp, read_shd)
from pyat.epscont import epscont
from numpy import unravel_index



##Class and functions definitions
class Empty:
    def __init__(self):
        return

##Define environment parameters 
def init_params(): 
    params = {} 
    params['freq'] = 300
    params['cw'] =	1500
    params['pw'] = 1
    params['aw']		=	0
    params['cb']		=	1600
    params['pb']		=	1.8
    params['ab']		=	0.2
    params['wavelength'] = (params['cw'])/(params['freq'])
    params['bottom_depth'] = 200
    params['depth'] = [0, params['bottom_depth']] 
    params['delta_speed'] = 0.01
    params['delta_density'] = 0
    params['num_sensors'] = 20
    params['num_snapshots'] = 1
    params['num_MC_runs'] = 25
    params['num_samples'] = 30
    params['num_samples_test'] = 30
    #array_start = 0
    #array_origin = 100
    sensor_low, sensor_high = 50,150
    params['sensor_distance'] = (sensor_high - sensor_low)/params['num_sensors']
    params['sensor_loc_true'] = np.arange(sensor_low,sensor_high, params['sensor_distance'])
    params['Z'] = np.arange(0, params['bottom_depth']*1.5, 1)
    params['range_linspace'] = 0.1
    X_max = 10
    params['X'] = np.arange(0,X_max,params['range_linspace'])
    params['rng'] = [0]
    return params

    
###Function to specify sound speed profile parameters
def ssp_specs(params):
    depth = params['depth']
    cw = params['cw']
    cb = params['cb']
    pw = params['pw']
    pb = params['pb']
    aw = params['aw']
    ab = params['ab']
    z1 = np.linspace(depth[0], depth[1], 1000)
    alphaR	=	cw*np.ones(z1.shape)
    betaR	=	0.0*np.ones(z1.shape)
    rho		=	pw*np.ones(z1.shape)
    alphaI	=	aw*np.ones(z1.shape)
    betaI	=	0.0*np.ones(z1.shape)
    ssp1 = SSPraw(z1, alphaR, betaR, rho, alphaI, betaI)
    raw = [ssp1]
    NMedia		=	1
    Opt			=	'CVW'	
    N			=	[z1.size]
    sigma		=	[.5,.5]	 # roughness at each layer. only effects attenuation (imag part)
    ssp = SSP(raw, depth, NMedia, Opt, N, sigma)
    # Layer 2
    hs = HS(alphaR=cb, betaR=0, rho = pb, alphaI=ab, betaI=0)
    Opt = 'A~'
    bottom = BotBndry(Opt, hs)
    top = TopBndry('CVW')
    bdy = Bndry(top, bottom)
    return ssp, bdy


###Function to calculate field
def calculate_field(pos_field):
    write_fieldflp('py_env', 'R', pos_field)
    system("krakenc.exe py_env")
    system("field.exe py_env")
    [x,x,x,x,Pos1,pressure_field]= read_shd('py_env.shd')
    return pressure_field, Pos1


##Function to generate acoustic measurements
def gen_measurements(params, source_loc, sensor_perturb, snr, env_perturb_factor, iterations):
    sensor_loc_true = params['sensor_loc_true'] 
    X = params['X']
    freq = params['freq']
    rng = params['rng']
    num_sensors = params['num_sensors']
    cb = 1600
    num_snapshots = params['num_snapshots']
    signals_sim = []
    num_samples = source_loc.shape[0]
    sigma_noise = 1/np.sqrt(params['num_sensors'])
    lambdavar = 10
    epsilon=0.0
    SNRoffset = 10*np.log10(1-epsilon+epsilon*lambdavar**2)
    
    #speeds = np.zeros((num_samples,2))
    #np.random.seed(23)
    #pf = -sensor_perturb + 2*sensor_perturb*np.random.rand(iterations)
    for n in range(iterations):
        pf = -sensor_perturb + 2*sensor_perturb*np.random.rand(1)
        sensor_loc = sensor_loc_true 
        s = Source(source_loc[n,0])
        r = Dom(rng,sensor_loc)
        pos = Pos(s,r)
        pos.s.depth	= [source_loc[n,0]]
       # pos.s.range = source_range[n]
        pos.r.depth = sensor_loc
        pos.r.range = [source_loc[n,1]]
        pos.Nsd = 1
        pos.Nrd = sensor_loc.shape[0]
        
        #cw_perturb = cw*(1 - delta_speed + 2*delta_speed*np.random.rand(1))
        #pw_perturb = pw*(1 - delta_density + 2*delta_density*np.random.rand(1))
        #cb_perturb = int(cb*(1 - 2*delta_speed + 4*delta_speed*np.random.rand(1)))
        params['cb'] = cb
        ssp,bdy = ssp_specs(params)
        cInt = Empty()
        cInt.High = int(cb)
        cInt.Low = 0 # compute automatically
        RMax = max(X)
        write_env('py_env.env', 'KRAKEN', 'Pekeris profile', freq, ssp, bdy, pos, [], cInt, RMax)
        
        ##Environment instantiation for true field
        pressure,Pos1 = calculate_field(pos)
        noise,nmask = epscont((pos.r.depth.shape[0], len(pos.r.range),num_snapshots),
                sigma=sigma_noise,epsilon=epsilon,lambdavar=lambdavar,return_mask=True)
        
        ##Generate noisy signal for true source location
       # pressure = pressure.repeat(num_snapshots,axis=1)
        if pressure.ndim!=2:
            pressure = pressure.reshape(pos.r.depth.shape[0], len(pos.r.range))
            noise = noise.reshape(pos.r.depth.shape[0], len(pos.r.range))
        rnl =  10 ** (-snr/ 20) * np.linalg.norm(pressure,'fro')/np.sqrt(num_snapshots)
        pressure_noisy = pressure + noise*rnl
        signals_sim.append((pressure_noisy))
    signals_sim = np.asarray(signals_sim).reshape(num_samples, num_sensors)
    return signals_sim

##Matched field processing (batched version)
def mfp(params, xvar, signals_sim, samples):
    num_MC_runs = params['num_MC_runs']
    sensor_loc_true = params['sensor_loc_true'] 
    X = params['X']
    Z = params['Z']
    freq = params['freq']
    rng = params['rng']
    range_linspace = params['range_linspace']
    num_sensors = params['num_sensors']
    cb = 1600
    ssp_model, bdy_model = ssp_specs(params)
    cInt = Empty()
    cInt.High =int(cb)
    cInt.Low = 0 # compute automatically
    RMax = max(X)
    
    
    loc_pred = np.zeros((samples, num_MC_runs,2))
    np.random.seed(23)
    for k in range(samples):
        #pf = -sensor_perturb + 2*sensor_perturb*np.random.rand(num_MC_runs)
        print("Running MFP for sample: ", k)
        for jj in range(num_MC_runs):
            params['cb']  = xvar[jj]*cb
            ssp_model, bdy_model = ssp_specs(params)
            ##Sensor location perturbations
           # pf = 0
          #  pf = -sensor_perturb + 2*sensor_perturb*np.random.rand(1)
            #sensor_loc = sensor_loc_true
            sensor_loc = sensor_loc_true 
           # cw_perturb = cw*(1 - delta_speed + 2*delta_speed*np.random.rand(1))
           # pw_perturb = pw*(1 - delta_density + 2*delta_density*np.random.rand(1))
           # cb_perturb = int(cb*(1 - 2*delta_speed + 4*delta_speed*np.random.rand(1)))
             
            ## Initialize ambiguity surface matrix and pressure replica matrix                   
            bp = np.zeros((Z.shape[0],X.shape[0]))
            pressure_replica = np.zeros((Z.shape[0], X.shape[0], num_sensors), dtype=np.complex_)
            
            ##Assumption: Tx-Rx path is same as Rx-Tx path
            ##Instead of computing replica fields at each grid point, assume each 
            ## grid point is receiver and each sensor location is a possible source
            ## location. Then do MFP for only sensor lcoations.
            ## Reduces computation from 300x100 to num_sensors.
            
            for ii in range(sensor_loc.shape[0]):
                r_field = Dom(X, Z)
                s_field = Source(np.asarray([sensor_loc[ii]]))
                pos_field = Pos(s_field,r_field)
                pos_field.s.depth = [sensor_loc[ii]]
                pos_field.s.range = np.asarray(rng)
                pos_field.r.depth = Z
                pos_field.r.range = X
                pos_field.Nsd = 1
                pos_field.Nrd = Z.shape[0]*X.shape[0]
                write_env('py_env.env', 'KRAKEN', 'Pekeris profile', freq, ssp_model, 
                          bdy_model, pos_field, [], cInt, RMax)
                pressure_, Pos_replica = calculate_field(pos_field)
                pressure_replica[:,:,ii] = pressure_
             
            ### MFP calculation 
            for ii in range(Z.shape[0]):    
                for mm in range(X.shape[0]):
                   
                    den = np.sum(np.square(np.abs(pressure_replica[ii,mm,:])))
                    num = np.square(np.abs(np.sum(np.multiply(signals_sim[k].reshape(num_sensors,), 
                                                              np.conjugate(pressure_replica[ii,mm])))))
                    bp[ii,mm] = num/den
                    
            ind_pred = np.asarray(unravel_index(bp.argmax(), bp.shape))
            loc_pred[k,jj,:] = np.multiply(ind_pred,[1,(range_linspace*1000)])
            
    return loc_pred



###random source locations
params = init_params()
num_MC_runs = params['num_MC_runs']
num_samples = params['num_samples']
num_samples_test = params['num_samples_test']


SNR_list = np.arange(-10,11,10)
sensor_perturb_list = [i*params['sensor_distance'] for i in [0.0,0.2,0.3,0.4]]
#env_perturb = [0.04]
xvar = sensor_perturb_list
#xvar = SNR_list
alpha = 0.1
snr = 20

err_mfp = np.zeros((len(xvar),2))
uq_noncf = np.zeros((len(xvar),2))
uq_cf = np.zeros((len(xvar),2))
coverage_score_noncf = np.zeros((len(xvar),2))
coverage_score_cf = np.zeros((len(xvar),2))
qhat = np.zeros((len(xvar),2))


###Simulate field measurements
for counter in range(len(xvar)):
    np.random.seed(23)
    source_depth = 30 + 140*np.random.rand(num_samples,1)
    source_range = 1 + 3*np.random.rand(num_samples,1)
    source_loc = np.concatenate((source_depth, source_range),axis=-1)

    np.random.seed(23)
    source_depth_test = 30 + 140*np.random.rand(num_samples_test,1)
    source_range_test = 1 + 3*np.random.rand(num_samples_test,1)
    source_loc_test = np.concatenate((source_depth_test, source_range_test),axis=-1)
      
    #snr = xvar[counter]
    signals_sim = []
    #num_samples = source_loc.shape[0]
   # env_perturb_factor_test = np.ones(num_samples_test)
    sp = 0
    rand_val = np.random.rand(num_samples)
    rand_test = np.random.rand(num_samples)
    env_perturb_factor = (1-xvar[counter]*rand_val)
    env_perturb_factor_test = (1-xvar[counter]*rand_test)
    signals_sim = gen_measurements(params, source_loc, sp, snr, env_perturb_factor, num_samples)
    signals_sim_test = gen_measurements(params, source_loc_test, sp, snr, env_perturb_factor_test, num_samples_test)
       
    loc_pred = mfp(params, env_perturb_factor, signals_sim, num_samples)
    
    ##UQ metric caluclation via traditional conf intervals and CP intervals
    dim_cnst = [0.01,5]
    calibration = True
    if calibration:
        source_loc_cvt = np.multiply(source_loc, [1,1000])
        mu_pred = np.mean(loc_pred,axis=1)
        std_pred = np.std(loc_pred,axis=1)
        std_pred[:,0] = np.where(std_pred[:,0]<dim_cnst[0], dim_cnst[0], std_pred[:,0])
        std_pred[:,1] = np.where(std_pred[:,1]<dim_cnst[1], dim_cnst[1], std_pred[:,1])
        for ind in range(2):
            n = num_samples
            y_valid = source_loc_cvt[:,ind].reshape((num_samples,1))
            #y_valid_pred = np.mean(loc_pred,axis=-1).reshape((num_samples,1))
            scores = np.abs(y_valid[:,0] - mu_pred[:,ind])/std_pred[:,ind]
            q_level = np.ceil((n+1)*(1-alpha))/n
            qhat[counter,ind] = np.quantile(scores, q_level)
    
    testing = True
    if testing: 
       # params['num_MC_runs'] = 1
        loc_pred_test = mfp(params, env_perturb_factor_test, signals_sim_test, num_samples_test)
        source_loc_cvt_test = np.multiply(source_loc_test, [1,1000])
        mu_pred = np.mean(loc_pred_test,axis=1)
        std_pred = np.std(loc_pred_test,axis=1)
        std_pred[:,0] = np.where(std_pred[:,0]<dim_cnst[0], dim_cnst[0], std_pred[:,0])
        std_pred[:,1] = np.where(std_pred[:,1]<dim_cnst[1], dim_cnst[1], std_pred[:,1])
        
        for ind in range(2):
            pt = 0
            temp_noncf = []
            for s in range(num_samples_test):
                scale_ind=st.sem(loc_pred_test[s,:,ind]) + dim_cnst[ind]
                pred_sets_noncf = st.t.interval(1-alpha, params['num_MC_runs'], loc=mu_pred[s,ind], 
                                                scale=scale_ind)
                temp_noncf.append(pred_sets_noncf[1] - pred_sets_noncf[0])
                
                pt = pt + (source_loc_cvt_test[s,ind] > pred_sets_noncf[0] and (source_loc_cvt_test[s,ind] < pred_sets_noncf[1]))
            
            ##Dimension independence assumption for CP metric calculation  
            temp_noncf = np.asarray(temp_noncf)
            temp_cf = 2*qhat[counter,ind]*std_pred[:,ind]
            err_mfp[counter,ind] = np.mean(np.abs(source_loc_cvt_test[:,ind]-mu_pred[:,ind]))
            
            uq_noncf[counter,ind] = np.mean(temp_noncf)
            uq_cf[counter,ind] = np.mean(temp_cf)
            coverage_score_cf[counter,ind] = sum((source_loc_cvt_test[:,ind]> mu_pred[:,ind]-temp_cf/2) & 
                                              (source_loc_cvt_test[:,ind]< mu_pred[:,ind]+temp_cf/2))/num_samples_test
            coverage_score_noncf[counter,ind] = pt/num_samples_test
   
            





