#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:06:46 2022

@author: ikhurjekar
"""
import numpy as np
from beamformer import plane_waves, scm
from epscont import epscont
import random
from scipy.stats import norm
 
## Define acoustic data parameters
def param_data(): 
    params = {}
    params['frequency'] = 200
    params['speed'] = 1500
    params['wavelength'] = 1500/200
    params['num_sources'] = 2
    params['num_sensors'] = 20
    params['num_sensors_gp'] = 50 
    params['kernel'] = 'rbf'
    params['source_amplitude'] = None
    params['num_snapshots'] = 50
    params['array_spacing'] = 0.5
    params['distance_sensors'] = params['array_spacing']*params['wavelength']   
    params['SNR'] = 10
    params['snrvar'] = False
    params['interference_var'] = True
   # params['gain_var'] = True
    #params['num_MC_runs'] = 1
    params['sensor_perturb'] = 0
    params['wavelength_distort'] = 0.05
    params['interference_level'] = 1
    params['gainvar_limit'] = 0
    return params

## Define training model parameters
def param_model():
    params = {}
    params['dropout_rate'] = 0.2
    params['training'] = False
    params['input_dim'] = 20*21
    params['bsize'] = 16
    params['epochs'] = 200
    params['num_MC_runs'] = 100
    params['num_ensemble'] = 8
    params['num_comp'] = 2
    #params['mode'] = 'mcdropout_reg_network'
    return params

##Create dataset for aprabola example
def create_rnddata(n_samples, input_limit, noise_std):
    x=np.random.uniform(0,input_limit,(n_samples,1))
    #zs=np.random.uniform(-10,10,(n_samples,1))
    noise=np.random.normal(0,noise_std,(n_samples,1))
    y = np.square(x)  + x - 20 + noise
    #y = 3*x + 5 + noise
    return y,x

## Create plane-wave acoustic data (input format for DNN -- real and imag part concatenated)    
def create_wavedata(params, angles):
    sigma_noise=1/np.sqrt(params['num_sensors'])
    distance_sensors = params['distance_sensors']
    interference_level = params['interference_level']
    lambdavar = 10
    epsilon = 0
    SNRoffset = 0
    snrvar = params['snrvar']
    snr = params['SNR']
    noise,mask = epscont((params['num_sensors'],params['num_snapshots'],len(angles)),
                sigma=sigma_noise,epsilon=epsilon,lambdavar=lambdavar,return_mask=True)
    data_shape = angles.shape[0]
    sensor_locations_true = (np.arange(params['num_sensors']) - (params['num_sensors'] - 1) /2) * distance_sensors
   # distance_sensors_gp = ( (params['num_sensors']- 1) / (params['num_sensors_gp'] - 1) * distance_sensors)
   # sensor_locations_gp = ( np.arange(params['num_sensors_gp']) - (params['num_sensors_gp'] - 1) / 2) * distance_sensors_gp  
    signals_scm = np.zeros((data_shape, 2*params['num_sensors']*params['num_sensors']))
   # signals_gp_scm = np.zeros((data_shape, 2*params['num_sensors_gp']*params['num_sensors_gp']))
    signal_noisefree = np.zeros((params['num_sensors'], params['num_snapshots']), dtype=np.complex_)
    signal_interference = np.zeros((params['num_sensors'], params['num_snapshots']),dtype=np.complex_)
    pf_values = []
    #np.random.seed(23)
   # int_sector = params['interference_sector']
    gain_sig = 0
    for i in range(angles.shape[0]):
       # gain_sig = params['gainvar_limit']*np.random.rand()
        # if params['interference_var']:
        #     interference_level = params['interference_limit']*np.random.rand()
        # else:
        #     interference_level = params['interference_limit']
       # if params['gain_var']:
       #     gain_sig = params['gainvar_limit']*np.random.rand()
       # else:
       #     gain_sig = params['gainvar_limit']
        
        source_phase = np.exp(1j * 2 * np.pi * np.random.rand(params['num_sources'],params['num_snapshots'])) 
        source_locations = source_phase[:,:]      #determistic source
        source_angle = angles[i]
        #if int_sector == -1:
      #  interference_angle = np.asarray([-90 + 180*np.random.rand()])*np.pi/180
        #else:    
        #interference_angle = np.asarray([-90 + 45*int_sector + 45*np.random.rand()])*np.pi/180
       # interference_phase = np.exp(1j * 2 * np.pi * np.random.rand(params['num_sources'], params['num_snapshots'])) 
       # interference_locations = interference_level * interference_phase[:,:]  
        
        if snrvar == True:
            snr = 30*np.random.rand(1)
        else:
            snr =  -3 + 6*np.random.rand(1) + params['SNR']
            
        signal_noisefree = plane_waves(sensor_locations_true, source_locations, 
                                       source_angle, params['wavelength'], gain_sig)
       # signal_interference = plane_waves(sensor_locations_true, interference_locations,
       #                                  interference_angle, params['wavelength'])
        rnl =  10 ** (-snr / 20) * np.linalg.norm(signal_noisefree,'fro')/np.sqrt(params['num_snapshots']) #deterministic source
        
        temp = signal_noisefree + noise[:,:,i]*rnl
        signals = np.concatenate((np.real(scm(temp)), 
                                np.imag(scm(temp))), axis = 1)
        signals_scm[i, :] = signals.flatten()

    return signals_scm, angles*180/np.pi




def create_multiwavedata(params, angle, amp, mdn_flag):
    sigma_noise=1/np.sqrt(params['num_sensors'])
    sigma_adv=2/np.sqrt(params['num_sensors'])
    distance_sensors = params['distance_sensors']
    interference_level = params['interference_level']
    #amp_level = params['amp_level']
  #  int_sector = params['interference_sector']
    lambdavar = 10
    epsilon = 0
    snrvar = params['snrvar']
    snr = params['SNR']
    ns = 300
    noise,mask = epscont((params['num_sensors'],params['num_snapshots'],1),
                sigma=sigma_noise,epsilon=epsilon,lambdavar=lambdavar,return_mask=True)
  
   # data_shape = len(angle)
    sensor_locations_true = (np.arange(params['num_sensors']) - (params['num_sensors'] - 1) /2) * distance_sensors
   # distance_sensors_gp = ( (params['num_sensors']- 1) / (params['num_sensors_gp'] - 1) * distance_sensors)
   # sensor_locations_gp = ( np.arange(params['num_sensors_gp']) - (params['num_sensors_gp'] - 1) / 2) * distance_sensors_gp  
    #signals_scm = np.zeros((data_shape, 2*params['num_sensors']*params['num_sensors']))
   # signals_gp_scm = np.zeros((dxata_shape, 2*params['num_sensors_gp']*params['num_sensors_gp']))
    signal_noisefree = np.zeros((params['num_sensors'], params['num_snapshots']), dtype=np.complex_)
    signal_interference = np.zeros((params['num_sensors'], params['num_snapshots']),dtype=np.complex_)
    
  #  pf_values = []
    #np.random.seed(23)s
    source_angle = np.array(angle)
    gain_sig = 0
    source_phase = np.exp(1j * 2 * np.pi * np.random.rand(len(angle),params['num_snapshots'])) 
    amps = np.ones(params['num_sources'])
    #amps = np.asarray([amp, 1])
    source_locations = np.multiply(source_phase[:,:],amps.reshape(source_angle.shape[0],1)) 
     #determistic source
    #source_locations = source_phase[:,:]
    
   # if int_sector == -1:
    interference_angle_1 = np.asarray([0 + 45*np.random.rand()])*np.pi/180
   # else:    
    # interference_angle_1 = np.asarray([angle[-1]])
   # interference_angle_2 = np.asarray([90*np.random.rand()])*np.pi/180
    interference_phase_1 = np.exp(1j * 2 * np.pi * np.random.rand(1, params['num_snapshots'])) 
   # interference_phase_2 = np.exp(1j * 2 * np.pi * np.random.rand(1, params['num_snapshots']))
    interference_locations_1 = interference_level * interference_phase_1[:,:]  
    #interference_locations_2 = 2*interference_level * interference_phase_2[:,:]  
     
    if snrvar == True:
        snr = 20*np.random.rand(1)
    else:
        snr =  -3 + 6*np.random.rand(1) + params['SNR']
         
    signal_noisefree = plane_waves(sensor_locations_true, source_locations, 
                                    source_angle, params['wavelength'], gain_sig)
    signal_interference_1 = plane_waves(sensor_locations_true, interference_locations_1,
                                      interference_angle_1, params['wavelength'], gain_sig)
  #  signal_interference_2 = plane_waves(sensor_locations_true, interference_locations_2,
  #                                    interference_angle_2, params['wavelength'], gain_sig)
    rnl =  10 ** (-snr / 20) * np.linalg.norm(signal_noisefree,'fro')/np.sqrt(params['num_snapshots']) #deterministic source
    temp = signal_noisefree 
    
    adv_cp = False
    if adv_cp == True:
        adv,mask = epscont((params['num_sensors'],params['num_snapshots'],ns),
                    sigma=sigma_noise,epsilon=epsilon,lambdavar=lambdavar,return_mask=True)
        temp = np.repeat(temp[:, :, np.newaxis], ns, axis=2)
        temp_adv = np.add(temp, adv*rnl)
        signal_scm = np.zeros(( 2*params['num_sensors']*params['num_sensors'], ns))
        for i in range(ns):
            signals = np.concatenate((np.real(scm(temp_adv[:,:,i])), 
                                      np.imag(scm(temp_adv[:,:,i]))), axis = 1)
            signal_scm[:,i] = signals.flatten()
    #if mdn_flag == 'train':
    temp = temp + noise[:,:,0]*rnl 
   # else:
    #    temp = temp + noise[:,:,0]*rnl + signal_interference_1
    temp_triu = scm(temp)[np.triu_indices(params['num_sensors'])]
    signal_scm = np.concatenate((np.real(temp_triu), 
                                     np.imag(temp_triu)))
        #signal_scm = signals.flatten()

    return signal_scm



def data_gen_mdn(n_samples, doa_range, doa_limit, params, mdn_flag):
    ##Inputs: For training, Create k total copies of input which has 'k' DOAs and store in x
    ## Inputs: For testing, just store directly
    ##Outputs: For training, flatten and store angle in radians in 1 1d array
    ##Outputs: For testing, store directly as 1d array (have to change for non-uniform # DOA's in each sample)
    list_input = []
    source_limit = [params['num_sources'],params['num_sources']+1]
    ##Fix every sample to have k sources. To be varied later on.
    n_sources_list = np.random.randint(source_limit[0],source_limit[1], 
                                       size = n_samples)
    angles  = []
    amp_level = []
    if mdn_flag != 'test':
        amp_level = np.where(np.random.rand(n_samples)>0.5, 1, 0.2)
       # amp_level = 0.05+ 0.95*np.random.rand(samples)
    else:
        amp_repeat = 200*[0.2]+200*[1]
        amp_level = int(n_samples/400)*amp_repeat
 
    for i in range(n_samples):
        if mdn_flag != 'test': 
            angs_temp = np.random.random(n_sources_list[i])
            angs_temp = np.sort(angs_temp)
            temp = []
            sectors = params['num_sources']
            for kk in np.arange(len(angs_temp)):
                temp.append((-doa_limit + doa_range*(1/sectors)*kk + 
                                  doa_range*(1/sectors)*angs_temp[kk])*3.14/180)
            angles.append(temp)
        else:
            temp = [-0.9594, 1.064]
            angles.append(temp)
        
        # if mdn_flag == 'test': 
        #     #for j in range(n_sources_list[i]):
        #      #   list_input.append(create_multiwavedata(params, temp, amp_level[i], mdn_flag))
        # else:
        #    list_input.append(create_multiwavedata(params, temp, amp_level[i], mdn_flag))
        
        list_input.append(create_multiwavedata(params, temp, amp_level[i], mdn_flag))
    
    adv_cp = False
    x = np.asarray(list_input)
    
    if mdn_flag == 'train':        
        #y = np.asarray([item for sublist in angles for item in sublist]).reshape(k, 1)
        y = np.asarray(angles)
    elif mdn_flag == 'val' and adv_cp == True:
        x = np.moveaxis(x, 1, 2)
        x = np.reshape(x,(-1, x.shape[-1]))
        y = np.asarray(angles)
        y = np.repeat(angles,300)
    else:
        y = np.asarray(angles)
        
    return x,y, amp_level



def scaler_func(x, scaling):
    dim = x.shape
    if scaling == 'minmax':
        for i in range(dim[0]):
            temp = x[i]
            x[i] = (temp - temp.min(axis=0)) / (temp.max(axis=0) - temp.min(axis=0))
    elif scaling == 'standard':
        for i in range(dim[0]):
            temp = x[i]
            x[i] = (temp - temp.mean(axis=0)) / (temp.std(axis=0))
    else:
        for i in range(dim[0]):
            temp = x[i]
            norm = (np.sqrt(np.sum(np.square(temp),axis=0))).reshape(1,dim[-1])
            x[i] = temp/norm
    return x


def correlation(x,y):
    corr = np.cov(x,y)/(np.std(x)*np.std(y))
    return corr[0,1]


def gmm_prob(parameters_mdn, n_comp, doa_limit):
    samples = parameters_mdn.shape[0]
    mu_pred = parameters_mdn[:,:n_comp]*(180/np.pi)
    std_pred = parameters_mdn[:,n_comp:2*n_comp]*(180/np.pi)
    std_pred = np.clip(std_pred, -doa_limit, doa_limit)
    mix_pred = np.zeros((samples,n_comp))
    doa_range = np.arange(-doa_limit, doa_limit,1)
    prob_doa = np.zeros((samples, doa_range.shape[0], n_comp))
    for i in range(samples):
       # mix_pred[i] = np.exp(parameters_mdn[i,2*n_comp:])*1/np.sum(np.exp(parameters_mdn[i,2*n_comp:]), 
       #                                                             axis = -1)
        temp = np.zeros((doa_range.shape[0], n_comp))
        for j in range(n_comp):
           # temp[:,j]= mix_pred[i,j]*norm.pdf(doa_range, mu[i,j], std_pred[i,j])
            #temp = norm.pdf(doa_range, mu_pred[i,j], std_pred[i,j])
            #temp_normalized = (temp-np.min(temp))/(np.max(temp) - np.min(temp) + 1e-10)
            prob_doa[i,:,j] = norm.pdf(doa_range, mu_pred[i,j], std_pred[i,j])
       # prob_doa[i] = np.sum(temp,axis=-1)
    
    return prob_doa


def calculate_mixprob(model, x, n_comp):
    cc=model.predict(x)
    cc=cc[:,2*n_comp:]
    den = np.sum(np.exp(cc),axis=-1)
    num = np.exp(cc)
    pis = np.divide(num,den.reshape(x.shape[0],1))
    return pis



