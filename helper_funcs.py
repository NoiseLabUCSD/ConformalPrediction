#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:06:46 2022

@author: ikhurjekar
"""
import numpy as np
from beamformer import plane_waves, scm
from epscont import epscont

## Define acoustic data parameters
def param_data(): 
    params = {}
    params['frequency'] = 200
    params['speed'] = 1500
    params['wavelength'] = 1500/200
    params['num_sources'] = None
    params['num_sensors'] = 20
    params['num_sensors_gp'] = 50 
    params['kernel'] = 'rbf'
    params['source_amplitude'] = None
    params['num_snapshots'] = 50
    params['array_spacing'] = 0.5
    params['distance_sensors'] = params['array_spacing']*params['wavelength']   
    params['SNR'] = 20
    params['snrvar'] = False
    params['interference_var'] = True
    params['gain_var'] = True
    #params['num_MC_runs'] = 1
    params['sensor_perturb'] = 0
    params['wavelength_distort'] = 0.05
    params['interference_limit'] = 0
    params['gainvar_limit'] = 0
    return params

## Define training model parameters
def param_model():
    params = {}
    params['dropout_rate'] = 0.15
    params['training'] = None
    params['input_dim'] = 800
    params['bsize'] = 16
    params['epochs'] = 200
    params['num_MC_runs'] = 100
    params['n_ensemble'] = 8
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
    interference_level = params['interference_limit']
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
    #interference_level = params['interference_limit']
    gain_sig = params['gainvar_limit']
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
        #interference_angle = np.asarray([-90+180*np.random.rand()])*np.pi/180
        #interference_angle = np.asarray([-90*np.pi/180])
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

    return signals_scm, angles*180/np.pi, pf_values


def data_processing_mdn(inputs, outputs):
    list_input, list_output = [], []
    k = 0
    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[1]):
            k = k + 1
            list_input.append(inputs[i,:])
            list_output.append(outputs[i][j])
            
    x = np.asarray(list_input).reshape(k, inputs.shape[1])
    y = np.asarray(list_output).reshape(k, 1)
    return x,y



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
    
