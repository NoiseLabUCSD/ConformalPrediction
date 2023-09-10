#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:19:20 2022

@author: ikhurjekar
"""

import numpy as np
import tensorflow as tf
import mdn
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPooling1D, Conv1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, initializers, metrics
#from tensorflow.keras import backend as K
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from losses import gaussian_nll, QuantileLoss, qloss
from helper_funcs import create_multiwavedata, param_data, param_model, scaler_func, data_gen_mdn
from conformal_pred_gmm import conformal_testing


## Create DL model
def create_model(params):
    inputs = Input(shape=(params['input_dim']))
    tr = params['training']
    if params['mode'] == 'ensemble_reg_network':
        loss_func = tf.keras.losses.MeanSquaredError()
        params['output_dim'] = 1 
        params['training'] = False
    elif params['mode'] == 'mcdropout_reg_network':
        loss_func = tf.keras.losses.MeanSquaredError()
        params['output_dim'] = 1
        params['training'] = True
    elif params['mode'] == 'mdn':
        n_dim = 1
        n_comp = params['num_comp']
        loss_func = mdn.get_mixture_loss_func(n_dim, n_comp) 
        #params['output_dim'] = 2
        params['training'] = False
        #loss_func = gaussian_nll
    elif params['mode'] == 'quantile_reg_network':
        perc_points = [0.05, 0.95]
        params['output_dim'] = len(perc_points)
        loss_func = QuantileLoss(perc_points)
        params['training'] = False
    elif params['mode'] == 'clf_network':
         loss_func = tf.keras.losses.CategoricalCrossentropy() 
         params['training'] = False
    else:
         loss_func = gaussian_nll
         params['output_dim'] = 2
   # layer = Conv1D(filters = 12, kernel_size = 3, activation = 'relu', 
   #                 kernel_initializer=initializers.RandomNormal(stddev=0.01))(inputs)   
   # layer = MaxPooling1D(pool_size=(2))(layer) 
    #layer = Flatten()(layer)
    # kernel_regularizer=regularizers.L2(l2=1e-9)
    layer = Dense(800, activation='relu')(inputs)
    layer = Dropout(params['dropout_rate'])(layer)
    layer = Dense(300, activation='relu')(layer)
    layer = Dropout(params['dropout_rate'])(layer)
    layer = Dense(100, activation='relu')(layer)
   # layer = Dropout(params['dropout_rate'])(layer)
    layer_pre = Dense(30, activation='relu')(layer)
   # if params['mode'] == 'clf_network':
   #     outputs = Dense(params['output_dim'], activation = 'softmax')(layer_pre, training = tr)
    if params['mode'] == 'mdn':
        outputs = mdn.MDN(n_dim, n_comp)(layer_pre)
    else:
        outputs = Dense(params['output_dim'], activation = 'linear')(layer_pre)
    model = Model(inputs = inputs, outputs = outputs)
    adam = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, 
                                    epsilon = None, decay=1e-6, amsgrad=False)

    model.compile(loss = loss_func, optimizer = adam, metrics=['accuracy'])
    model.summary()    
    return model


if __name__ == "__main__":
    ##Instantiate
    params_data = param_data()
    params_model = param_model()
    n_samples = 10000
    val_samples = 1000
    test_samples = 1200
    doa_limit = 90
    doa_range = 2*doa_limit
    alpha = 0.1
    params_data['snrvar'] = False
    
    n_sources = params_data['num_sources']
    params_data['source_amplitude'] = np.asarray(params_data['num_sources']*[1])
    d_sensors = params_data['distance_sensors']

    
    # params_data['interference_var'] = True
    # angles = (-doa_limit + doa_range*np.random.rand(n_samples,params_data['num_sources']))*np.pi/180
    # x_train, y_train, pfvalues_train = create_wavedata(params_data, angles)
    # x_train = scaler_func(x_train, scaling = 'standard')
    
    # angles = (-doa_limit + doa_range*np.random.rand(val_samples,params_data['num_sources']))*np.pi/180
    # x_valid, y_valid, pfvalues_valid = create_wavedata(params_data, angles)
    # x_valid = scaler_func(x_valid, scaling = 'standard')
    
   #### Individual models training on cumulative data
   
   ##MC-dropout 
  # num_MC_runs = params_model['num_MC_runs']
    # params_model['mode'] = 'mcdropout_reg_network'
    # params_model['training'] = True
    # params_model['dropout_rate'] = 0.25
    # params_model['epochs'] = 100
    # model_doa_mcd = create_model(params_model)
    # history = model_doa_mcd.fit(x_train, y_train, batch_size = params_model['bsize'], 
    #           epochs = params_model['epochs'])
    # y_valid_pred_mcd = np.zeros((val_samples, num_MC_runs))
    # for mm in range(num_MC_runs):
    #     y_valid_pred_mcd[:,mm] = model_doa_mcd.predict(x_valid).reshape(val_samples,) 
       
    # ## Ensemble model
   # ensemble_size = params_model['num_ensemble']
   #  model_doa_ensemble = []
   #  params_model['mode'] = 'ensemble_reg_network'
   #  params_model['training'] = False
   #  params_model['epochs'] = 100
   #  for i in range(ensemble_size):
   #      model_doa = create_model(params_model)
   #      history = model_doa.fit(x_train, y_train, batch_size = params_model['bsize'], 
   #                epochs = params_model['epochs'])
   #      model_doa_ensemble.append(model_doa)
   #  y_valid_pred_de = np.zeros((val_samples, ensemble_size))
   #  for i in range(ensemble_size):
   #      y_valid_pred_de[:,i] = model_doa_ensemble[i].predict(x_valid).reshape(val_samples,) 
    
        
   # ## Deep quantile regression
    # params_model['mode'] = 'quantile_reg_network'
    # params_model['training'] = False
    # params_model['dropout_rate'] = 0.25
    # params_model['epochs'] = 100
    # model_doa_qr = create_model(params_model)
    # history = model_doa_qr.fit(x_train, y_train, batch_size = params_model['bsize'], 
    #           epochs = params_model['epochs'])
    # y_valid_pred_qr = np.zeros((val_samples, 2))
    # y_valid_pred_qr = model_doa_qr.predict(x_valid)
    
    #Gaussian likelihood
    # params_model['mode'] = 'gaussian_likelihood'
    # params_model['training'] = False
    # params_model['dropout_rate'] = 0.2
    # params_model['epochs'] = 400
    # model_doa_gl= create_model(params_model)
    # history = model_doa_gl.fit(x_train, y_train, batch_size = params_model['bsize'], 
    #         epochs = params_model['epochs'])
    
    
    ##MDN training
   # params_data['interference_sector'] = -1
  #  params_data['interference_level'] = 0.15
  #  x_train, y_train = data_gen_mdn(n_samples, doa_range, doa_limit, params_data, mdn_flag = 'train')
    #y_train = y_train*180/np.pi
  #  x_train = scaler_func(x_train, scaling = 'standard')
   
   # x_valid, y_valid = data_gen_mdn(val_samples, doa_range, doa_limit, params_data, mdn_flag = 'val')
    #y_valid = y_valid*180/np.pi
   # x_valid = scaler_func(x_valid, scaling = 'standard')
    
    # params_model['mode'] = 'mdn'
    # params_model['num_sources'] = params_data['num_sources']
    # model_doa_mdn = create_model(params_model)
    # history = model_doa_mdn.fit(x_train, y_train, batch_size = params_model['bsize'], 
    #         epochs = params_model['epochs'])
  #  y_valid_pred_mdn = model_doa_mdn.predict(x_valid)
    
    
    ###Train and test on each uncertainty level separately (to reproduce results in paper)
    
    SNR_list = np.arange(-20,21,10)
    perturb_factor = [0.2, 0.4, 0.6, 0.8, 1.0]
    int_level = [0.5]
    gainvar_limit = [0.05,0.1,0.15,0.2,0.25]
    xvar = int_level
    runs = 1
    uq_metric_cf_de = np.zeros((len(xvar),runs))
    uq_metric_noncf_de = np.zeros((len(xvar),runs))
    coverage_score_cf_de = np.zeros((len(xvar),runs))
    coverage_score_noncf_de = np.zeros((len(xvar),runs))
    uq_metric_cf_mcd = np.zeros((len(xvar),runs))
    uq_metric_noncf_mcd = np.zeros((len(xvar),runs))
    coverage_score_cf_mcd = np.zeros((len(xvar),runs))
    coverage_score_noncf_mcd = np.zeros((len(xvar),runs))
    uq_metric_cf_qr = np.zeros((len(xvar),runs))
    uq_metric_noncf_qr = np.zeros((len(xvar),runs))
    coverage_score_cf_qr = np.zeros((len(xvar),runs))
    coverage_score_noncf_qr = np.zeros((len(xvar),runs))
    uq_metric_cf_gl = np.zeros((len(xvar),runs))
    uq_metric_noncf_gl = np.zeros((len(xvar),runs))
    coverage_score_cf_gl = np.zeros((len(xvar),runs))
    coverage_score_noncf_gl = np.zeros((len(xvar),runs))
    
    uq_metric_cf_mdn = np.zeros((len(xvar),2))
    uq_metric_noncf_mdn = np.zeros((len(xvar),2))
    coverage_score_cf_mdn = np.zeros((len(xvar),2))
    coverage_score_noncf_mdn = np.zeros((len(xvar),2))
    err_mdn = np.zeros((len(xvar),2))
    
    err_de = np.zeros((len(xvar),1))
    err_mcd = np.zeros((len(xvar),1))
    err_qr = np.zeros((len(xvar),1))
    err_gl = np.zeros((len(xvar),1))
    qhat_scores = np.zeros((len(xvar),2))
    
    params_data['snrvar'] = False
    for ii in range(len(xvar)):
        ##Test condition init   -- toggle appropriately between uncertainty sources 
         np.random.seed(23)
         #params_data['SNR'] = xvar[ii]
         params_data['interference_level'] = xvar[ii]
       
        # params_data['gain_var'] = False
        # angles_train = (-doa_limit + 2*doa_limit*np.random.rand(n_samples,1))*(np.pi/180)
         x_train, y_train = data_gen_mdn(n_samples, doa_range, doa_limit, 
                                        params_data, mdn_flag = 'train')
         x_valid, y_valid = data_gen_mdn(val_samples, doa_range, doa_limit, 
                                        params_data, mdn_flag = 'val')
         x_test, y_test = data_gen_mdn(test_samples, doa_range, doa_limit, 
                                        params_data, mdn_flag = 'test')

         x_train = scaler_func(x_train, scaling = 'standard')
        # x_train = np.expand_dims(x_train, axis = -1)
         x_valid = scaler_func(x_valid, scaling = 'standard')
         #x_valid = np.expand_dims(x_valid, axis = -1)
         x_test = scaler_func(x_test, scaling = 'standard')
         #x_test = np.expand_dims(x_test, axis = -1)
         #y_train = (y_train)/3.14
         #y_valid = (y_valid)/3.14
         #y_test = (y_test)/3.14
         
         params_model['mode'] = 'mdn'
         model_doa_mdn = create_model(params_model)
         history = model_doa_mdn.fit(x_train, y_train, batch_size = params_model['bsize'], 
                epochs = params_model['epochs'])
         y_valid_pred_mdn = model_doa_mdn.predict(x_valid)
         error, uq_cf, coverage_cf, predint_cf, uq_noncf, coverage_noncf, predint_noncf, scores  = conformal_testing(params_model, 
                      model_doa_mdn, x_test, y_test, y_valid, y_valid_pred_mdn, doa_limit, 
                      test_samples, alpha)
         
         err_mdn[ii] = error[:,0]
         uq_metric_cf_mdn[ii] = uq_cf[:,0]
         uq_metric_noncf_mdn[ii] = uq_noncf[:,0]
         coverage_score_cf_mdn[ii] = coverage_cf[:,0]
         coverage_score_noncf_mdn[ii] = coverage_noncf[:,0]
         
         sector_coverage = False
         if sector_coverage == True:
             sectors = 4
             angles_val_sector, angles_test_sector, scores_sector, predints_sector = [], [], [], []
             for i in range(sectors):        
                 angles_valid = (-doa_limit + doa_range*(1/sectors)*i + 
                                  doa_range*(1/sectors)*np.random.rand(val_samples,1))*(np.pi/180)
                 x_valid, y_valid = data_gen_mdn(val_samples, doa_range, doa_limit, 
                                                             params_data, angles_valid, mdn_flag = 'val')
                 # y_valid = y_valid*180/np.pi
                 x_valid = scaler_func(x_valid, scaling = 'standard')
               
                
                  # ##Test data generation
                 angles_test = (-doa_limit + doa_range*(1/sectors)*i + 
                                  doa_range*(1/sectors)*np.random.rand(val_samples,1))*(np.pi/180)
                 x_test, y_test = data_gen_mdn(test_samples, doa_range, doa_limit, 
                                                params_data, angles_test, mdn_flag = 'test')
                  #y_test = y_test*180/np.pi
                 x_test = scaler_func(x_test, scaling = 'standard')
          
            
                 y_valid_pred_mdn = model_doa_mdn.predict(x_valid)
                 error, uq_cf, coverage_cf, predint_cf, uq_noncf, coverage_noncf, predint_noncf, scores  = conformal_testing(params_model, 
                            model_doa_mdn, x_test, y_test, y_valid, y_valid_pred_mdn, doa_limit, 
                            test_samples, alpha)
                 
                 angles_val_sector.append(angles_valid)
                 angles_test_sector.append(angles_test)
                 scores_sector.append(scores)
                 predints_sector.append(predint_cf)
         
         
         # params_model['mode'] = 'quantile_reg_network'
         # params_model['training'] = False
         # params_model['dropout_rate'] = 0.25
         # params_model['epochs'] = 100
         # model_doa_qr = create_model(params_model)
         # history = model_doa_qr.fit(x_train, y_train, batch_size = params_model['bsize'], 
         #          epochs = params_model['epochs'])
         # y_valid_pred_qr = np.zeros((val_samples, 2))
         # y_valid_pred_qr = model_doa_qr.predict(x_valid)
        
         # params_model['mode'] = 'quantile_reg_network'
         # params_model['training'] = False
         # error, uq_cf, coverage_cf, uq_noncf, coverage_noncf  = conformal_testing(params_model, model_doa_qr, 
         #              x_test, y_test, y_valid, y_valid_pred_qr, doa_limit, test_samples, alpha)
         # err_qr[ii] = error
         # uq_metric_cf_qr[ii] = uq_cf
         # uq_metric_noncf_qr[ii] = uq_noncf
         # coverage_score_cf_qr[ii] = coverage_cf
         # coverage_score_noncf_qr[ii] = coverage_noncf
        
         # model_doa_ensemble = []
         # params_model['mode'] = 'ensemble_reg_network'
         # params_model['training'] = False
         # params_model['epochs'] = 100
         # for i in range(ensemble_size):
         #     model_doa = create_model(params_model)
         #     history = model_doa.fit(x_train, y_train, batch_size = params_model['bsize'], 
         #              epochs = params_model['epochs'])
         #     model_doa_ensemble.append(model_doa)
         # y_valid_pred_de = np.zeros((val_samples, ensemble_size))
         # for i in range(ensemble_size):
         #     y_valid_pred_de[:,i] = model_doa_ensemble[i].predict(x_valid).reshape(val_samples,) 
      
         # params_model['mode'] = 'ensemble_reg_network'
         # params_model['training'] = False
         # error, uq_cf, coverage_cf, uq_noncf, coverage_noncf  = conformal_testing(params_model, model_doa_ensemble, 
         #            x_test, y_test, y_valid, y_valid_pred_de, doa_limit, test_samples, alpha)
         # err_de[ii] = error
         # uq_metric_cf_de[ii] = uq_cf
         # uq_metric_noncf_de[ii] = uq_noncf
         # coverage_score_cf_de[ii] = coverage_cf
         # coverage_score_noncf_de[ii] = coverage_noncf
         
     #    params_model['mode'] = 'mcdropout_reg_network'
     #    params_model['training'] = True
   #      error, uq_cf, coverage_cf, uq_noncf, coverage_noncf  = conformal_testing(params_model, model_doa_mcd, 
    #                    x_test, y_test, y_valid, y_valid_pred_mcd, doa_limit, test_samples, alpha)
    #     err_mcd[ii] = error
    #     uq_metric_cf_mcd[ii] = uq_cf
    #     uq_metric_noncf_mcd[ii] = uq_noncf
    #     coverage_score_cf_mcd[ii] = coverage_cf
    #     coverage_score_noncf_mcd[ii] = coverage_noncf
    
    









    
    
    
    
    
    