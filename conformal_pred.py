#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:31:22 2022

@author: ikhurjekar
"""

import numpy as np
import scipy.stats as st
#from tensorflow.keras.utils import to_categorical

## Conformalize
#y_valid: calibration ground truth
#y_valid_pred: calibration predictions
#y_test: ground truth
#y_pred: predictions from the model
#alpha: error rate

def conformalize(y_valid, y_valid_pred, y_test_pred, model, alpha, doa_limit, mode):
    n = y_valid.shape[0]
    t = y_test_pred.shape[0]
        
    if mode == 'ensemble_reg_network' or mode == 'mcdropout_reg_network':
        if np.mean(y_valid_pred,axis=-1).shape != y_valid.shape:
            y_valid.reshape(np.mean(y_valid_pred,axis=-1).shape)
        std_pred = np.std(y_valid_pred,axis=-1)
        std_pred = np.where(std_pred == 0, 0.0001, std_pred)
        #std_pred = np.where(std_pred == 0, 1e-5, std_pred)
        scores = np.abs(y_valid.reshape(n,) - np.mean(y_valid_pred,axis=-1))/std_pred
        q_level = np.ceil((n+1)*(1-alpha))/n
        qhat = np.quantile(scores, q_level, method='higher')
        mu_pred = np.mean(y_test_pred,axis=-1)
        std_pred = np.std(y_test_pred,axis=-1)
        pred_int_lower = mu_pred - std_pred*qhat
        pred_int_higher = mu_pred + std_pred*qhat
        prediction_sets = (pred_int_lower, pred_int_higher)
        
    # elif mode == 'mdn':
    #     mu_pred = y_valid_pred[:,:n_sources]
    #     std_pred = y_valid_pred[:,n_sources:2*n_sources]
    #     mix_pred = np.zeros((y_valid_pred.shape[0],2))
    #     for i in range(y_valid_pred.shape[0]):
    #         mix_pred[i] = np.exp(y_valid_pred[i,2*n_sources:])*1/np.sum(np.exp(y_valid_pred[i,2*n_sources:]), 
    #                                                                     axis = -1)
    #     scores = np.zeros((n,n_sources)) 
    #     qhat = np.zeros((n_sources,1)) 
    #     for j in range(n_sources):
    #         scores[:,j] = np.abs(y_valid[:,j] - mu_pred[:,j])/std_pred[:,j]
    #         q_level = np.ceil((n+1)*(1-alpha))/n
    #         qhat[j] = np.quantile(scores[:,j], q_level, method='higher')   
        
    #     mu_pred = y_test_pred[:,:n_sources]
    #     std_pred = y_test_pred[:,n_sources:2*n_sources]
    #     mix_pred = np.zeros((y_test_pred.shape[0],2))
    #     for i in range(y_test_pred.shape[0]):
    #         mix_pred[i] = np.exp(y_test_pred[i,2*n_sources:])*1/np.sum(np.exp(y_test_pred[i,2*n_sources:]), 
    #                                                                    axis = -1)
    #     prediction_sets = np.zeros((t,2,n_sources))
    #     for j in range(n_sources):    
    #         prediction_sets[:,0,j] = mu_pred[:,j] - std_pred[:,j]*qhat[j]
    #         prediction_sets[:,1,j] = mu_pred[:,j] + std_pred[:,j]*qhat[j]
    #         #prediction_sets.append([mu_pred - std_pred*qhat[j], mu_pred + std_pred*qhat[j]])
            
        
    elif mode == 'quantile_reg_network':
        val_scores = np.maximum(y_valid_pred[:,0]-y_valid[:,0], y_valid[:,0] - y_valid_pred[:,1])
        #val_scores = np.minimum(y_valid[:,0] - y_valid_pred[:,0], y_valid_pred[:,1]-y_valid[:,0])
        qhat = np.quantile(val_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
        prediction_sets = [y_test_pred[:,0]-qhat, y_test_pred[:,1]+qhat]
    
    elif mode == 'clf_network':
        labels = np.argmax(y_valid,axis=1).reshape(y_valid.shape[0])
        scores = 1-y_valid_pred[:,labels[0]]
        q_level = np.ceil((n+1)*(1-alpha))/n
        qhat = np.quantile(scores, q_level, method='higher')
        prediction_sets = []
        for i in range(y_test_pred.shape[0]):
            prediction_sets.append(1*(y_test_pred[i:i+1,:] >= (1-qhat)))
            
    elif mode == 'gp_beamformer':
        scores = np.abs(y_valid.reshape(n,) - np.mean(y_valid_pred,axis=-1))/np.std(y_valid_pred,axis=-1)
        q_level = np.ceil((n+1)*(1-alpha))/n
        qhat = np.quantile(scores, q_level, interpolation ='higher')
        
        mu_pred = np.mean(y_test_pred,axis=-1)
        std_pred = np.std(y_test_pred,axis=-1)
        prediction_sets = (mu_pred - std_pred*qhat, mu_pred + std_pred*qhat)
        
    elif mode == 'gaussian_likelihood':
        y_valid_meanpred = y_valid_pred[:,:1]
        std_pred = np.exp(y_valid_pred[:,1:] + 0.0001).reshape(n,1)
        scores = np.abs(y_valid.reshape(n,1) - y_valid_meanpred.reshape(n,1))/std_pred
        q_level = np.ceil((n+1)*(1-alpha))/n
        qhat = np.quantile(scores, q_level, method='higher')
        
        y_test_meanpred = y_test_pred[:,:1]
        mu_pred = np.mean(y_test_meanpred,axis=1).reshape(t,1)
        std_pred = np.exp(y_test_pred[:,1:] + 0.0001).reshape(t,1)
        pred_int_lower = mu_pred - std_pred*qhat
        pred_int_higher = mu_pred + std_pred*qhat
        prediction_sets = (pred_int_lower, pred_int_higher)
        
        
    return prediction_sets

def conformal_testing(params_model, model_doa, x_test, y_test, y_valid, y_valid_pred, doa_limit, test_samples, alpha):
    runs = 1
    ensemble_size = params_model['n_ensemble']
    num_MC_runs = params_model['num_MC_runs']
    if params_model['mode'] == 'ensemble_reg_network':
        print('Evaluating Ensemble network')
        params_model['training'] = False
        for r in range(runs):    
            y_test_pred = np.zeros((test_samples,ensemble_size))
            for mm in range(ensemble_size):
                y_test_pred[:,mm] = model_doa[mm].predict(x_test).reshape(test_samples,)
            y_valid_pred = np.clip(y_valid_pred, -doa_limit, doa_limit)
            y_test_pred = np.clip(y_test_pred, -doa_limit, doa_limit)
            mu_pred = np.mean(y_test_pred, axis=-1)
            pred_sets = conformalize(y_valid, y_valid_pred, 
                     y_test_pred, model_doa, alpha, doa_limit, mode = params_model['mode'])
            pred_sets = np.asarray(pred_sets)
            pred_sets = np.clip(pred_sets, -doa_limit, doa_limit)
            predwidth_cf = pred_sets[1] - pred_sets[0]
            uq_metric_cf = np.mean(predwidth_cf)
            coverage_score_cf = ((y_test[:,0] > pred_sets[0,:]) & 
                                   (y_test[:,0] < pred_sets[1,:])).sum()/test_samples
        error = np.abs(mu_pred.reshape((test_samples,1)) - y_test.reshape(test_samples,1))
        mae = np.mean(error)
        pt = 0
        predwidth_noncf = np.zeros((y_test.shape[0],1))
        pred_set = []
        for s in range(y_test.shape[0]):
            pred_sets_noncf = st.t.interval(1-alpha, int(y_test_pred.shape[1]-1),
                                loc=np.mean(y_test_pred[s,:]), scale=(st.sem(y_test_pred[s,:]+0.0001)))
            pred_sets_noncf = np.clip(pred_sets_noncf, -doa_limit, doa_limit)
            pred_set.append(pred_sets_noncf)
            predwidth_noncf[s] = pred_sets_noncf[1] - pred_sets_noncf[0]
            pt = pt + (y_test[s,0] > pred_sets_noncf[0] and (y_test[s,0] < pred_sets_noncf[1]))
        coverage_score_noncf = pt/test_samples
        uq_metric_noncf = np.mean(predwidth_noncf)
        
    # params_model['mode'] = 'mcdropout_reg_network'
    if params_model['mode'] == 'mcdropout_reg_network':
        print('Evaluating MC dropout')
        params_model['training'] = True
        for r in range(runs):    
            y_test_pred = np.zeros((test_samples,num_MC_runs))
            for mm in range(num_MC_runs):
                y_test_pred[:,mm] = model_doa.predict(x_test).reshape(test_samples,)
            y_valid_pred = np.clip(y_valid_pred, -doa_limit, doa_limit)
            y_test_pred = np.clip(y_test_pred, -doa_limit, doa_limit)
             
            mu_pred = np.mean(y_test_pred, axis=-1)
            pred_sets = conformalize(y_valid, y_valid_pred, 
                     y_test_pred, model_doa, alpha, doa_limit, params_model['mode'])   
            pred_sets = np.asarray(pred_sets)
            pred_sets = np.clip(pred_sets, -doa_limit, doa_limit)
            predwidth_cf = pred_sets[1] - pred_sets[0]
            uq_metric_cf = np.mean(predwidth_cf)
            coverage_score_cf = ((y_test[:,0] > pred_sets[0]) & 
                               (y_test[:,0] < pred_sets[1])).sum()/test_samples
            #err.append(np.abs(mu_pred.reshape((test_samples,1)) - y_test.reshape(test_samples,1)))
        error = np.abs(mu_pred.reshape((test_samples,1)) - y_test.reshape(test_samples,1))
        mae = np.mean(error) 
        pt, tempuq, pred_set = 0, [], []
        for s in range(y_test.shape[0]):
            pred_sets_noncf = (st.t.interval(1-alpha, y_test_pred.shape[1]-1,
                                  loc=np.mean(y_test_pred[s,:]), scale=st.sem(y_test_pred[s,:])))
            pred_set.append(np.clip(pred_sets_noncf, -doa_limit, doa_limit))
            tempuq.append(pred_sets_noncf[1] - pred_sets_noncf[0])
            pt = pt + (y_test[s,0] > pred_sets_noncf[0] and (y_test[s,0] < pred_sets_noncf[1]))
        coverage_score_noncf = pt/test_samples
        uq_metric_noncf = np.mean(np.asarray(tempuq))
        predwidth_noncf = np.asarray(tempuq)   

    if params_model['mode'] == 'quantile_reg_network':
        print('Evaluating quantile regression')
        params_model['training'] = False
        for r in range(runs):
            y_valid_pred = np.clip(y_valid_pred, -doa_limit, doa_limit)
            y_test_pred = np.zeros((test_samples,2))
            y_test_pred = model_doa.predict(x_test)
            y_test_pred = np.clip(y_test_pred, -doa_limit, doa_limit)
            mu_pred = np.mean(y_test_pred, axis=-1)
            pred_sets = conformalize(y_valid, y_valid_pred, 
                    y_test_pred, model_doa, alpha, doa_limit, mode = params_model['mode'])
            pred_sets = np.asarray(pred_sets)
            pred_sets = np.clip(pred_sets, -doa_limit, doa_limit)
            predwidth_cf = pred_sets[1] - pred_sets[0]
            uq_metric_cf = np.mean(predwidth_cf)
            coverage_score_cf = ((y_test[:,0] > pred_sets[0]) & 
                             (y_test[:,0] < pred_sets[1])).sum()/test_samples
        error = np.abs(mu_pred.reshape((test_samples,1)) - y_test.reshape(test_samples,1))
        mae = np.mean(error)
        uq_metric_noncf = np.mean(y_test_pred[:,1] - y_test_pred[:,0])
        coverage_score_noncf = ((y_test[:,0] > y_test_pred[:,0]) & 
                                  (y_test[:,0] < y_test_pred[:,1])).sum()/test_samples
        predwidth_noncf = y_test_pred[:,1] - y_test_pred[:,0]
    
        
    if params_model['mode'] == 'gaussian_likelihood':    
        print('Evaluating Gaussian likelihood')
        for r in range(runs):
            y_valid_pred = np.clip(y_valid_pred, -doa_limit, doa_limit)
            y_test_pred = np.zeros((test_samples,2))
            y_test_pred = model_doa.predict(x_test)
            #y_test_pred = np.clip(y_test_pred, -doa_limit, doa_limit)
            mu_pred = y_test_pred[:,0]
            pred_sets = conformalize(y_valid, y_valid_pred, 
                     y_test_pred, model_doa, alpha, doa_limit, params_model['mode'])   
            pred_sets = np.asarray(pred_sets)
            pred_sets = np.clip(pred_sets, -doa_limit, doa_limit)
            predwidth_cf = pred_sets[1] - pred_sets[0]
            uq_metric_cf = np.mean(predwidth_cf)
            coverage_score_cf = ((y_test[:,0] > pred_sets[0]) & 
                               (y_test[:,0] < pred_sets[1])).sum()/test_samples
            #err.append(np.abs(mu_pred.reshape((test_samples,1)) - y_test.reshape(test_samples,1)))
        error = np.abs(mu_pred.reshape((test_samples,1)) - y_test.reshape(test_samples,1))
        mae = np.mean(error) 
        pt, tempuq, pred_set = 0, [], []
        for s in range(y_test.shape[0]):
            scale_adj = np.exp(y_test_pred[s,1:]+0.00001)
            pred_sets_noncf = st.norm.interval(1-alpha, 
                                  loc=np.mean(y_test_pred[s,0]), scale=scale_adj)
            pred_set.append(np.clip(pred_sets_noncf, -doa_limit, doa_limit))
            tempuq.append(pred_sets_noncf[1] - pred_sets_noncf[0])
            pt = pt + (y_test[s,0] > pred_sets_noncf[0] and (y_test[s,0] < pred_sets_noncf[1]))
        coverage_score_noncf = pt/test_samples
        uq_metric_noncf = np.mean(np.asarray(tempuq))
        predwidth_noncf = np.asarray(tempuq)   
        
    doa_range_illustration = False
    if doa_range_illustration: 
         import matplotlib.pyplot as plt
         pred_sets = np.asarray(pred_sets).T
        # pred_set = np.asarray(pred_set)
         
         plt.rcParams.update({'font.size': 23})
         ind = np.argsort(y_test[:,0])
         plt.figure()
         #plt.plot(y_test[ind,0], mu_pred[ind], label = 'Predicted DOA')
         plt.plot(y_test[ind,0], y_test[ind,0], 'k--', linewidth = 2.5, label = 'True DOA')
         plt.fill_between(y_test[ind,0], pred_sets[ind,0], 
                           pred_sets[ind,1], label = 'CP interval')
         plt.xlim([0,85])
         plt.ylim([0,90])
         plt.xlabel('DOA true value$^\circ$')
         plt.ylabel('DNN-QR prediction ($^\circ$)')
         plt.grid()
         plt.legend(prop={'size': 16}, loc = 'best')
         
         # plt.figure()
         # plt.rcParams.update({'font.size': 23})
         # ind = np.argsort(y_test[:,0])
         # plt.figure()
         #  #plt.plot(y_test[ind,0], mu_pred[ind], label = 'Predicted DOA')
         # plt.plot(y_test[ind,0], y_test[ind,0], 'k--', linewidth = 2.5, label = 'True DOA')
         # plt.fill_between(y_test[ind,0], pred_set[ind,0], 
         #                  pred_set[ind,1], label = 'Conf. tnterval')
         # plt.xlim([0,85])
         # plt.ylim([0,90])
         # plt.xlabel('DOA true value$^\circ$')
         # plt.ylabel('DNN-QR prediction ($^\circ$)')
         # plt.grid()
         # plt.legend(prop={'size': 16}, loc = 'best')
   
        
    return mae, uq_metric_cf, coverage_score_cf, uq_metric_noncf, coverage_score_noncf 


   