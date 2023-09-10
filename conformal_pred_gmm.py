#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:58:37 2023

@author: ishan
"""



import numpy as np
import scipy.stats as st
from helper_funcs import gmm_prob
from tensorflow.keras.utils import to_categorical

## Conformalize
#y_valid: calibration ground truth
#y_valid_pred: calibration predictions
#y_test: ground truth
#y_pred: predictions from the model
#alpha: error rate

def conformalize(y_valid, y_valid_pred, y_test_pred, model, alpha, doa_limit, n_sources, mode):
    n = y_valid.shape[0]
    t = y_test_pred.shape[0]
    q_level = np.ceil((n+1)*(1-alpha))/n
    cp_calc = 'likelihood'
    n_comp = n_sources 
    
    if mode == 'mdn':
        if cp_calc == 'likelihood':
            y_valid_pred[:,:n_sources] = np.clip(y_valid_pred[:,:n_sources], -np.pi/2, np.pi/2)
            y_test_pred[:,:n_sources] = np.clip(y_test_pred[:,:n_sources], -np.pi/2, np.pi/2)
            mu_pred = y_valid_pred[:,:n_sources]*180/np.pi
            std_pred = y_valid_pred[:,n_sources:2*n_sources]*180/np.pi
            scores = np.abs(mu_pred - y_valid[:,:n_sources]*180/np.pi)/std_pred
           # prob_doa = gmm_prob(y_valid_pred, n_comp, doa_limit)
           # scores = -prob_doa.reshape(n*2*doa_limit,n_comp)
           # scores = scores/np.linalg.norm(scores, axis=0)
            n_gmm = scores.shape[0]
            q_level = np.ceil((n_gmm+1)*(1-alpha))/n_gmm
            qhat = np.zeros((n_comp,))
            for j in range(n_comp):
                qhat[j] = np.quantile(scores[:,j], q_level, method='higher')  
           # prob_doa_test = gmm_prob(y_test_pred, n_comp, doa_limit)
            prediction_sets = np.zeros((y_test_pred.shape[0],2,n_comp))
            
            for i in range(y_test_pred.shape[0]):
                std = y_test_pred[i,n_sources:2*n_sources]*180/np.pi
                for j in range(n_comp): 
                    #temp = prob_doa_test[i,:,j]
                   # temp = np.argwhere(prob_doa_test[i,:,j]>-qhat[j])
                   # prediction_sets[i,0,j] = temp[0] - 89
                   # prediction_sets[i,1,j] = temp[-1] - 89
                    prediction_sets[i,0,j] = y_test_pred[i,j]*180/np.pi - std[j]*qhat[j]
                    prediction_sets[i,1,j] = y_test_pred[i,j]*180/np.pi + std[j]*qhat[j]
            
        elif cp_calc == 'flatten_nae':
            y_valid = y_valid.flatten()
            mu_pred = y_valid_pred[:,:n_sources].flatten()
            std_pred = y_valid_pred[:,n_sources:2*n_sources].flatten()
            mix_pred = np.zeros((y_valid_pred.shape[0],n_sources))
            for i in range(y_valid_pred.shape[0]):
                mix_pred[i] = np.exp(y_valid_pred[i,2*n_sources:])*1/np.sum(np.exp(y_valid_pred[i,2*n_sources:]), 
                                                                            axis = -1)
            scores = np.abs(y_valid[:,] - mu_pred)/std_pred
            n = scores.shape[0]
            q_level = np.ceil((n+1)*(1-alpha))/n
            #qhat = np.zeros(2,)
            #for j in range(n_sources):
            #    qhat[j] = np.quantile(scores[:,j], q_level, method='higher')   
            qhat = np.quantile(scores, q_level, method='higher')   
            
            mu_pred = y_test_pred[:,:n_sources].flatten()
            std_pred = y_test_pred[:,n_sources:2*n_sources].flatten()
            mix_pred = np.zeros((y_test_pred.shape[0],n_sources))
            for i in range(y_test_pred.shape[0]):
                mix_pred[i] = np.exp(y_test_pred[i,2*n_sources:])*1/np.sum(np.exp(y_test_pred[i,2*n_sources:]), 
                                                                            axis = -1)
            prediction_sets = np.zeros((mu_pred.shape[0],2))
            prediction_sets[:,0] = (mu_pred - std_pred*qhat)*180/np.pi
            prediction_sets[:,1] = (mu_pred + std_pred*qhat)*180/np.pi
        
        elif cp_calc == 'separate_nae':
            y_valid = y_valid
            mu_pred = y_valid_pred[:,:n_sources]
            std_pred = y_valid_pred[:,n_sources:2*n_sources]
            mix_pred = np.zeros((y_valid_pred.shape[0],n_sources))
            for i in range(y_valid_pred.shape[0]):
                mix_pred[i] = np.exp(y_valid_pred[i,2*n_sources:])*1/np.sum(np.exp(y_valid_pred[i,2*n_sources:]), 
                                                                            axis = -1)
    
            scores = np.abs(y_valid[:,] - mu_pred)/std_pred
            qhat = np.zeros(n_sources,)
            for j in range(n_sources):
                qhat[j] = np.quantile(scores[:,j], q_level, method='higher')   
            
            mu_pred = y_test_pred[:,:n_sources]
            std_pred = y_test_pred[:,n_sources:2*n_sources]
            mix_pred = np.zeros((y_test_pred.shape[0],n_sources))
            for i in range(y_test_pred.shape[0]):
                mix_pred[i] = np.exp(y_test_pred[i,2*n_sources:])*1/np.sum(np.exp(y_test_pred[i,2*n_sources:]), 
                                                                            axis = -1)
            prediction_sets = np.zeros((mu_pred.shape[0],2,n_sources))
            for j in range(n_sources):    
                prediction_sets[:,0,j] = (mu_pred[:,j] - std_pred[:,j]*qhat[j])*180/np.pi
                prediction_sets[:,1,j] = (mu_pred[:,j] + std_pred[:,j]*qhat[j])*180/np.pi
               
             
        else:
            pass
        
        
        return prediction_sets, scores
    
    
    
  
    
def conformal_testing(params_model, model_doa, x_test, y_test, y_valid, 
                      y_valid_pred, doa_limit, test_samples, alpha):
    runs = 1
        
        
    if params_model['mode'] == 'mdn':   
        cp_calc = 'likelihood'
        print('Evaluating mixture density network.')
        n_sources = params_model['num_comp']
        y_test_pred = model_doa.predict(x_test)
        
        if params_model['num_comp'] > 1:
            aa=np.abs(y_valid_pred[:,:n_sources] - y_valid)
            bb=np.abs(y_valid_pred[:,:n_sources] - y_valid[:,[1,0]])
            
            if aa[0,0] > bb[0,0]:
                y_valid_pred[:,[0,1]] = y_valid_pred[:,[1,0]]
                y_test_pred[:,[0,1]] = y_test_pred[:,[1,0]]
                y_valid_pred[:,[2,3]] = y_valid_pred[:,[3,2]]
                y_test_pred[:,[2,3]] = y_test_pred[:,[3,2]]
                y_valid_pred[:,[4,5]] = y_valid_pred[:,[5,4]]
                y_test_pred[:,[4,5]] = y_test_pred[:,[5,4]]
            else:
                pass
   
        y_test = y_test*180/np.pi
        test_samples_flatten = y_test.shape[0]
        pred_sets, scores = conformalize(y_valid, y_valid_pred, 
                                  y_test_pred, model_doa, alpha, doa_limit, 
                                  n_sources, params_model['mode'])   
        pred_sets = np.clip(pred_sets, -doa_limit, doa_limit)
        
        if cp_calc == 'flatten_nae':
            y_test = y_test.flatten()
            temp_cf = pred_sets[:,1] - pred_sets[:,0]
            uq_metric_cf = np.mean(temp_cf)
            mu_pred = y_test_pred[:,:n_sources].flatten()*180/np.pi
            test_samples_flatten = y_test.shape[0]
            coverage_score_cf = ((y_test[:,] > pred_sets[:,0]) & 
                                    (y_test[:,] < pred_sets[:,1])).sum()/test_samples_flatten
            error = np.abs(mu_pred.reshape((test_samples_flatten,1)) - y_test.reshape(test_samples_flatten,1))
            mae = np.mean(error) 
            pt, tempuq, pred_set = 0, [], []
            for s in range(y_test.shape[0]):
                scale_adj = y_test_pred[:,n_sources:2*n_sources].flatten()*180/np.pi
                pred_sets_noncf = st.norm.interval(1-alpha, 
                                      loc=np.mean(mu_pred[s]), scale=scale_adj)
                pred_sets_noncf = np.asarray(pred_sets_noncf).reshape(test_samples_flatten,2)
                pred_set.append(np.clip(pred_sets_noncf, -doa_limit, doa_limit))
                tempuq.append(pred_sets_noncf[:,1] - pred_sets_noncf[:,0])
            coverage_score_noncf = ((y_test[:,] > pred_sets_noncf[:,0]) & 
                                    (y_test[:,] < pred_sets_noncf[:,1])).sum()/test_samples_flatten
            uq_metric_noncf = np.mean(np.asarray(tempuq))
            predwidth_noncf = np.asarray(tempuq)   
            
    
        elif cp_calc == 'separate_nae' or 'likelihood':
            mu_pred = y_test_pred[:,:n_sources]*180/np.pi
            std_pred = (y_test_pred[:,n_sources:2*n_sources])*180/np.pi
            
            uq_metric_cf = np.zeros((n_sources,1))
            coverage_score_cf = np.zeros((n_sources,1))
            uq_metric_noncf = np.zeros((n_sources,1))
            coverage_score_noncf = np.zeros((n_sources,1))
            mae = np.zeros((n_sources,1))
            predwidth_cf, predwidth_noncf = [], []
            
            for j in range(n_sources):
                tempuq_cf = pred_sets[:,1,j] - pred_sets[:,0,j]
                uq_metric_cf[j] = np.mean(tempuq_cf)
                predwidth_cf.append(np.asarray(tempuq_cf))
                coverage_score_cf[j] = ((y_test[:,j] > pred_sets[:,0,j]) & 
                                    (y_test[:,j] < pred_sets[:,1,j])).sum()/test_samples_flatten
                error = np.abs(mu_pred[:,j].reshape((test_samples_flatten,1)) - 
                                y_test[:,j].reshape(test_samples_flatten,1))
                mae[j] = np.mean(error) 
                
                pt, tempuq_noncf, pred_sets_noncf = 0, [], []
                for s in range(y_test.shape[0]):
                    scale_adj = std_pred[s,j]
                    conf_int = st.norm.interval(1-alpha, 
                                          loc=np.mean(mu_pred[s,j]), scale=scale_adj)
                    pred_sets_noncf.append(np.clip(conf_int, -doa_limit, doa_limit))
                
                pred_sets_noncf = np.asarray(pred_sets_noncf)
                tempuq_noncf.append(pred_sets_noncf[:,1] - pred_sets_noncf[:,0])
                coverage_score_noncf[j] = ((y_test[:,j] > pred_sets_noncf[:,0]) & 
                                        (y_test[:,j] < pred_sets_noncf[:,1])).sum()/test_samples_flatten
                uq_metric_noncf[j] = np.mean(np.asarray(tempuq_noncf))
                predwidth_noncf.append(np.asarray(tempuq_noncf) )
        
       # elif cp_calc == 'likelihood':
       #     temp = to_categorical(y_test*180/np.pi, num_classes=180)
       #     y_test_cat = np.add(temp[:,0,:], temp[:,1,:])
       #     uq_metric_cf = np.zeros((n_sources,1))
       #     coverage_score_cf = np.zeros((n_sources,1))
       #     uq_metric_noncf = np.zeros((n_sources,1))
       #     coverage_score_noncf = np.zeros((n_sources,1))
       #     mae = np.zeros((n_sources,1))
       #     predwidth_cf, predwidth_noncf = [], []
       #     uq_metric_cf = np.sum(pred_sets.flatten())/(2*test_samples_flatten)
       #     bitpdt = 0
       #     for i in range(test_samples_flatten):
               # doa_index = np.zeros((2*doa_limit,1))
               # for j in range(n_sources):
               #     doa_index[np.floor(y_test[i,j])+90] = 1
       #         bitpdt += np.sum(np.multiply(y_test_cat[i,j], pred_sets[i,:]))
       #     coverage_score_cf = bitpdt/(test_samples_flatten*2)
            
        else:
            pass
        
        
    doa_range_illustration = False
    if doa_range_illustration: 
         import matplotlib.pyplot as plt
         pred_sets = np.asarray(pred_sets).T
        # pred_set = np.asarray(pred_set)
         
         plt.rcParams.update({'font.size': 20})
         ind = np.argsort(y_test[:,0])
         plt.figure()
         #plt.plot(y_test[ind,0], mu_pred[ind], label = 'Predicted DOA')
         plt.plot(y_test[ind,0], y_test[ind,0], 'k--', linewidth = 2.5, label = 'True DOA')
         plt.fill_between(y_test[ind,0], pred_sets[ind,0], 
                           pred_sets[ind,1], label = 'CP interval')
         plt.xlim([-85,85])
         plt.ylim([-100,100])
         plt.yticks([-90,0,90])
         plt.xlabel('DOA true value$^\circ$')
         plt.ylabel('DNN-MC prediction ($^\circ$)')
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
   
        
    return mae, uq_metric_cf, coverage_score_cf, predwidth_cf, uq_metric_noncf, coverage_score_noncf, predwidth_noncf, scores 

  






    
    
        