# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import os
import mxnet as mx
import numpy as np
import pandas as pd
import data
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import model
from autoencoder import AutoEncoderModel
from solver import Solver, Monitor
import logging
import sklearn
from sklearn.manifold import TSNE
from utilities import *
try:
   import cPickle as pickle
except:
   import pickle
import gzip

# for visualization
from sklearn.manifold import TSNE
from utilities import *
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from itertools import cycle

NME_nxgraphs = r'Z:\Cristina\Section3\paper_notes_section3_MODIFIED\datasets'
    
allNMEs_dynamic = pd.read_csv(os.path.join(NME_nxgraphs,'dyn_roi_records_allNMEs_descStats.csv'), index_col=0)

allNMEs_morphology = pd.read_csv(os.path.join(NME_nxgraphs,'morpho_roi_records_allNMEs_descStats.csv'), index_col=0)

allNMEs_texture = pd.read_csv(os.path.join(NME_nxgraphs,'text_roi_records_allNMEs_descStats.csv'), index_col=0)

allNMEs_stage1 = pd.read_csv(os.path.join(NME_nxgraphs,'stage1_roi_records_allNMEs_descStats.csv'), index_col=0)

# to load SERw matrices for all lesions
with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures_allNMEs_descStats.pklz'), 'rb') as fin:
    nxGdatafeatures = pickle.load(fin)

# to load discrall_dict dict for all lesions
with gzip.open(os.path.join(NME_nxgraphs,'nxGnormfeatures_allNMEs_descStats.pklz'), 'rb') as fin:
    discrall_dict_allNMEs = pickle.load(fin)           

#########
# shape input (798L, 427L)     
nxGdiscfeatures = discrall_dict_allNMEs   
print('Loading {} leasions with nxGdiscfeatures of size = {}'.format(nxGdiscfeatures.shape[0], nxGdiscfeatures.shape[1]) )

print('Normalizing dynamic {} leasions with features of size = {}'.format(allNMEs_dynamic.shape[0], allNMEs_dynamic.shape[1]))
normdynamic = (allNMEs_dynamic - allNMEs_dynamic.mean(axis=0)) / allNMEs_dynamic.std(axis=0)
normdynamic.mean(axis=0)

print('Normalizing morphology {} leasions with features of size = {}'.format(allNMEs_morphology.shape[0], allNMEs_morphology.shape[1]))
normorpho = (allNMEs_morphology - allNMEs_morphology.mean(axis=0)) / allNMEs_morphology.std(axis=0)
normorpho.mean(axis=0)

print('Normalizing texture {} leasions with features of size = {}'.format(allNMEs_texture.shape[0], allNMEs_texture.shape[1]))
normtext = (allNMEs_texture - allNMEs_texture.mean(axis=0)) / allNMEs_texture.std(axis=0)
normtext.mean(axis=0)

print('Normalizing stage1 {} leasions with features of size = {}'.format(allNMEs_stage1.shape[0], allNMEs_stage1.shape[1]))
normstage1 = (allNMEs_stage1 - allNMEs_stage1.mean(axis=0)) / allNMEs_stage1.std(axis=0)
normstage1.mean(axis=0)

# shape input (798L, 427L)    
combX_allNME = np.concatenate((nxGdiscfeatures, normdynamic.as_matrix(), normorpho.as_matrix(), normtext.as_matrix(), normstage1.as_matrix()), axis=1)       
YnxG_allNME = np.asarray([nxGdatafeatures['roi_id'].values,
        nxGdatafeatures['classNME'].values,
        nxGdatafeatures['nme_dist'].values,
        nxGdatafeatures['nme_int'].values])

print('Loading {} all NME of size = {}'.format(combX_allNME.shape[0], combX_allNME.shape[1]) )
print('Loading all NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_allNME[0].shape[0])   )

# define variables for DEC 
roi_labels = YnxG_allNME[1]  
roi_labels = ['K' if rl=='U' else rl for rl in roi_labels]

## use y_dec to  minimizing KL divergence for clustering with known classes
ysup = ["{}_{}_{}".format(a, b, c) if b!='nan' else "{}_{}".format(a, c) for a, b, c in zip(YnxG_allNME[1], YnxG_allNME[2], YnxG_allNME[3])]
ysup = ['K'+rl[1::] if rl[0]=='U' else rl for rl in ysup]
classes = [str(c) for c in np.unique(ysup)]
numclasses = [i for i in range(len(classes))]
y_dec = []
for k in range(len(ysup)):
    for j in range(len(classes)):
        if(str(ysup[k])==classes[j]): 
            y_dec.append(numclasses[j])
y_dec = np.asarray(y_dec)
combX_allNME[np.isnan(combX_allNME)] = 0.00001

######################################################
## From Pre-train/fine tune the SAE
######################
from decModel_wimgF_dualopt_descStats import *
from utilities import *
import model

save_to = r'Z:\Cristina\Section3\paper_notes_section3_MODIFIED\save_to\SAEmodels'
input_size = combX_allNME.shape[1]
latent_size = [input_size/rxf for rxf in [15,10,5,2]]

scoresM = np.zeros((len(latent_size),5))
scoresM_titles=[]

sns.set_color_codes("pastel")

######################
# DEC: define num_centers according to clustering variable
######################   
# to load a prevously DEC model  
# to save all hyperparapsm
cvorigXAUC = []; 
cvZspaceAUC_cvtrain = []; 
cvZspace_stdAUC_cvtrain = [];
cvZspaceAUC_cvVal = []; 
cvZspace_stdAUC_cvVal = [];
TestAUC = [];

valAUC = []
cv_SAEAUC = [] 
for ik,znum in enumerate(latent_size):
    X = combX_allNME
    y = roi_labels
    xpu = mx.cpu()
    ae_model = AutoEncoderModel(xpu, [X.shape[1],500,500,2000,znum], pt_dropout=0.2)
    print('Loading autoencoder of znum = {}, post training'.format(znum))
    ae_model.load( os.path.join(save_to,'SAE_zsize{}_wimgfeatures_descStats_zeromean.arg'.format(str(znum))) ) 

    data_iter = mx.io.NDArrayIter({'data': X}, 
                                  batch_size=X.shape[0], shuffle=False,
                                  last_batch_handle='pad')
    # extract only the encoder part of the SAE                              
    feature = ae_model.encoder
    zspace = model.extract_feature(feature, ae_model.args, None, data_iter, X.shape[0], xpu).values()[0]   
        
    # pool Z-space variables
    datalabels = np.asarray(y)
    dataZspace = zspace
        
    #####################
    # unbiased assessment: SPlit train/held-out test
    #####################
    # to compare performance need to discard unkown labels, only use known labels (only B or M)
    Z = dataZspace[datalabels!='K',:]
    y = datalabels[datalabels!='K']
  
    print '\n... MLP fully coneected layer trained on Z_train tested on Z_test' 
    sep = int(X.shape[0]*0.10)
    Z_test = Z[:sep]
    yZ_test = np.asanyarray(y[:sep]=='M').astype(int) 
    Z_train = Z[sep:]
    yZ_train = np.asanyarray(y[sep:]=='M').astype(int) 
        
    #####################
    # Z-space MLP fully coneected layer for classification
    #####################
    figROCs = plt.figure(figsize=(12,4)) 
    batch_size = 50
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=5, random_state=33)
    # Evaluate a score by cross-validation           
    tprs_train = []; aucs_train = []
    tprs_val = []; aucs_val = []
    mean_fpr = np.linspace(0, 1, 100)    
    cvi = 0
    ## to append pooled predictions
    pooled_pred_train = pd.DataFrame()
    pooled_pred_val = pd.DataFrame()
    for train, test in cv.split(Z_train, yZ_train):
        # Multilayer Perceptron
        MLP_train_iter = mx.io.NDArrayIter(Z_train[train], yZ_train[train], batch_size, shuffle=False)
        MLP_val_iter = mx.io.NDArrayIter(Z_train[test], yZ_train[test], batch_size)    
    
        # We’ll define the MLP using MXNet’s symbolic interface
        dataMLP = mx.sym.Variable('data')
        #The following code declares two fully connected layers with 128 and 64 neurons each. 
        #Furthermore, these FC layers are sandwiched between ReLU activation layers each 
        #one responsible for performing an element-wise ReLU transformation on the FC layer output.
        # The first fully-connected layer and the corresponding activation function
        fc1  = mx.sym.FullyConnected(data=dataMLP, num_hidden = 128)
        act1 = mx.sym.Activation(data=fc1, act_type="relu")
        fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 32)
        act2 = mx.sym.Activation(data=fc2, act_type="relu")
        # data has 2 classes
        fc3  = mx.sym.FullyConnected(data=act2, num_hidden=2)
        # Softmax with cross entropy loss
        mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
        
        # create a trainable module on CPU                   
        mon = mx.mon.Monitor(interval=100, pattern='.*', sort=True); # Defaults to mean absolute value |x|/size(x)
        #checkpoint = mx.callback.do_checkpoint('mlp_model_params_z{}_mu{}.arg'.format(self.best_args['znum'],self.best_args['num_centers']))
        mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
        mlp_model.fit(MLP_train_iter,  # train data
                      optimizer='sgd',  # use SGD to train
                      optimizer_params={'learning_rate':0.01},  # use fixed learning rate
                      eval_metric= 'acc', #MLPacc(yZ_val, Z_val),  # report accuracy during trainin
                      monitor=mon,
                      num_epoch=850)
                      #epoch_end_callbackcheckpoint)  # train for at most 10 dataset passes. extras:               #monitor=mon,
    
        ############### on train
        prob_train = mlp_model.predict(MLP_train_iter)
        # Compute ROC curve and area the curve
        fpr_train, tpr_train, thresholds_train = roc_curve(yZ_train[train], prob_train.asnumpy()[:,1])
        # to create an ROC with 100 pts
        tprs_train.append(interp(mean_fpr, fpr_train, tpr_train))
        tprs_train[-1][0] = 0.0
        roc_auc = auc(fpr_train, tpr_train)
        aucs_train.append(roc_auc)
        
        ############### on validation
        # prob[i][j] is the probability that the i-th validation contains the j-th output class.
        prob_val = mlp_model.predict(MLP_val_iter)
        # Compute ROC curve and area the curve
        fpr_val, tpr_val, thresholds_val = roc_curve(yZ_train[test], prob_val.asnumpy()[:,1])
        # to create an ROC with 100 pts
        tprs_val.append(interp(mean_fpr, fpr_val, tpr_val))
        tprs_val[-1][0] = 0.0
        roc_auc = auc(fpr_val, tpr_val)
        aucs_val.append(roc_auc)
        # plot
        #axaroc.plot(fpr, tpr, lw=1, alpha=0.6) # with label add: label='cv %d, AUC %0.2f' % (cvi, roc_auc)
        cvi += 1
        # save for xnum 261 or 2x reduction
        if(ik==3):
            ## appends
            pooled_pred_train = pooled_pred_train.append( pd.DataFrame({"labels":yZ_train[train],
                                  "probC":prob_train.asnumpy()[:,1],
                                  "probNC":prob_train.asnumpy()[:,0]}), ignore_index=True)
        
            pooled_pred_val = pooled_pred_val.append( pd.DataFrame({"labels":yZ_train[test],
                                  "probC":prob_val.asnumpy()[:,1],
                                  "probNC":prob_val.asnumpy()[:,0]}), ignore_index=True)
            
    # plot for cv Train
    axaroc_train = figROCs.add_subplot(1,3,1)
    # add 50% or chance line
    axaroc_train.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.9)
    # plot mean and +- 1 -std as fill area
    mean_tpr_train = np.mean(tprs_train, axis=0)
    mean_tpr_train[-1] = 1.0
    mean_auc_train = auc(mean_fpr, mean_tpr_train)
    std_auc_train = np.std(aucs_train)
    axaroc_train.plot(mean_fpr, mean_tpr_train, color='b',
                label=r'cv Train (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_train, std_auc_train),lw=3, alpha=1)     
    std_tpr = np.std(tprs_train, axis=0)
    tprs_upper = np.minimum(mean_tpr_train + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr_train - std_tpr, 0)
    axaroc_train.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.') 
    # set labels
    axaroc_train.set_xlabel('False Positive Rate',fontsize=16)
    axaroc_train.set_ylabel('True Positive Rate',fontsize=16)
    axaroc_train.set_title('SAE Zspace + cv MLP classifier Zspace dim={}'.format(Z.shape[1]),fontsize=18)
    axaroc_train.legend(loc="lower right",fontsize=16)
       
    # plot for cv val
    axaroc_val = figROCs.add_subplot(1,3,2)
    # add 50% or chance line
    axaroc_val.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.9)
    # plot mean and +- 1 -std as fill area
    mean_tpr_val = np.mean(tprs_val, axis=0)
    mean_tpr_val[-1] = 1.0
    mean_auc_val = auc(mean_fpr, mean_tpr_val)
    std_auc_val = np.std(aucs_val)
    axaroc_val.plot(mean_fpr, mean_tpr_val, color='g',
                label=r'cv Val (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_val, std_auc_val),lw=3, alpha=1)     
    std_tpr = np.std(tprs_val, axis=0)
    tprs_upper = np.minimum(mean_tpr_val + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr_val - std_tpr, 0)
    axaroc_val.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.') 
    # set labels
    axaroc_val.set_xlabel('False Positive Rate',fontsize=16)
    axaroc_val.set_ylabel('True Positive Rate',fontsize=16)
    axaroc_val.legend(loc="lower right",fontsize=16)
    
    ################
    # plot AUC on heldout set
    ################
    MLP_heldout_iter = mx.io.NDArrayIter(Z_test, None, batch_size)   
    probas_heldout = mlp_model.predict(MLP_heldout_iter)
       
    # plot for cv val
    axaroc_test = figROCs.add_subplot(1,3,3)
    # add 50% or chance line
    axaroc_test.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.9)
    # Compute ROC curve and area the curve
    fpr_test, tpr_test, thresholds_test = roc_curve(yZ_test, probas_heldout.asnumpy()[:, 1])
    auc_test = auc(fpr_test, tpr_test)
    axaroc_test.plot(fpr_test, tpr_test, color='r',
                label=r'Test (AUC = %0.2f)' % (auc_test),lw=3, alpha=1)     
    # set labels            
    axaroc_test.set_xlabel('False Positive Rate',fontsize=16)
    axaroc_test.set_ylabel('True Positive Rate',fontsize=16)
    axaroc_test.legend(loc="lower right",fontsize=16)
    plt.show()
    
    ############# append to 
    cvorigXAUC.append(0.69)
    cvZspaceAUC_cvtrain.append(mean_auc_train)
    cvZspace_stdAUC_cvtrain.append(std_auc_train)
    cvZspaceAUC_cvVal.append(mean_auc_val)
    cvZspace_stdAUC_cvVal.append(std_auc_val)
    TestAUC.append(auc_test)

    ############# append to 
    cv_SAEAUC.append(mean_auc_val)
    valAUC.append(auc_test)
    scoresM[ik,0] = mean_auc_train
    scoresM_titles.append("SAE cv mean_auc_train")
    scoresM[ik,1] = std_auc_train
    scoresM_titles.append("SAE cv std_auc_train")    
    scoresM[ik,2] = mean_auc_val
    scoresM_titles.append("SAE cv mean_auc_val")
    scoresM[ik,3] = std_auc_val
    scoresM_titles.append("SAE cv std_auc_val")       
    scoresM[ik,4] = auc_test
    scoresM_titles.append("SAE held-out test AUC")      

    ######################
    # Combined ROCs
    ######################  
     # plot for cv Train
    figROCs = plt.figure(figsize=(5,5))    
    axaroc = figROCs.add_subplot(1,1,1)
    # add 50% or chance line
    axaroc.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.9)
    # plot mean and +- 1 -std as fill area
    mean_tpr_train = np.mean(tprs_train, axis=0)
    mean_tpr_train[-1] = 1.0
    mean_auc_train = auc(mean_fpr, mean_tpr_train)
    std_auc_train = np.std(aucs_train)
    axaroc.plot(mean_fpr, mean_tpr_train, color='b',
                label=r'cv Train (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_train, std_auc_train),lw=3, alpha=1)     
    std_tpr = np.std(tprs_train, axis=0)
    tprs_upper = np.minimum(mean_tpr_train + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr_train - std_tpr, 0)
    axaroc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.') 

    # plot for cv val
    mean_tpr_val = np.mean(tprs_val, axis=0)
    mean_tpr_val[-1] = 1.0
    mean_auc_val = auc(mean_fpr, mean_tpr_val)
    std_auc_val = np.std(aucs_val)
    axaroc.plot(mean_fpr, mean_tpr_val, color='g',
                label=r'cv Val (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_val, std_auc_val),lw=3, alpha=1)     
    std_tpr = np.std(tprs_val, axis=0)
    tprs_upper = np.minimum(mean_tpr_val + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr_val - std_tpr, 0)
    axaroc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.') 
      
    ################
    # plot AUC on heldout set
    ################
    MLP_heldout_iter = mx.io.NDArrayIter(Z_test, None, batch_size)   
    probas_heldout = mlp_model.predict(MLP_heldout_iter)
      
    # plot for axaroc
    # add 50% or chance line
    axaroc.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.9)
    # Compute ROC curve and area the curve
    fpr_test, tpr_test, thresholds_test = roc_curve(yZ_test, probas_heldout.asnumpy()[:, 1])
    auc_test = auc(fpr_test, tpr_test)
    axaroc.plot(fpr_test, tpr_test, color='r',
                label=r'Test (AUC = %0.2f)' % (auc_test),lw=3, alpha=1)     
    # set labels            
    axaroc.set_xlabel('False Positive Rate',fontsize=16)
    axaroc.set_ylabel('True Positive Rate',fontsize=16)
    axaroc.set_title('Unsupervised DEC + cv MLP classifier Zspace dim={}'.format(znum),fontsize=18)
    axaroc.legend(loc="lower right",fontsize=16)
    plt.show()    
    
    if(ik==3):
        pred_test = pd.DataFrame({"labels":yZ_test,
                          "probC":probas_heldout.asnumpy()[:,1],
                          "probNC":probas_heldout.asnumpy()[:,0]})
    
# plot latent space Accuracies vs. original
colors = plt.cm.jet(np.linspace(0, 1, 16))
fig2 = plt.figure(figsize=(12,6))
#ax2 = plt.axes()
sns.set_context("notebook")
ax1 = fig2.add_subplot(2,1,1)
ax1.plot(latent_size, valAUC, color=colors[2], label='valAUC')
ax1.plot(latent_size, cv_SAEAUC, color=colors[6], label='cv_SAEAUC')
h1, l1 = ax1.get_legend_handles_labels()
ax1.legend(h1, l1, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':16})

################
# to hold the hyperparams
grd = np.array([zs for zs in latent_size])
grdperf_SAE = pd.DataFrame(grd)
grdperf_SAE.columns = ["Zsize"]

# finish fromating DEC performance
grdperf_SAE['cvorigXAUC'] = cvorigXAUC
grdperf_SAE['cvZspaceAUC_cvtrain'] = cvZspaceAUC_cvtrain
grdperf_SAE['cvZspace_stdAUC_cvtrain'] = cvZspace_stdAUC_cvtrain
grdperf_SAE['cvZspaceAUC_cvVal'] = cvZspaceAUC_cvVal
grdperf_SAE['cvZspace_stdAUC_cvVal'] = cvZspace_stdAUC_cvVal
grdperf_SAE['TestAUC'] = TestAUC
print(grdperf_SAE)

# save pooled model probabilties
grdperf_SAE.to_csv('datasets/grdperf_SAE.csv', header=True, index=False)

######################
# Combined ROCs
######################
for u, znum in enumerate(latent_size):
    print('znum = {} , post training DEC results'.format(znum))
    max_test_val = scoresM[u,4]
    print "auc train cv, at knum centroids", scoresM[u,0]
    print "std auc train cv, at knum centroids", scoresM[u,1]
    print "auc val cv, at knum centroids", scoresM[u,2]
    print "std of max auc val cv, at knum centroids", scoresM[u,3]
    print "held out auc, at knum centroids", scoresM[u,4]
    
    

################
# save pooled model probabilties
#pooled_pred_train.to_csv('datasets/exp2_pooled_pred_train.csv', header=True, index=False)
#pooled_pred_val.to_csv('datasets/exp2_pooled_pred_val.csv', header=True, index=False)
#pred_test.to_csv('datasets/exp2_pred_test.csv', header=True, index=False)