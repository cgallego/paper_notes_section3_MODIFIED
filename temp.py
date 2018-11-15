\# -*- coding: utf-8 -*-
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
%matplotlib inline
import seaborn as sns

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

save_to = r'Z:\Cristina\Section3\paper_notes_section3_MODIFIED\save_to\SAEmodels'
input_size = combX_allNME.shape[1]
latent_size = [input_size/rxf for rxf in [2,5,10,15]]
varying_mu = [int(np.round(var_mu)) for var_mu in np.linspace(3,12,10)]


scoresM = np.zeros((len(latent_size),len(varying_mu),5))
scoresM_titles=[]

sns.set_color_codes("pastel")
######################
# DEC: define num_centers according to clustering variable
######################   
# to load a prevously DEC model  
for ik,znum in enumerate(latent_size):
    valAUC = []
    cvRSAEinitKmeansAUC = [] 
    for ic,num_centers in enumerate(varying_mu): 
        X = combX_allNME
        y = roi_labels
        print('Loading autoencoder of znum = {}, mu = {} , post training DEC results'.format(znum,num_centers))
        dec_model = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\paper_notes_section3_MODIFIED\\save_to\\SAEmodels') 

        # extract zSpace after initialization of autoencoder
        test_iter = mx.io.NDArrayIter({'data': X}, 
                                      batch_size=X.shape[0], shuffle=False,
                                      last_batch_handle='pad')
        args = {k: mx.nd.array(v.asnumpy(), ctx=dec_model.xpu) for k, v in dec_model.args.items()}
        ## embedded point zi 
        zspace = model.extract_feature(dec_model.feature, args, None, test_iter, X.shape[0], dec_model.xpu).values()[0]   
        
        # compute model-based best-pspace or dec_model['pspace']
        pspace = np.zeros((zspace.shape[0], dec_model.num_centers))
        dec_model.dec_op.forward([zspace, args['dec_mu'].asnumpy()], [pspace])

        # pool Z-space variables
        datalabels = np.asarray(y)
        dataZspace = np.concatenate((zspace, pspace), axis=1) 

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
        for train, test in cv.split(Z_train, yZ_train):
            # Multilayer Perceptron
            MLP_train_iter = mx.io.NDArrayIter(Z_train[train], yZ_train[train], batch_size, shuffle=True)
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
        axaroc_train.set_title('SAE + init kmeans Zspace + cv MLP classifier Zspace dim={}'.format(Z.shape[1]),fontsize=18)
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
        cvRSAEinitKmeansAUC.append(mean_auc_val)
        valAUC.append(auc_test)
        scoresM[ik,ic,0] = mean_auc_train
        scoresM_titles.append("SAE initKmeans cv mean_auc_train")
        scoresM[ik,ic,1] = std_auc_train
        scoresM_titles.append("SAE initKmeans cv std_auc_train")    
        scoresM[ik,ic,2] = mean_auc_val
        scoresM_titles.append("SAE initKmeans cv mean_auc_val")
        scoresM[ik,ic,3] = std_auc_val
        scoresM_titles.append("SAE initKmeans cv std_auc_val")       
        scoresM[ik,ic,4] = auc_test
        scoresM_titles.append("SAE initKmeans held-out test AUC")          
        
    # plot latent space Accuracies vs. original
    colors = plt.cm.jet(np.linspace(0, 1, 16))
    fig2 = plt.figure(figsize=(12,6))
    #ax2 = plt.axes()
    sns.set_context("notebook")
    ax1 = fig2.add_subplot(2,1,1)
    ax1.plot(varying_mu, valAUC, color=colors[2], label='valAUC')
    ax1.plot(varying_mu, cvRSAEinitKmeansAUC, color=colors[6], label='cvRSAEinitKmeansAUC')
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':16})