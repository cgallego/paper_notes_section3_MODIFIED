# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 11:53:06 2018

@author: DeepLearning
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


##################################################################
##  Unsupervised learning in optimal LD space: Fitting aN MLP DUAL OPTIMIZATION
################################################################## 
from decModel_wimgF_dualopt_descStats import *
labeltype = 'wimgF_dualopt_descStats_saveparams' 
save_to = r'Z:\Cristina\Section3\paper_notes_section3_MODIFIED\save_to'
#r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_wimgF_dualopt_descStats_saveparams'

# to load a prevously DEC model  
input_size = combX_allNME.shape[1]
latent_size = [input_size/rxf for rxf in [15,10,5,2]]
varying_mu = [int(np.round(var_mu)) for var_mu in np.linspace(3,12,10)]

scoresM = np.zeros((len(latent_size),len(varying_mu),5))
scoresM_titles=[]

sns.set_color_codes("pastel")

######################
# DEC: define num_centers according to clustering variable
######################   
# to save all hyperparapsm
cvorigXAUC = []; 
cvZspaceAUC_cvtrain = []; 
cvZspace_stdAUC_cvtrain = [];
cvZspaceAUC_cvVal = []; 
cvZspace_stdAUC_cvVal = [];
TestAUC = []; 

for ik,znum in enumerate(latent_size):
    to_plotcvOrigXAUC = []
    to_plotinitAUC = []
    to_plotcvZspaceAUC_cvtrain = []
    to_plotcvZspaceAUC_cvVal = []
    to_plotTestAUC = []
    for ic,num_centers in enumerate(varying_mu): 
        X = combX_allNME
        y = roi_labels
        y_train_roi_labels = np.asarray(y)

        print('Loading autoencoder of znum = {}, mu = {} , post training DEC results'.format(znum,num_centers))
        dec_model = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\paper_notes_section3_MODIFIED\\save_to\\SAEmodels') 

        with gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fu:
            dec_model = pickle.load(fu)
          
        with gzip.open(os.path.join(save_to,'outdict_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fu:
            outdict = pickle.load(fu)
        
        print('DEC train init AUC = {}'.format(outdict['meanAuc_cv'][0]))
        max_meanAuc_cv = outdict['meanAuc_cv'][-1]
        indmax_meanAuc_cv = outdict['meanAuc_cv'].index(max_meanAuc_cv)
        print r'DEC train max meanAuc_cv = {} $\pm$ {}'.format(max_meanAuc_cv,dec_model['std_auc'][indmax_meanAuc_cv])
        print('DEC validation AUC at max meanAuc_cv = {}'.format(outdict['auc_val'][indmax_meanAuc_cv]))
        
        #####################
        # extract Z-space from optimal DEC model
        #####################
        # saved output results
        dec_args_keys = ['encoder_1_bias', 'encoder_3_weight', 'encoder_0_weight', 
        'encoder_0_bias', 'encoder_2_weight', 'encoder_1_weight', 
        'encoder_3_bias', 'encoder_2_bias']
        dec_args = {key: v for key, v in dec_model.items() if key in dec_args_keys}
        dec_args['dec_mubestacci'] = dec_model['dec_mu']
        
        N = X.shape[0]
        all_iter = mx.io.NDArrayIter({'data': X}, batch_size=X.shape[0], shuffle=False,
                                                  last_batch_handle='pad')   
        ## extract embedded point zi 
        mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           
        aDEC = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\paper_notes_section3_MODIFIED\\save_to\\SAEmodels') 
        
        # organize weights and biases
        l1=[v.asnumpy().shape for k,v in aDEC.ae_model.args.iteritems()]
        k1=[k for k,v in aDEC.ae_model.args.iteritems()]
        l2=[v.asnumpy().shape for k,v in mxdec_args.iteritems()]
        k2=[k for k,v in mxdec_args.iteritems()]

        for ikparam,sizeparam in enumerate(l1):
            for jkparam,savedparam in enumerate(l2):
                if(sizeparam == savedparam):
                    #print('updating layer parameters: {}'.format(savedparam))
                    aDEC.ae_model.args[k1[ikparam]] = mxdec_args[k2[jkparam]]

        zbestacci = model.extract_feature(aDEC.feature, mxdec_args, None, all_iter, X.shape[0], aDEC.xpu).values()[0]      

        # compute model-based best-pbestacci or dec_model['pbestacci']
        pbestacci = np.zeros((zbestacci.shape[0], dec_model['num_centers']))
        aDEC.dec_op.forward([zbestacci, dec_args['dec_mubestacci'].asnumpy()], [pbestacci])
        #pbestacci = dec_model['pbestacci']
        
        # pool Z-space variables
        datalabels = np.asarray(y)
        dataZspace = np.concatenate((zbestacci, pbestacci), axis=1) 

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
       
        # We’ll load MLP using MXNet’s symbolic interface
        dataMLP = mx.sym.Variable('data')
        # MLP: two fully connected layers with 128 and 32 neurons each. 
        fc1  = mx.sym.FullyConnected(data=dataMLP, num_hidden = 128)
        act1 = mx.sym.Activation(data=fc1, act_type="relu")
        fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 32)
        act2 = mx.sym.Activation(data=fc2, act_type="relu")
        # data has 2 classes
        fc3  = mx.sym.FullyConnected(data=act2, num_hidden=2)
        # Softmax output layer
        mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
        # create a trainable module on CPU     
        batch_size = 50
        mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
        # pass train/test data to allocate model (bind state)
        MLP_train_iter = mx.io.NDArrayIter(Z_train, yZ_train, batch_size, shuffle=False)
        mlp_model.bind(MLP_train_iter.provide_data, MLP_train_iter.provide_label)
        mlp_model.init_params()   
        mlp_model.init_optimizer()
        mlp_model_params = mlp_model.get_params()[0]
        
        # update parameters based on optimal found during cv Training
        from mxnet import ndarray
        params_dict = ndarray.load(os.path.join(save_to,'mlp_model_params_z{}_mu{}.arg'.format(znum,num_centers)))
        arg_params = {}
        aux_params = {}
        for k, value in params_dict.items():
            arg_type, name = k.split(':', 1)
            if arg_type == 'arg':
                arg_params[name] = value
            elif arg_type == 'aux':
                aux_params[name] = value
            else:
                raise ValueError("Invalid param file ")

        # order of params: [(128L, 266L),(128L,),(32L, 128L),(32L,),(2L, 32L),(2L,)]
        # organize weights and biases
        l1=[v.asnumpy().shape for k,v in mlp_model_params.iteritems()]
        k1=[k for k,v in mlp_model_params.iteritems()]
        l2=[v.asnumpy().shape for k,v in arg_params.iteritems()]
        k2=[k for k,v in arg_params.iteritems()]

        for ikparam,sizeparam in enumerate(l1):
            for jkparam,savedparam in enumerate(l2):
                if(sizeparam == savedparam):
                    #print('updating layer parameters: {}'.format(savedparam))
                    mlp_model_params[k1[ikparam]] = arg_params[k2[jkparam]]
        # upddate model parameters
        mlp_model.set_params(mlp_model_params, aux_params)
        
        #####################
        # ROC: Z-space MLP fully coneected layer for classification
        #####################
        figROCs = plt.figure(figsize=(12,4))    
        # Run classifier with cross-validation and plot ROC curves
        cv = StratifiedKFold(n_splits=5,random_state=3)
        # Evaluate a score by cross-validation
        tprs_train = []; aucs_train = []
        tprs_val = []; aucs_val = []
        mean_fpr = np.linspace(0, 1, 100)
        cvi = 0
        for train, test in cv.split(Z_train, yZ_train):
            ############### on train
            MLP_train_iter = mx.io.NDArrayIter(Z_train[train], yZ_train[train], batch_size)  
            # prob[i][j] is the probability that the i-th validation contains the j-th output class.
            prob_train = mlp_model.predict(MLP_train_iter)
            # Compute ROC curve and area the curve
            fpr_train, tpr_train, thresholds_train = roc_curve(yZ_train[train], prob_train.asnumpy()[:,1])
            # to create an ROC with 100 pts
            tprs_train.append(interp(mean_fpr, fpr_train, tpr_train))
            tprs_train[-1][0] = 0.0
            roc_auc = auc(fpr_train, tpr_train)
            aucs_train.append(roc_auc)
            
            ############### on validation
            MLP_val_iter = mx.io.NDArrayIter(Z_train[test], yZ_train[test], batch_size)    
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
        axaroc_train.set_title('Unsupervised DEC + cv MLP classifier Zspace dim={}'.format(Z.shape[1]),fontsize=18)
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
        #axaroc_val.set_title('Unsupervised DEC + cv MLP classifier Zspace dim={}'.format(Z.shape[1]),fontsize=14)
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
        #axaroc.set_title('ROC LD DEC optimized space={}, all features={} - Unsupervised DEC + cv MLP classifier'.format(Z.shape[0],Z.shape[1]),fontsize=18)
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
        to_plotcvOrigXAUC.append(0.69)
        to_plotinitAUC.append(dec_model['meanAuc_cv'][0])
        to_plotcvZspaceAUC_cvtrain.append(mean_auc_train)
        to_plotcvZspaceAUC_cvVal.append(mean_auc_val)
        to_plotTestAUC.append(auc_test)
        
        scoresM[ik,ic,0] = mean_auc_train
        scoresM_titles.append("DEC cv mean_auc_train")
        scoresM[ik,ic,1] = std_auc_train
        scoresM_titles.append("DEC cv std_auc_train")    
        scoresM[ik,ic,2] = mean_auc_val
        scoresM_titles.append("DEC cv mean_auc_val")
        scoresM[ik,ic,3] = std_auc_val
        scoresM_titles.append("DEC cv std_auc_val")       
        scoresM[ik,ic,4] = auc_test
        scoresM_titles.append("DEC heal-out test AUC")          
        
    # plot latent space Accuracies vs. original
    colors = plt.cm.jet(np.linspace(0, 1, 16))
    fig2 = plt.figure(figsize=(12,6))
    #ax2 = plt.axes()
    sns.set_context("notebook")
    ax1 = fig2.add_subplot(2,1,1)
    ax1.plot(varying_mu, to_plotcvZspaceAUC_cvtrain, color=colors[0], ls=':', label="DEC+MLP cv Train")
    ax1.plot(varying_mu, to_plotcvZspaceAUC_cvVal, color=colors[2], ls=':', label="DEC+MLP cv Validation")
    ax1.plot(varying_mu, to_plotTestAUC, color=colors[8], ls='--', label="DEC+MLP held-out test")
    ax1.plot(varying_mu, to_plotcvOrigXAUC, color=colors[6], label='HD space MLP held-out test')
    ax1.set_title("Performance AUC for x{} times dimentionality reduction".format(input_size/znum))
    ax1.set_xlabel("num clusters")
    ax1.set_ylabel("AUC")
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':16})
    
    print("summary stats at x{} times dimentionality reduction".format(input_size/znum))
    print("mean cvRFZspaceAUC_cvtrain ={}".format(np.mean(to_plotcvZspaceAUC_cvtrain)))
    print("std cvRFZspaceAUC_cvtrain ={}".format(np.std(to_plotcvZspaceAUC_cvtrain)))
    print("mean cvRFZspaceAUC_cvVal ={}".format(np.mean(to_plotcvZspaceAUC_cvVal)))
    print("std cvRFZspaceAUC_cvVal ={}".format(np.std(to_plotcvZspaceAUC_cvVal)))
    print("mean TestAUC ={}".format(np.mean(to_plotTestAUC)))
    print("std TesAUC ={}".format(np.std(to_plotTestAUC)))
    
    
################
# to hold the hyperparams
grd = np.array([(zs, mus) for zs in latent_size for mus in varying_mu])
grdperf_DEC = pd.DataFrame(grd)
grdperf_DEC.columns = ["Zsize","n_mu"]

# finish fromating DEC performance
grdperf_DEC['cvorigXAUC'] = cvorigXAUC
grdperf_DEC['cvZspaceAUC_cvtrain'] = cvZspaceAUC_cvtrain
grdperf_DEC['cvZspace_stdAUC_cvtrain'] = cvZspace_stdAUC_cvtrain
grdperf_DEC['cvZspaceAUC_cvVal'] = cvZspaceAUC_cvVal
grdperf_DEC['cvZspace_stdAUC_cvVal'] = cvZspace_stdAUC_cvVal
grdperf_DEC['TestAUC'] = TestAUC
print(grdperf_DEC)

# save pooled model probabilties
grdperf_DEC.to_csv('datasets/grdperf_DEC.csv', header=True, index=False)


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter
import matplotlib.cm

figscoresM, axes = plt.subplots(nrows=5, ncols=1, figsize=(14, 24)) 
for k,ax in enumerate(axes.flat):
    im = ax.imshow(scoresM[:,:,k], cmap='BuPu_r', interpolation='nearest')
    ax.grid(False)
    for u in range(len(latent_size)):        
        for v in range(len(varying_mu)):
            ax.text(v,u,'{:.2f}'.format(scoresM[u,v,k]), color=np.array([0.05,0.15,0.15,1]),
                         fontdict={'weight': 'bold', 'size': 14})
    # set ticks
    ax.xaxis.set_major_locator(FixedLocator(np.linspace(0,9,10)))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    
    mu_labels = [str(mu) for mu in varying_mu]
    ax.set_xticklabels(mu_labels, minor=False,fontsize=16)
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(0,3,4)))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    
    znum_labels = ['2x','5x','10x','15x',] #[str(znum) for znum in latent_size]
    ax.set_yticklabels(znum_labels, minor=False,fontsize=16)
    ax.xaxis.set_label('latent space reduction')
    ax.yaxis.set_label('# cluster centroids')
    ax.set_title(scoresM_titles[k],fontsize=16)

######################
# Combined ROCs
######################  
sns.set_color_codes("pastel")
## to append pooled predictions
pooled_pred_train = pd.DataFrame()
pooled_pred_val = pd.DataFrame()

for u, znum in enumerate(latent_size):
    print('znum = {} , post training DEC results'.format(znum))
    max_test_val = max(scoresM[u,:,4])
    print "max auc heal-out test cv, at knum centroids", max_test_val
    iknum = [i for i, aucj in enumerate(scoresM[u,:,4]) if aucj == max_test_val]
    print varying_mu[iknum[0]]
    print "auc train cv, at knum centroids", scoresM[u,i,0]
    print "std auc train cv, at knum centroids", scoresM[u,i,1]
    print "auc val cv, at knum centroids", scoresM[u,i,2]
    print "std of max auc val cv, at knum centroids", scoresM[u,i,3]

    num_centers =  varying_mu[iknum[0]]
    X = combX_allNME
    y = roi_labels
    y_train_roi_labels = np.asarray(y)

    print('Loading autoencoder of znum = {}, mu = {} , post training DEC results'.format(znum,num_centers))
    dec_model = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\paper_notes_section3_MODIFIED\\save_to\\SAEmodels') 

    with gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fu:
        dec_model = pickle.load(fu)
      
    with gzip.open(os.path.join(save_to,'outdict_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fu:
        outdict = pickle.load(fu)
        
    print('DEC train init AUC = {}'.format(outdict['meanAuc_cv'][0]))
    max_meanAuc_cv = outdict['meanAuc_cv'][-1]
    indmax_meanAuc_cv = outdict['meanAuc_cv'].index(max_meanAuc_cv)
    print r'DEC train max meanAuc_cv = {} $\pm$ {}'.format(max_meanAuc_cv,dec_model['std_auc'][indmax_meanAuc_cv])
    print('DEC validation AUC at max meanAuc_cv = {}'.format(outdict['auc_val'][indmax_meanAuc_cv]))
    
    #####################
    # extract Z-space from optimal DEC model
    #####################
    # saved output results
    dec_args_keys = ['encoder_1_bias', 'encoder_3_weight', 'encoder_0_weight', 
    'encoder_0_bias', 'encoder_2_weight', 'encoder_1_weight', 
    'encoder_3_bias', 'encoder_2_bias']
    dec_args = {key: v for key, v in dec_model.items() if key in dec_args_keys}
    dec_args['dec_mubestacci'] = dec_model['dec_mu']
    
    N = X.shape[0]
    all_iter = mx.io.NDArrayIter({'data': X}, batch_size=X.shape[0], shuffle=False,
                                              last_batch_handle='pad')   
    ## extract embedded point zi 
    mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           
    aDEC = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\paper_notes_section3_MODIFIED\\save_to\\SAEmodels') 
    
    # organize weights and biases
    l1=[v.asnumpy().shape for k,v in aDEC.ae_model.args.iteritems()]
    k1=[k for k,v in aDEC.ae_model.args.iteritems()]
    l2=[v.asnumpy().shape for k,v in mxdec_args.iteritems()]
    k2=[k for k,v in mxdec_args.iteritems()]

    for ikparam,sizeparam in enumerate(l1):
        for jkparam,savedparam in enumerate(l2):
            if(sizeparam == savedparam):
                #print('updating layer parameters: {}'.format(savedparam))
                aDEC.ae_model.args[k1[ikparam]] = mxdec_args[k2[jkparam]]

    zbestacci = model.extract_feature(aDEC.feature, mxdec_args, None, all_iter, X.shape[0], aDEC.xpu).values()[0]      

    # compute model-based best-pbestacci or dec_model['pbestacci']
    pbestacci = np.zeros((zbestacci.shape[0], dec_model['num_centers']))
    aDEC.dec_op.forward([zbestacci, dec_args['dec_mubestacci'].asnumpy()], [pbestacci])
    #pbestacci = dec_model['pbestacci']
    
    # pool Z-space variables
    datalabels = np.asarray(y)
    dataZspace = np.concatenate((zbestacci, pbestacci), axis=1) 

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
   
    # We’ll load MLP using MXNet’s symbolic interface
    dataMLP = mx.sym.Variable('data')
    # MLP: two fully connected layers with 128 and 32 neurons each. 
    fc1  = mx.sym.FullyConnected(data=dataMLP, num_hidden = 128)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")
    fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 32)
    act2 = mx.sym.Activation(data=fc2, act_type="relu")
    # data has 2 classes
    fc3  = mx.sym.FullyConnected(data=act2, num_hidden=2)
    # Softmax output layer
    mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
    # create a trainable module on CPU     
    batch_size = 50
    mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
    # pass train/test data to allocate model (bind state)
    MLP_train_iter = mx.io.NDArrayIter(Z_train, yZ_train, batch_size, shuffle=False)
    mlp_model.bind(MLP_train_iter.provide_data, MLP_train_iter.provide_label)
    mlp_model.init_params()   
    mlp_model.init_optimizer()
    mlp_model_params = mlp_model.get_params()[0]
    
    # update parameters based on optimal found during cv Training
    from mxnet import ndarray
    params_dict = ndarray.load(os.path.join(save_to,'mlp_model_params_z{}_mu{}.arg'.format(znum,num_centers)))
    arg_params = {}
    aux_params = {}
    for k, value in params_dict.items():
        arg_type, name = k.split(':', 1)
        if arg_type == 'arg':
            arg_params[name] = value
        elif arg_type == 'aux':
            aux_params[name] = value
        else:
            raise ValueError("Invalid param file ")

    # order of params: [(128L, 266L),(128L,),(32L, 128L),(32L,),(2L, 32L),(2L,)]
    # organize weights and biases
    l1=[v.asnumpy().shape for k,v in mlp_model_params.iteritems()]
    k1=[k for k,v in mlp_model_params.iteritems()]
    l2=[v.asnumpy().shape for k,v in arg_params.iteritems()]
    k2=[k for k,v in arg_params.iteritems()]

    for ikparam,sizeparam in enumerate(l1):
        for jkparam,savedparam in enumerate(l2):
            if(sizeparam == savedparam):
                #print('updating layer parameters: {}'.format(savedparam))
                mlp_model_params[k1[ikparam]] = arg_params[k2[jkparam]]
    # upddate model parameters
    mlp_model.set_params(mlp_model_params, aux_params)
    
    #####################
    # ROC: Z-space MLP fully coneected layer for classification
    ####################
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=5,random_state=3)
    # Evaluate a score by cross-validation
    tprs_train = []; aucs_train = []
    tprs_val = []; aucs_val = []
    mean_fpr = np.linspace(0, 1, 100)
    cvi = 0
    for train, test in cv.split(Z_train, yZ_train):
        ############### on train
        MLP_train_iter = mx.io.NDArrayIter(Z_train[train], yZ_train[train], batch_size)  
        # prob[i][j] is the probability that the i-th validation contains the j-th output class.
        prob_train = mlp_model.predict(MLP_train_iter)
        # Compute ROC curve and area the curve
        fpr_train, tpr_train, thresholds_train = roc_curve(yZ_train[train], prob_train.asnumpy()[:,1])
        # to create an ROC with 100 pts
        tprs_train.append(interp(mean_fpr, fpr_train, tpr_train))
        tprs_train[-1][0] = 0.0
        roc_auc = auc(fpr_train, tpr_train)
        aucs_train.append(roc_auc)
        
        ############### on validation
        MLP_val_iter = mx.io.NDArrayIter(Z_train[test], yZ_train[test], batch_size)    
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
        ## appends
        if(u==3):
            pooled_pred_train = pooled_pred_train.append( pd.DataFrame({"labels":yZ_train[train],
                                  "probC":prob_train.asnumpy()[:,1],
                                  "probNC":prob_train.asnumpy()[:,0]}), ignore_index=True)
        
            pooled_pred_val = pooled_pred_val.append( pd.DataFrame({"labels":yZ_train[test],
                                  "probC":prob_val.asnumpy()[:,1],
                                  "probNC":prob_val.asnumpy()[:,0]}), ignore_index=True)
                                  
       
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
    
    if(u==3):
        pred_test = pd.DataFrame({"labels":yZ_test,
                          "probC":probas_heldout.asnumpy()[:,1],
                          "probNC":probas_heldout.asnumpy()[:,0]})
                          

 
################
# save pooled model probabilties
#pooled_pred_train.to_csv('datasets/exp3_pooled_pred_train.csv', header=True, index=False)
#pooled_pred_val.to_csv('datasets/exp3_pooled_pred_val.csv', header=True, index=False)
#pred_test.to_csv('datasets/exp3_pred_test.csv', header=True, index=False)



