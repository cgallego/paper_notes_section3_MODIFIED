# -*- coding: utf-8 -*-
"""
Created on Mon Apr 02 17:32:13 2018

@author: DeepLearning
"""

import pandas as pd
import numpy as np
import os
import os.path
import shutil
import glob
import tempfile
import subprocess
import SimpleITK as sitk

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
%matplotlib inline
import sys
sys.path.insert(0,'Z:\\Cristina\Section3\\paper_notes_section3_MODIFIED\\breast_MR_NME_biological')

from query_localdatabase import *
import glob
import six.moves.cPickle as pickle
import gzip
import networkx as nx

graphs_path = 'Z:\\Cristina\\Section3\\paper_notes_section3_MODIFIED\\breast_MR_NME_biological\\processed_NMEs'
nxGfeatures_path = 'Z:\\Cristina\\Section3\\paper_notes_section3_MODIFIED\\breast_MR_NME_biological\\NMEs_SER_nxgmetrics'
mha_data_loc= 'Z:\\Cristina\mha'
processed_path = r'Z:\Cristina\Section3\breast_MR_NME_pipeline\processed_data'

lesion_id = 107
localdata = Querylocaldb()
dflesion  = localdata.querylocalDatabase_wRad(lesion_id)      
cond = dflesion[0]
lesion_record = dflesion[1]
roi_record = dflesion[2]
nmlesion_record = dflesion[3]
StudyID = lesion_record['cad_pt_no_txt']
AccessionN = lesion_record['exam_a_number_txt']
DynSeries_id = nmlesion_record['DynSeries_id']  
roi_id = roi_record['roi_id']
label = roi_record['roi_label']
c = roi_record['roi_centroid']
centroid = c[c.find("(")+1:c.find(")")].split(',')
zslice = int(roi_record['zslice'])
p1 = roi_record['patch_diag1']
patch_diag1 = p1[p1.find("(")+1:p1.find(")")].split(',')
patch_diag1 = [float(p) for p in patch_diag1]
p2 = roi_record['patch_diag2']
patch_diag2 = p2[p2.find("(")+1:p2.find(")")].split(',')
patch_diag2 = [float(p) for p in patch_diag2]    

print("====================")
print('StudyID: ', StudyID)
print('AccessionN: ', AccessionN)
print('DynSeries_id: ', DynSeries_id)
print('label: ', label)
print('lesion_id: ', lesion_id)
print('roi_id: ', roi_id)
print("====================")

#############################
###### 1) Accesing mc images and lesion prob maps
#############################
# get dynmic series info
precontrast_id = int(DynSeries_id) 
DynSeries_nums = [str(n) for n in range(precontrast_id,precontrast_id+5)]

print "Reading MRI volumes..."
DynSeries_imagefiles = []
mriVols = []
preCon_filename = '{}_{}_{}'.format(StudyID.zfill(4),AccessionN,DynSeries_nums[0] )
print preCon_filename
glob_result = glob.glob(os.path.join(processed_path,preCon_filename+'@*')) #'*':do not to know the exactly acquistion time
if glob_result != []:
    filename = glob_result[0]
# read Volumnes
DynSeries_imagefiles.append(filename)
mriVolDICOM = sitk.ReadImage(filename)
mriVols.append( sitk.GetArrayFromImage(sitk.Cast(mriVolDICOM,sitk.sitkFloat32)) )
mriVolSize = mriVolDICOM.GetSize()
print "MRI volumes Size = [%f,%f,%f]..." % mriVolSize
mriVolSpacing = mriVolDICOM.GetSpacing()
print "MRI volumes spacing = [%f,%f,%f]..." % mriVolSpacing
mriVolVoxratio = mriVolSpacing[2]/mriVolSpacing[0]        

ext_x = [int(ex) for ex in [np.min([patch_diag1[0],patch_diag2[0]])-25,np.max([patch_diag1[0],patch_diag2[0]])+25] ] 
ext_y = [int(ey) for ey in [np.min([patch_diag1[1],patch_diag2[1]])-25,np.max([patch_diag1[1],patch_diag2[1]])+25] ] 

for j in range(1,5):
    DynSeries_filename = '{}_{}_{}'.format(StudyID.zfill(4),AccessionN,DynSeries_nums[j] )
    glob_result = glob.glob(os.path.join(processed_path,DynSeries_filename+'@*')) 
    if glob_result != []:
        filename = [name for name in glob_result if '_mc' in name][0] #glob_result[0]
        print filename

    # add side info from the side of the lesion
    DynSeries_imagefiles.append(filename)
    # read Volumnes
    mriVolDICOM = sitk.ReadImage(filename)
    mriVols.append( sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(DynSeries_imagefiles[j]),sitk.sitkFloat32)) )

    
## to read graph
with gzip.open(os.path.join(graphs_path,'{}_{}_{}_{}_lesion_nxgraph.pklz'.format(roi_id,StudyID.zfill(4),AccessionN,label)), 'rb') as f:
    try:
        lesionG = pickle.load(f, encoding='latin1')
    except:
        lesionG = pickle.load(f)
        
import seaborn as sns
sns.set_style("darkgrid", {'axes.grid' : False, "legend.frameon": True})
sns.set_context("paper", font_scale=2)  
nodes = [u for (u,v) in lesionG.nodes(data=True)]
pts = [v.values()[0] for (u,v) in lesionG.nodes(data=True)]
pos = dict(zip(nodes,pts))

Degree = nx.degree_centrality(lesionG) 
pd_Degree = pd.Series(Degree.values(), name="Degree")

fig, ax = plt.subplots(figsize=(16,16), dpi=160)
ax.imshow(mriVols[4][zslice,:,:], cmap=plt.cm.gray)
Dvalues = np.asarray([Degree.get(node) for node in lesionG.nodes()])

v = np.linspace(min(Dvalues), max(Dvalues), 10, endpoint=True) 
nx.draw_networkx_edges(lesionG, pos, ax=ax,  width=1, edge_color='w')
ax.axis((ext_y[0], ext_y[1]+15, ext_x[1], ext_x[0]))

# remove some nodes by degree values
remnodes = np.asarray(nodes)[[Dvalues > 0.006] or [Dvalues < 0.012]]
newDvalues = Dvalues[[Dvalues < 0.006] or [Dvalues > 0.012]]
for rn in remnodes:
    lesionG.remove_node(rn)

nxg = nx.draw_networkx_nodes(lesionG, pos, ax=ax, node_color=newDvalues, cmap=plt.cm.jet,  
                 node_vmin=np.min(newDvalues), node_vmax=np.max(newDvalues),
                 with_labels=False, node_size=125)
ax.set_axis_off()
ax.set_adjustable('box-forced')
ax.set_xlabel('Degree nodes')    

divider = make_axes_locatable(ax)
caxEdges = divider.append_axes("right", size="10%", pad=0.05)
plt.colorbar(nxg, cax=caxEdges, ticks=v) 





    
## to read graph
with gzip.open(os.path.join(graphs_path,'{}_{}_{}_{}_lesion_nxgraph.pklz'.format(roi_id,StudyID.zfill(4),AccessionN,label)), 'rb') as f:
    try:
        lesionG = pickle.load(f, encoding='latin1')
    except:
        lesionG = pickle.load(f)

Betweenness = nx.betweenness_centrality(lesionG)
pd_Betweenness = pd.Series(Betweenness.values(), name="Betweenness")
fig, ax = plt.subplots(figsize=(6,6), dpi=160)
ax.hist(Betweenness.values(),color='b')
sns.distplot(pd_Betweenness, label="Betweenness", ax=ax, hist=False)

fig, ax = plt.subplots(figsize=(16,16), dpi=160)
ax.imshow(mriVols[4][zslice,:,:], cmap=plt.cm.gray)
Betweennessvalues = np.asarray([Betweenness.get(node) for node in lesionG.nodes()])

v = np.linspace(min(Betweennessvalues), max(Betweennessvalues), 10, endpoint=True) 
nx.draw_networkx_edges(lesionG, pos, ax=ax, width=1, edge_color='w')
ax.axis((ext_y[0], ext_y[1]+15, ext_x[1], ext_x[0]))

# remove some nodes by degree values
remnodes = np.asarray(nodes)[Betweennessvalues < 0.067]
newDvalues = Betweennessvalues[Betweennessvalues > 0.067]
for rn in remnodes:
    lesionG.remove_node(rn)
    
nxg = nx.draw_networkx_nodes(lesionG, pos, ax=ax, node_color=newDvalues, cmap=plt.cm.jet,  
                 node_vmin=min(Betweennessvalues), node_vmax=max(Betweennessvalues),
                 with_labels=False, node_size=125)

ax.set_axis_off()
ax.set_adjustable('box-forced')
ax.set_xlabel('Degree nodes')    
divider = make_axes_locatable(ax)
caxEdges = divider.append_axes("right", size="10%", pad=0.05)
plt.colorbar(nxg, cax=caxEdges, ticks=v) 




## to read graph
with gzip.open(os.path.join(graphs_path,'{}_{}_{}_{}_lesion_nxgraph.pklz'.format(roi_id,StudyID.zfill(4),AccessionN,label)), 'rb') as f:
    try:
        lesionG = pickle.load(f, encoding='latin1')
    except:
        lesionG = pickle.load(f)
        
closeness = nx.closeness_centrality(lesionG)
pd_closeness = pd.Series(closeness.values(), name="closeness")
fig, ax = plt.subplots(figsize=(6,6), dpi=160)
ax.hist(closeness.values(),color='b')
sns.distplot(pd_closeness, label="closeness", ax=ax, hist=False)

fig, ax = plt.subplots(figsize=(16,16), dpi=160)             
ax.imshow(mriVols[4][zslice,:,:], cmap=plt.cm.gray)
Closenvalues = np.asarray([closeness.get(node) for node in lesionG.nodes()])

v = np.linspace(min(Closenvalues), max(Closenvalues), 10, endpoint=True) 
nx.draw_networkx_edges(lesionG, pos, ax=ax, width=1, edge_color='w')
ax.axis((ext_y[0], ext_y[1]+15, ext_x[1], ext_x[0]))

# remove some nodes by degree values
remnodes = np.asarray(nodes)[Closenvalues < 0.07]
newDvalues = Closenvalues[Closenvalues > 0.07]
for rn in remnodes:
    lesionG.remove_node(rn)
    
nxg = nx.draw_networkx_nodes(lesionG, pos, ax=ax, node_color=newDvalues, cmap=plt.cm.jet,  
                 node_vmin=min(Closenvalues), node_vmax=max(Closenvalues),
                 with_labels=False, node_size=125)
                 
ax.set_axis_off()
ax.set_adjustable('box-forced')
ax.set_xlabel('Closeness')    

divider = make_axes_locatable(ax)
caxEdges = divider.append_axes("right", size="10%", pad=0.05)
plt.colorbar(nxg, cax=caxEdges, ticks=v) 




## to read graph
with gzip.open(os.path.join(graphs_path,'{}_{}_{}_{}_lesion_nxgraph.pklz'.format(roi_id,StudyID.zfill(4),AccessionN,label)), 'rb') as f:
    try:
        lesionG = pickle.load(f, encoding='latin1')
    except:
        lesionG = pickle.load(f)

import itertools
## NOTE: clustering coefficient replaced/updated as follows:
# recalculate clustering coefficient
nodes = [u for (u,v) in lesionG.nodes(data=True)]
#weights = [d['weight'] for (u,v,d) in lesionG.edges(data=True)]
edgesdict = lesionG.edge
clustering = []
for nij  in nodes:
    #print 'node = %d' % nij
    # The pairs must be given as 2-tuples (u, v) where u and v are nodes in the graph. 
    node_adjacency_dict = edgesdict[nij]
    cc_total = 0.0
    for pairs in itertools.combinations(lesionG.neighbors_iter(nij), 2):
        #print pairs
        adjw = np.sum([node_adjacency_dict[pairs[0]].get("weight",0), node_adjacency_dict[pairs[1]].get("weight",0)])/2.0
        cc_total += adjw
    #print cc_total    
    clustering.append( cc_total )
    
pd_clustering = pd.Series(clustering, name="clustering")
fig, ax = plt.subplots(figsize=(6,6), dpi=160)
ax.hist(clustering,color='b')
sns.distplot(pd_clustering, label="clustering", ax=ax, hist=False)

fig, ax = plt.subplots(figsize=(16,16), dpi=160)             
ax.imshow(mriVols[4][zslice,:,:], cmap=plt.cm.gray)
clustering_vals = np.asarray([c for c in clustering])

v = np.linspace(min(clustering_vals), max(clustering_vals), 10, endpoint=True) 
nx.draw_networkx_edges(lesionG, pos, ax=ax, width=1, edge_color='w')
ax.axis((ext_y[0], ext_y[1]+15, ext_x[1], ext_x[0]))

# remove some nodes by degree values
remnodes = np.asarray(nodes)[clustering_vals < 0.1]
newDvalues = clustering_vals[clustering_vals > 0.1]
for rn in remnodes:
    lesionG.remove_node(rn)
    
nxg = nx.draw_networkx_nodes(lesionG, pos, ax=ax, node_color=newDvalues, cmap=plt.cm.jet,  
                 node_vmin=min(clustering_vals), node_vmax=max(clustering_vals),
                 with_labels=False, node_size=125)
                 
                 
ax.set_axis_off()
ax.set_adjustable('box-forced')
ax.set_xlabel('clustering')    
divider = make_axes_locatable(ax)


caxEdges = divider.append_axes("right", size="10%", pad=0.05)
plt.colorbar(nxg, cax=caxEdges, ticks=v)   