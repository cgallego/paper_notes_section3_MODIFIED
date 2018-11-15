# -*- coding: utf-8 -*-
"""
Created on Sun Apr 01 16:46:20 2018

@author: DeepLearning
"""

# to save graphs
import sys
import os
import six.moves.cPickle as pickle
import gzip
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams.update({'font.size': 36})

import seaborn as sns# start by loading nxGdatafeatures
NME_nxgraphs = r'Z:\Cristina\Section3\paper_notes_section3_MODIFIED\\breast_MR_NME_biological\NMEs_SER_nxgmetrics'

# to load nxGdatafeatures df for all lesions
with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures_allNMEs_10binsize.pklz'), 'rb') as fin:
    nxGdatafeatures = pickle.load(fin)






# gather other infor such as patient levels
sys.path.insert(0,'Z:\\Cristina\Section3\\paper_notes_section3_MODIFIED\\breast_MR_NME_biological')
from query_localdatabase import *
from sqlalchemy.orm import sessionmaker, joinedload_all
from sqlalchemy import create_engine
querydb = Querylocaldb()
# configure Session class with desired options
Session = sessionmaker()
queryengine = create_engine('sqlite:///Z:\\Cristina\\Section3\\paper_notes_section3_MODIFIED\\breast_MR_NME_biological\\nonmass_roibiological.db', echo=False) # now on, when adding new cases # for filled by Z:\\Cristina\\Section3\\NME_DEC\\imgFeatures\\NME_forBIRADSdesc\\nonmass_roirecords.db
Session.configure(bind=queryengine)  # once engine is available
session = Session() #instantiate a Session


U_lesion_ids = pd.Series(nxGdatafeatures['lesion_id'], dtype="str")[pd.Series(nxGdatafeatures['classNME'], dtype="str") == 'U']
pt_info = pd.DataFrame()
roi_info = pd.DataFrame()
for lesion_id in U_lesion_ids:
    # perform query
    ############# by lesion id
    lesion = session.query(localdatabase.Lesion_record, localdatabase.Radiology_record, localdatabase.ROI_record).\
        filter(localdatabase.Radiology_record.lesion_id==localdatabase.Lesion_record.lesion_id).\
        filter(localdatabase.ROI_record.lesion_id==localdatabase.Lesion_record.lesion_id).\
        filter(localdatabase.Lesion_record.lesion_id == str(lesion_id)).options(joinedload_all('*')).all()
    # print results
    if not lesion:
        print "lesion is empty"
        lesion_id = lesion_id+1
        continue
    
    lesion = lesion[0]    
    print lesion_id
    
    # lesion frame       
    casesFrame = pd.Series(lesion.Lesion_record.__dict__)
    pt_info = pt_info.append(casesFrame, ignore_index=True)   
    roiFrame = pd.Series(lesion.ROI_record.__dict__)
    roi_info = roi_info.append(roiFrame, ignore_index=True)  
    print roiFrame
    
status6m = []
status2y = []
cond2y = []
cond6m =[]
for i in range(415):
    status6m.append( roi_info['FU6months'][i] ) 
    upto2y = roi_info['FU6months'][i] and roi_info['FU1year'][i] and roi_info['FU2year'][i]
    only2y = roi_info['FU2year'][i]
    status = upto2y or only2y
    status2y.append( status )
    cond2y.append( roi_info['FU2ycondition'][i] )
    cond6m.append( roi_info['FU6mcondition'][i] )
   
# numbber of follow ups at 6 months
sum(np.asarray(roi_info['FU6months'])[3::]) 
sum(np.asarray(roi_info['FU1year'])[3::]) 
sum(np.asarray(roi_info['FU2year'])[3::]) 

# all followups
sum(np.asarray(roi_info['FU6months'])[3::]) + sum(np.asarray(roi_info['FU1year'])[3::]) + sum(np.asarray(roi_info['FU2year'])[3::])

sum(np.asarray(status6m)[3::]) 
pdcond6m = pd.Series(cond6m, dtype="str")
cat_levels = pdcond6m.unique()
total = len(pdcond6m)
for k in range(len(cat_levels)):
    totalcat = np.sum(pdcond6m == cat_levels[k]) 
    print '%s (n=%d) (%d/%d = %f percent) \n' % (cat_levels[k], totalcat, totalcat, total, totalcat/float(total) )    


pdcond2y = pd.Series(cond2y, dtype="str")
cat_levels = pdcond2y.unique()
total = len(pdcond2y)
for k in range(len(cat_levels)):
    totalcat = np.sum(pdcond2y == cat_levels[k]) 
    print '%s (n=%d) (%d/%d = %f percent) \n' % (cat_levels[k], totalcat, totalcat, total, totalcat/float(total) )    
