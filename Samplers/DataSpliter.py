'''
Created on Mar 4, 2017

@author: hshi
'''
import numpy as np
import random as rd


class DataSpliter(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        
        
    
    def split(self, sampleSubject, ratio, isSubjectIndependent = False):
                
        subject = sampleSubject[:,1]
        sampleInd = sampleSubject[:,0]
        
        subjectId = np.unique(subject)
        subjectIdNum = subjectId.shape[0]
        
        subjectId_p1 = []
        subjectId_p2 = []
        sampleInd_p1 = []
        sampleInd_p2 = []
        
        
        if isSubjectIndependent:
            
            subjectIdNum_p1 = int(ratio * subjectIdNum) 
            #subjectIdNum_p2 = subjectIdNum - subjectIdNum_p1
            
            rd.shuffle(subjectId)
            
            subjectId_p1 = subjectId[0:subjectIdNum_p1]
            subjectId_p2 = subjectId[subjectIdNum_p1: subjectIdNum]
            
            sampleInd_p1 = sampleInd[np.where(np.in1d(subject.reshape([len(subject), ]), subjectId_p1))[0]]   
            sampleInd_p2 = sampleInd[np.where(np.in1d(subject.reshape([len(subject), ]), subjectId_p2))[0]]
            
            
        else:
            sampleNum = sampleInd.shape[0]
            sampleNum_p1 = int(sampleNum * ratio)
            
            rd.shuffle(sampleInd)
            
            sampleInd_p1 = sampleInd[0:sampleNum_p1]
            sampleInd_p2 = sampleInd[sampleNum_p1:sampleNum]
            
            
        
        return sampleInd_p1, \
               sampleInd_p2, \
               subjectId_p1, \
               subjectId_p2
            
            
            
            
            
            
            