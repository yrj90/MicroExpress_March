'''
Created on Mar 3, 2017

@author: hshi
'''
import numpy as np
from DataSpliter import DataSpliter
from Results.SamplingResult import SamplingResult
class LearveOneSubjectOutSampler(object):
    '''
    classdocs
    '''


    def __init__(self, 
                 sampleSubject,
                 ratio_valid, 
                 SUBJECT_INDEPENDENT_VALIDATION = False):
        '''
        Constructor
        '''
        
        self.mySpliter_valid = DataSpliter()
        
        self.SUBJECT_INDEPENDENT_VALIDATION = SUBJECT_INDEPENDENT_VALIDATION
        self.sampleSubject = sampleSubject
        self.ratio_valid = ratio_valid
        
        self.subject = self.sampleSubject[:,1]
        self.sampleInd = self.sampleSubject[:,0]
        
        self.subjectId = np.unique(self.subject)
        
        self.subjectIdNum = self.subjectId.shape[0]
        self.sampleNum = self.sampleSubject.shape[0]
        
        
        self.n_folds = self.subjectIdNum
        self.cvCursor = 0
        
    def getCursor(self):
        return self.cvCursor
        
    def getNextSamplingResult(self):
        
        if self.cvCursor < self.n_folds:
            
            subjectId_test = self.subjectId[self.cvCursor]
        
            sampleInd_test = self.sampleInd[np.where(np.in1d(self.subject.reshape([len(self.subject),]), subjectId_test))[0]]
            
            subjectId_train_valid = np.concatenate((self.subjectId[0:self.cvCursor], 
                                                    self.subjectId[self.cvCursor + 1 : len(self.subjectId)]))
            
            
            sampleSubjectInd_train_valid = np.where(np.in1d(self.subject.reshape([len(self.subject),]),
                                                     subjectId_train_valid))[0]
                                                
            sampleSubject_train_valid = self.sampleSubject[sampleSubjectInd_train_valid,:]
            
            sampleInd_valid, \
            sampleInd_train, \
            subjectId_valid, \
            subjectId_train = self.mySpliter_valid.split(sampleSubject_train_valid, self.ratio_valid, self.SUBJECT_INDEPENDENT_VALIDATION)
            
            
            return SamplingResult(sampleInd_train, sampleInd_test, sampleInd_valid,
                                  subjectId_train, subjectId_test, subjectId_valid)
                   
        
        
        else:
            None
        
        