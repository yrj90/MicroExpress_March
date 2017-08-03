'''
Created on Mar 4, 2017

@author: hshi
'''

class SamplingResult(object):
    '''
    classdocs
    '''


    def __init__(self,
                 sampleInd_train,
                 sampleInd_test,
                 sampleInd_valid = None,
                 subjectId_train = None,
                 subjectId_test = None,
                 subjectId_valid = None):
        '''
        Constructor
        '''
        self.sampleInd_train = sampleInd_train
        self.sampleInd_test = sampleInd_test
        self.sampleInd_valid = sampleInd_valid
        self.subjectId_train = subjectId_train
        self.subjectId_test = subjectId_test
        self.subjectId_valid = subjectId_test
        
        
    def getSampleInd_train(self):
        return self.sampleInd_train
    
    def getSampleInd_test(self):
        return self.sampleInd_test
    
    def getSampleInd_valid(self):
        return self.sampleInd_valid
    
    def getSubjectId_train(self):
        return self.subjectId_train
    
    def getSubjectId_test(self):
        return self.subjectId_test
    
    def getSubjectId_valid(self):
        return self.subjectId_valid