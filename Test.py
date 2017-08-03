'''
Created on Dec 16, 2016

@author: hshi
'''
import caffe
import numpy as np
from utils import viterbi_path_log



def testOneNet(netPath, 
               weightPath, 
               transmat,
               priors,
               sample,
               stateTypeNumAll,
               batchSize,
               scalar_):
    
    predicts = list()
    
    net = caffe.Net(netPath, weightPath, caffe.TEST)

    
    for i in range(len(sample)):
        
        currentSample = sample[i].reshape(-1, sample[i].shape[2]).transpose()
        
        currentSample = scalar_.transform(currentSample)
        
        
        emssionMatrixFinal = forwardOneSample(net, batchSize, currentSample, stateTypeNumAll)    
        emssionMatrixFinal = emssionMatrixFinal - np.log(np.tile(priors.reshape(stateTypeNumAll,1), emssionMatrixFinal.shape[1]))
    
        [currentPath, _, _] = viterbi_path_log(np.log(transmat), np.log(priors), emssionMatrixFinal)

        
        
        predicts.append(currentPath)
        
    return predicts

def getNextSample(sample, label, clipMarker):
    
    if sample.shape[0] > 0:
        if clipMarker[0] == 0:
            tmp = clipMarker[0:len(clipMarker)]
        
            for i in range(len(tmp)):
                if tmp[i] == 0:
                    break
            
            return i
        
        return 0
    
    return 0
        
    
def forwardOneSample(net, batchSize, Feature, stateTypeNumAll):
        
    if Feature.shape[0] % batchSize != 0:
        input_feature = np.zeros(shape = (batchSize * (Feature.shape[0]/batchSize + 1),Feature.shape[1]), dtype=np.float)
        input_feature[0:Feature.shape[0],:] = Feature
                                
    else:
        input_feature = Feature
                                
    input_cm = np.ones(input_feature.shape[0], dtype=np.uint8)
    input_cm[0] = 0                
    emssionMatrix = np.zeros(shape=(stateTypeNumAll, input_feature.shape[0]), dtype=np.float)
                               
    # GET LSTM OUTPUT 
    for i in range(input_feature.shape[0]/batchSize):

        net.blobs['sample'].data[...] = input_feature[batchSize * i:batchSize * (i + 1),:].reshape([batchSize, 1, 1, -1])    
        net.blobs['clip_marker'].data[...] = input_cm [batchSize * i:batchSize * (i + 1)].reshape([batchSize, 1])
        net.forward(start='sample_1')             
                    
        emssionMatrix[:,batchSize * i:batchSize * (i + 1)] = net.blobs['lstm_1'].data.T
                                              
    return emssionMatrix[:,0:Feature.shape[0]]


def vertibDecoding(currentPath, stateTypeNum_perClass):
    
    begFrameState = currentPath[0]      
                     
    if begFrameState % stateTypeNum_perClass == 0:
                         
        if 1: #endFrameState == begFrameState + statusTypeNumPerClass - 1:
                                         
            currentPrediction = begFrameState / stateTypeNum_perClass + 1
                             
            for i in range(len(currentPath) - 1):
                if (currentPath[i+1] - currentPath[i]) != 1 and 0:
                    currentPrediction = -1
                    break;
                                     
        else:
            currentPrediction = -1
                                      
    else:
        currentPrediction = -1
        
    return currentPrediction
