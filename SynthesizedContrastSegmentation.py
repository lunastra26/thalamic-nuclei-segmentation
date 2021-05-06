''' 
***************************************************************
Author: LU
Date: Nov 2020
Description:
    Class for synthesized contrast segmentation framework. 
    Uses an MPRAGE (CSF-nulled T1-weighted volume) to first synthesize Wmn-MPRAGE
    contrast and predict thalamic nuclei + thalamus on the new representation
***************************************************************
'''
import os
import numpy as np
from keras import optimizers
from ModelSetup import *
from utils import *
 


class SCS:
    '''
    Synthesized Contrast based segmentation framework for thalamic nuclei segmentation
    '''
    def __init__(
        self,
        visible_gpu='0',
        loadPath="Pretrained_models/",
        
    ):
        self.visible_gpu = visible_gpu
        self.opShape = (200,200)
        self.LR = 2e-4
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.blkSize = 5
        self.num_classes = 12
        self.conv_kernel=(3,3,3)
        self.loadPath = loadPath
        self.optimizer = optimizers.Adam(self.LR, self.beta_1, self.beta_2)
        self.modelSavePath_synthesis = os.path.join(self.loadPath,'pretrained_synthesis.h5')
        self.modelSavePath_segmentation = os.path.join(self.loadPath,'pretrained_segmentation_swmn.h5')
        self.SegmentationCNN  = self.build_SegmentationCNN()
        self.SynthesisCNN = self.build_SynthesisCNN()
        
    def build_SegmentationCNN(self):
        '''
        Instantiate segmentation CNN and load it with pretrained weights for
        thalamic nuclei segmentation using synthesized WMn-MPRAGE contrast
        '''
        params = dict()
        params['ipShape'] = (self.opShape[0],self.opShape[1],self.blkSize,1) 
        params['visible_gpu'] = self.visible_gpu
        params['num_classes'] = self.num_classes
        params['optimizer'] = self.optimizer
        pretrained_weights = self.modelSavePath_segmentation
        model_seg = initializeSegmentationCNN(UNET3D_Segmentation,params,pretrained_weights)
        return model_seg
    
    def build_ContrastSynthesisCNN(self):
        '''
        Instantiate contrast synthesis CNN and load it with pretrained weights for
        transforming MPRAGE representation to WMn-MPRAGE representation
        '''
        ipShape = (self.opShape[0],self.opShape[1],self.blkSize,1) 
        model_synth = UNET3D_ContrastSynthesis_perceptual(ipShape,self.optimizer,self.visible_gpu)
        model_synth.load_weights(self.modelSavePath_synthesis)
        return model_synth
    
    def predictSynthesis(t1_nii, brainMask_nii, self):
        # Predict synthesized WMn-MPRAGE from T1-MPRAGE
        target_shape = (self.opShape[0],self.opShape[1],self.blkSize,1) 
        img = myCrop3D(t1_nii,target_shape)        
        predImg = predictSynthesis(img, self.SynthesisCNN)  
        synth_wmn_mprage = restoreCrop3D(predImg,t1_nii.shape)
        synth_wmn_mprage = synth_wmn_mprage * brainMask_nii
        synth_wmn_mprage = synth_wmn_mprage.astype(t1_nii.dtype) 
        
        return synth_wmn_mprage
    
    def predictThalamicNuclei(t1_nii, brainMask_nii, self):
        # Predict synthesized WMn-MPRAGE from T1-MPRAGE    
        target_shape = (self.opShape[0],self.opShape[1],self.blkSize,1) 
        img = myCrop3D(t1_nii,target_shape)
        mask = myCrop3D(brainMask_nii,target_shape)
        synth_wmn_mprage = predictSynthesis(img, self.SynthesisCNN)  
        synth_wmn_mprage = synth_wmn_mprage * mask
        synth_wmn_mprage = synth_wmn_mprage.astype(t1_nii.dtype)
              
        # Pre-process synthesized nii for segmeentation
        synth_wmn_mprage = np.clip(synth_wmn_mprage,0,1)
        img = contrastStretching(synth_wmn_mprage,None, 1, 99)
 
        # Thalamus and nuclei predictions for left side
        T_L_pred, FN_L_pred  = predictLabels(img, self.SegmentationCNN, blkSize=5)
        T_L_pred, FN_L_pred  = postProcessLabels(T_L_pred, FN_L_pred)
        
        # predictions for right side
        img = np.fliplr(img)
        T_R_pred, FN_R_pred  = predictLabels(img, self.SegmentationCNN, blkSize=5)
        T_R_pred, FN_R_pred  = postProcessLabels(T_R_pred, FN_R_pred)
        
        # restore predictions to original image size
        T_L_pred  = restoreCrop3D(T_L_pred,t1_nii.shape)
        FN_L_pred = restoreCrop3D(FN_L_pred,t1_nii.shape)
        T_R_pred  = restoreCrop3D(T_R_pred,t1_nii.shape)
        FN_R_pred = restoreCrop3D(FN_R_pred,t1_nii.shape)
        
        return T_L_pred, FN_L_pred, T_R_pred, FN_R_pred
















 