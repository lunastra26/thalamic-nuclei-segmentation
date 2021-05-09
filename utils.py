"""
***************************************************************
Author: LU
Date: Nov 2020
Utilities for thalamic nuclei segmentation

***************************************************************
"""


# import files
import numpy as np
import nibabel as nib
from scipy.ndimage import morphology 
from skimage.exposure import rescale_intensity as rescale
import time
from skimage import measure
from nibabel.processing import resample_to_output

'''
Pre-processing
''' 
def contrastStretching(img, mask, lwr_prctile, upr_prctile):
    print("Contrast stretching...")
    mm = img[mask > 0]
    p_lwr = np.percentile(mm,lwr_prctile)
    p_upr = np.percentile(mm,upr_prctile)
    opImg = rescale(img,in_range=(p_lwr,p_upr))  
    return opImg

def loadMPRAGE_nii(t1_loadPath,t1_brainMaskPath, isotropic_resample=False, lwr_thresh=1, upr_thresh=99):
    print('Loading pre-processed data')      
    # Load data 
    t1_nii = nib.load(t1_loadPath)
    brainMask = nib.load(t1_brainMaskPath)
    if isotropic_resample:
        t1_nii    = resample_to_output(t1_nii,1.0)
        brainMask = resample_to_output(brainMask,1.0)
    t1_nii = t1_nii.get_data()
    brainMask = brainMask.get_data()
    t1_nii = t1_nii * brainMask
    t1_nii = np.abs(t1_nii)    
    t1_nii = contrastStretching(t1_nii.astype('float64'),brainMask,lwr_thresh,upr_thresh)
    return t1_nii, brainMask

 
def predictSynthesis(ipImg_t1, model, blkSize=5): 
    print('Sliding window predictions for synthesis...')
    tStart = time.time()
    x_test = createTestVolumefromImg(ipImg_t1,blkSize)
    y_pred = model.predict(x_test)
    final_pred = createImgfromTestVolume(y_pred,ipImg_t1.shape,blkSize)
    tElapsed = time.time() - tStart
    print('Time Elapsed: ', tElapsed)
    return final_pred
 
 
def predictLabels(ipImg, model, blkSize=5,str_el=(3,3,3)):
    x_test = createTestVolumefromImg(ipImg,blkSize)
    y_pred = model.predict(x_test)
    maskT = np.squeeze(y_pred[1])
    mask_individual = y_pred[0]
    # thalamus label
    maskT = createImgfromTestVolume(maskT,ipImg.shape,blkSize)  
    thalamusLabel = np.zeros(maskT.shape)
    thalamusLabel[maskT > 0.5] = 1
    # Nuclei label
    numNuclei = mask_individual.shape[4]   # excludes BG and interlaminar
    nucleiLabel = np.zeros(ipImg.shape)
    for num_idx in range(numNuclei):  
        temp =  mask_individual[:,:,:,:,num_idx] 
        temp = createImgfromTestVolume(temp,ipImg.shape,blkSize)
        temp2 = np.zeros(ipImg.shape)
        temp2[temp > 0.5] = 1
        # Fill holes insides labels
        temp2 = morphology.binary_closing(temp2.astype('uint8'),np.ones(str_el))
        nucleiLabel[temp2 > 0] = num_idx+1
    return thalamusLabel, nucleiLabel


def createTestVolumefromImg(ipImg,blockSize=7):
    ''' Creates a 2.5D block for evaluation using trained CNNs.   
    Blocks are created using a sliding window
    '''
    numSlices = ipImg.shape[2]
    tempNum = int(np.floor(blockSize/2))
    testVolume = []
    for idx in range(tempNum,numSlices - tempNum-1):
        slc2fetch1 = range(idx - tempNum,idx + tempNum + 1)
        x_temp = np.expand_dims(ipImg[:,:,slc2fetch1],axis=0)
        testVolume.append(x_temp)          
    testVolume = np.concatenate(testVolume,axis=0)
    testVolume = np.expand_dims(testVolume,axis=-1)  
    return testVolume

def mapLabels2THOMAS(oldLabel):
    # Maps labels ain accordance with THOMAS labels
    ip_subset = [1,2,3,4,5,6,7,8,9,10,11,12]
    op_subset = [2,4,5,6,7,8,9,10,11,12,13,14]
    newLabel = np.zeros(oldLabel.shape)
    for idx in range(len(ip_subset)):
        newLabel[oldLabel == ip_subset[idx]] = op_subset[idx]
    return newLabel


def createImgfromTestVolume(predImg,ipShape,blockSize): 
    '''  Converts 2.5D posteriors to posterior volume.   
        The center slice of each predicted block is retained.     
    '''
    final_prediction = np.zeros((ipShape))
    count=0
    numSlices = ipShape[2]
    tempNum = int(np.floor(blockSize/2))
    if len(predImg.shape) >= 5:
        predImg = np.squeeze(predImg[...,0])
    for idx in range(tempNum,numSlices - tempNum-1):
        final_prediction[:,:,idx] = predImg[count,:,:,tempNum]
        count+= 1
    return final_prediction


def postProcessLabels(thalamusLabel, nucleiLabel,str_el=(3,3,3)):
    # remove any spurious predictions outside of the main thalamus mask
    thalamusLabel = thalamusLabel.astype('uint8')
    nucleiLabel = nucleiLabel.astype('uint8')
    str_el = np.ones(str_el)
    xx = morphology.binary_dilation(thalamusLabel,str_el)
    mask = largest_connected_component(xx)   
    thalamusLabel = thalamusLabel * mask.astype('uint8')
    # remove any holes inside the predictions
    thalamusLabel = morphology.binary_closing(thalamusLabel,str_el)
    nucleiLabel = thalamusLabel * nucleiLabel
    return thalamusLabel, nucleiLabel 


def myCrop3D(ipImg,opShape,padval=0):
    '''  Creates a 3D cropped volume from ipImg based on opShape (xDim,yDim)
    ipImg is a 3D volume    
    '''
    xDim,yDim = opShape
    zDim = ipImg.shape[2]
    if padval == 0:
        opImg = np.zeros((xDim,yDim,zDim))
    else:
        opImg = np.ones((xDim,yDim,zDim)) * np.min(ipImg)
    
    xPad = xDim - ipImg.shape[0]
    yPad = yDim - ipImg.shape[1]
    
    x_lwr = int(np.ceil(np.abs(xPad)/2))
    x_upr = int(np.floor(np.abs(xPad)/2))
    y_lwr = int(np.ceil(np.abs(yPad)/2))
    y_upr = int(np.floor(np.abs(yPad)/2))
    if xPad >= 0 and yPad >= 0:
        opImg[x_lwr:xDim - x_upr ,y_lwr:yDim - y_upr,:] = ipImg
    elif xPad < 0 and yPad < 0:
        xPad = np.abs(xPad)
        yPad = np.abs(yPad)
        opImg = ipImg[x_lwr: -x_upr ,y_lwr:- y_upr,:]
    elif xPad < 0 and yPad >= 0:
        xPad = np.abs(xPad)
        temp_opImg = ipImg[x_lwr: -x_upr,:,:]
        opImg[:,y_lwr:yDim - y_upr,:] = temp_opImg
    else:
        yPad = np.abs(yPad)
        temp_opImg = ipImg[:,y_lwr: -y_upr,:]
        opImg[x_lwr:xDim - x_upr,:,:] = temp_opImg
    return opImg

def restoreCrop3D(croppedImg,origShape):
    '''Function to restore cropped mask in ipMask with shape opShape to 
    ipImg's shape 
    '''
    
    xDim,yDim,_ = croppedImg.shape
    opImg = np.zeros(origShape)
    
    xPad = xDim - origShape[0]
    yPad = yDim - origShape[1]
    
    x_lwr = int(np.ceil(np.abs(xPad)/2))
    x_upr = int(np.floor(np.abs(xPad)/2))
    y_lwr = int(np.ceil(np.abs(yPad)/2))
    y_upr = int(np.floor(np.abs(yPad)/2))
    
    if xPad >= 0 and yPad >= 0:
        opImg = croppedImg[x_lwr:xDim - x_upr ,y_lwr:yDim - y_upr,:]
    elif xPad < 0 and yPad < 0:
        xPad = np.abs(xPad)
        yPad = np.abs(yPad)
        opImg[x_lwr: -x_upr ,y_lwr:- y_upr,:] = croppedImg
    elif xPad < 0 and yPad >= 0:
        xPad = np.abs(xPad)
        
        temp_opImg= croppedImg[:,y_lwr:yDim - y_upr,:] 
        opImg[x_lwr: -x_upr,:,:] = temp_opImg
    else:
        yPad = np.abs(yPad)
        temp_opImg = croppedImg[x_lwr:xDim - x_upr,:,:]
        opImg[:,y_lwr: -y_upr,:] = temp_opImg
    return opImg

def binarizePosterior(posterior,threshold=0.5):
    img = np.zeros(posterior.shape)
    img[posterior >= threshold] = 1
    img[posterior< threshold] = 0   
    return img

# Identify the largest connected component in a binary mask
# ***************************************************************
def largest_connected_component(ipImg):       
    ipShape =  ipImg.shape
    if len(ipShape) == 2:
        conn = 2
        tempImg = np.zeros((ipImg.shape))
        for idx in range(ipImg.shape[0]):
            tempImg[idx,:] = ipImg[idx,:]
    elif len(ipShape) == 3:
        tempImg = np.zeros((ipImg.shape))
        for idx in range(ipImg.shape[0]):
            tempImg[idx,:,:] = ipImg[idx,:,:]
    conn = 3
    [labeled,num_labels] = measure.label(tempImg, background=False, connectivity=conn,return_num='true')

    rp = measure.regionprops(labeled)
    num_el = len(rp)
    max_label = 1
    for idx in range(num_el):
        if(rp[idx].label > 0):
            if(rp[idx].area > rp[max_label-1].area):
                max_label = rp[idx].label
    # Set predictions for all other labels to zero
    tempImg[labeled != max_label] = 0 
    return tempImg