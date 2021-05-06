'''
Author: LU
Date: Nov 2020
Demo script to use SCS framework to predict thalamic nuclei on an MPRAGE test volumes
Script makes the following assumptions about data
1) MPRAGE images are brain extracted and bias corrected
2) Acquired axially. Otherwise, MPRAGE images would have to be adjusted to be in
axial orientation
3) Data is available in subfolder 'Data/'
4) GPU '0' is the default GPU

Script uses an MPRAGE (CSF-nulled T1-weighted volume) to first synthesize Wmn-MPRAGE
contrast and predict thalamic nuclei + thalamus on the new representation
'''
import os
from utils import loadMPRAGE_nii
from ModelSetup import setTF_environment


#%% User parameters
 
loadPath="Data/"

t1_loadPath = os.path.join(loadPath,'t1.nii')
t1_brainMaskPath = os.path.join(loadPath,'t1_brainMask.nii')

'''
To ensure MPRAGE images are loaded in axial orientation with correct left/right
orientType = 1 does 'np.rot90(nii,-1)'
orientType = 2 does np.rot90(np.fliplr(nii),-1)
'''
orientType = 1

setTF_environment   
#%% Load MPRAGE data
# Load nifti files, 

mprage_nii, brainMask_nii = loadMPRAGE_nii(t1_loadPath,t1_brainMaskPath,orientType)

#%% Instantiate synthesized contrast segmentation framework

from SynthesizedContrastSegmentation import SCS as SCS

SCS_CNN = SCS()

#%% Generate synthesized WMn-MPRAGE representation from MPRAGE
synth_wmn_mprage = SCS_CNN.predictSynthesis(mprage_nii, brainMask_nii)


#%% Predict thalamic nuclei using SCS framework
'''
T_L: Binary mask for thalamus (left)
N_L: Binary mask for thalamic nuclei (left)
T_R: Binary mask for thalamus (right)
N_R: Binary mask for thalamic nuclei (right)
'''
T_L, N_L, T_R, N_R = SCS_CNN.predictThalamicNuclei(mprage_nii, brainMask_nii)

