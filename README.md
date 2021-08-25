# Convolutional Neural Network based Thalamic Nuclei Segmentation for MPRAGE images
Thalamic nuclei have been implicated in several neurological diseases. Thalamic nuclei parcellation from structural MRI is challenging due to poor intra-thalamic nuclear contrast while methods based on diffusion and functional MRI are affected by limited spatial resolution and image distortion. 
Existing multi-atlas based techniques are often computationally intensive and time-consuming. 
In this work, we propose a 3D Convolutional Neural Network (CNN) based framework for thalamic nuclei parcellation using T1-weighted Magnetization Prepared Rapid Gradient Echo (MPRAGE) images. Transformation of images to an efficient representation has been proposed to improve the performance of subsequent classification tasks especially when working with limited labeled data. We investigate this by transforming the MPRAGE images to White-Matter-nulled MPRAGE (WMn-MPRAGE) contrast, previously shown to exhibit good intra-thalamic nuclear contrast, prior to the segmentation step.

We trained two 3D segmentation frameworks using MPRAGE images (n=35 subjects): a) a native contrast segmentation (NCS) on MPRAGE images and b) a synthesized contrast segmentation (SCS) where synthesized WMn-MPRAGE representation generated by a contrast synthesis CNN were used. Thalamic nuclei labels were generated using THOMAS, a multi-atlas segmentation technique proposed for WMn-MPRAGE images. 

More details are available in the following article:
L Umapathy, MB Keerthivasan, NM Zahr, Ali Bilgin, and Manojkumar Saranathan. ***Convolutional Neural Network based frameworks for fast automatic segmentation of thalamic nuclei from native and synthesized contrast structural MRI***. To appear in Neuroinformatics.
