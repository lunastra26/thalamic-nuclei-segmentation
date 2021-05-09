# Convolutional Neural Network based Thalamic Nuclei Segmentation for MPRAGE images
Thalamic nuclei have been implicated in several neurological diseases. White-Matter-nulled Magnetization Prepared Rapid Gradient Echo (WMn-MPRAGE) images have been shown to provide better intra-thalamic nuclear contrast compared to conventional MPRAGE images (Tourdias et al. 2014) but the additional acquisition results in increased examination times. 

In this work, we investigated 3D Convolutional Neural Network (CNN) based techniques for thalamic nuclei parcellation from conventional MPRAGE images. We used a CNN trained on MPRAGE contrast (Native contrast segmentation or NCS). We also investigated the impact of transforming MPRAGE images to a new representation, white matter nulled MPRAGE using a contrast synthesis CNN prior to the segmentation CNN. We refer to this as synthesized contrast segmentation (SCS).
We trained the two segmentation frameworks using MPRAGE images (n=35) and thalamic nuclei labels generated on WMn-MPRAGE images using a multi-atlas based parcellation technique (Thalamus Optimized Multi Atlas Segmentation or THOMAS) (Su et al. 2019).

More details are available in:

L Umapathy, MB Keerthivasan, N Zahr, A Bilgin, and M Saranathan ” A Contrast Synthesized Thalamic Nuclei Segmentation Scheme using Convolutional Neural Networks. Under revision: Neuroinformatics.  
