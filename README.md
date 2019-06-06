# CS431 Final Project
## Mapping from a deep convolutional neural network to V4 voxels
#### Dawn Finzi

This project attempts to find a mapping from a deep convolutional neural network to human fMRI V4 
responses to naturalistic stimuli. The natural images and fMRI data used come from Kay et al., 2008 
(available at https://crcns.org/data-sets/vc/vim-1/about-vim-1). This open dataset includes peak BOLD 
responses in two participants for each of 1,750 training and 120 validation images, with indices provided 
for the voxels corresponding to V4 (our region of interest). The CNN of choice for this project was AlexNet
trained on ImageNet, as the feature space of the output of the Conv-3 layer of this network has been shown to 
be a good predictor of V4 neural responses in macaques (Bashivan, Kar & DiCarlo, 2019). The model checkpoint 
for this project was provided by Eshed Margalit and not included in this repository due to size limitations. 
I also save the alexnet conv3 feature weights for the naturalistic stimuli as .h5 files to load in other 
notebooks and these are similarly not included due to size. 
