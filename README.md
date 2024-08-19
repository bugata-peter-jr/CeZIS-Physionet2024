# CeZIS Team submission for the The George B. Moody PhysioNet Challenge 2024

## Abstract

Within The George B. Moody PhysioNet Challenge 2024, we developed a method for classifying electrocardiograms (ECGs) captured from images or paper printouts. Our prediction model consists of two neural networks. The first network predicts the rotation angle of the image for its reverse rotation. The second network then processes the modified image and predicts 11 possible labels for the ECG. In both cases, we used the modern residual convolutional network ConvNeXt inspired by visual transformers.

## Authors - members of CeZIS Team
    ├── VSL
    │   ├──  Peter Bugata
    │   ├──  Peter Bugata ml. 
    │   ├──  Vladimíra Kmečová
    │   ├──  Monika Staňková
    │   ├──  Dávid Hudák
    │   ├──  Dávid Gajdoš
    ├── UPJŠ
    │   ├──  Ľubomír Antoni
    │   ├──  Gabriela Vozáriková
    │   ├──  Ivan Žežula
    │   ├──  Erik Bruoth
    │   ├──  Manohar Gowdru Shridhara

## Submission structure
    ├── CeZIS Team Submission Structure
    │   ├── pretrained - folder containing trained models and prediction file
    │   │   ├── rotation              
    │   │   │   ├──weights_{FOLD}_{A/B} - .h5 files with weights from trained neural networks for rotation model
    │   │   ├── dx              
    │   │   │   ├──weights_{FOLD}_{A/B} - .h5 files with weights from trained neural networks for classification model
    │   ├── team_code.py - modified required functions for Physionet Challenge 
    │   ├── dataset.py - dataset yielding ECG images resized to 600 * 512 px
    │   ├── network.py - ConvNext network
    │   ├── metrics.py - evaluation metrics
    │   ├── losses.py - loss functions
    │   ├── config.py - model configuration
    │   ├── net_utils.py - utilities to train neural network(s)
    │   ├── train_convnext.py - training and finetuning ConvNext networks for classification


## Output structure
    ├── Submission Outputs
    │   ├── model - folder containing trained models
    │   │   ├── weights_{FOLD} - .h5 files with weights from finetuned neural networks




