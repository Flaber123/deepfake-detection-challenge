# Deepfake Detection Challenge
Collection of resources and code for the Deepfake Detection Challenge hosted on Kaggle.

### Dataset
- The Deepfake Detection Challenge (DFDC) Preview Dataset: https://arxiv.org/abs/1910.08854

### Face recognition
- Pretrained Pytorch face detection (MTCNN) and recognition (InceptionResnet) models: https://github.com/timesler/facenet-pytorch

### Pre-processing
- Subtracting pixel mean from each image

### Methods to try out
- Unmasking with simple features: https://github.com/cc-hpc-itwm/DeepFakeDetection
- MesoNet: https://github.com/DariusAf/MesoNet
- EfficientNet: https://github.com/mingxingtan/efficientnet

### Validation
- Group fold split by folder: https://www.kaggle.com/c/deepfake-detection-challenge/discussion/124127
- Undersampling FAKE, oversampling REAL: 

## Pipeline
Current training / predicting pipeline:
1. Extract faces in 1:3 ratio with image size of 240x240
2. Train EfficientNetB1 model with base layers frozen
3. Train MesoNet model with completely new weights
4. Train SVM using simple features
