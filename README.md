# Deepfake Detection Challenge
Collection of resources and code for the Deepfake Detection Challenge hosted on Kaggle.

### Dataset
- The Deepfake Detection Challenge (DFDC) Preview Dataset: https://arxiv.org/abs/1910.08854

### Face recognition
- Pretrained Pytorch face detection (MTCNN) and recognition (InceptionResnet) models: https://github.com/timesler/facenet-pytorch

### Methods to try out
- Unmasking with simple features: https://github.com/cc-hpc-itwm/DeepFakeDetection
- MesoNet: https://github.com/DariusAf/MesoNet
- EfficientNet: https://github.com/mingxingtan/efficientnet

### Validation
- Yifan Xie (29th in this Competition): "I have done some local experiment with both validation by group, and validation by original videos. My observation is that the gap between training and validation error is a lot wider when validating by original videos. For me that is a clear indication of overfitting. As pointed out by @liuftvafas, we do have the same actor in different groups. nevertheless, base on the above evidence I would go with validation by groups until better idea is shared :)" https://www.kaggle.com/c/deepfake-detection-challenge/discussion/124127
