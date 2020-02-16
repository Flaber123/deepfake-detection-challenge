import os, sys, time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRAIN_PATH = 'D:/Data/deepfake-detection-challenge/dfdc_train_all'
TRAIN_DIRS = os.listdir(TRAIN_PATH)

