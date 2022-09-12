# sudo apt-get update
# sudo apt-get install ffmpeg libsm6 libxext6  -y

import numpy as np
import pandas as pd
import cv2
import os
import csv 
from qboost import QBoostClassifier


train_dir = "ckioto"  # Directory della cartella del dataset di training
labels = []
with open(os.path.join(train_dir,"_classes.csv"), newline="\n") as csvfile:
    reader = csv.reader(csvfile,delimiter=",")
    for row in reader:
        labels.append((row[1]))  # 0 se è maggiorenne, 1 se è minorenne

    #lista di etichette per calcolo quantistico (1:Minorenne,-1:Maggiorenne)
    result = []
    for values in labels:
        if values == 0:
            result.append(1)
        else:
            result.append(-1)


#inizializzazione orb detector
orb = cv2.ORB_create()

#creazione lista di feature
img_list = []
#img_listgray = []
keypoints = []
descriptors = []
path = 'kioto'
dirs = os.listdir(path)
for file in dirs:
    img_list.append(cv2.imread(os.sep + 'workspace' + os.sep + 'qboost' + os.sep + 'kioto' + os.sep + file))
    #img_listgray = cv2.cvtColor(img_list, cv2.COLOR_BGR2GRAY)
    keypoints = orb.detect(img_list,None)
    keypoints, descriptors = orb.compute(img_list,keypoints)

X = descriptors
y = result
features = pd.DataFrame.from_records(X)
# np.flatten(features.values)
features['Id'] = descriptors

qboost = QBoostClassifier(X,y,1)