# sudo apt-get update
# sudo apt-get install ffmpeg libsm6 libxext6  -y

import numpy as np
import pandas as pd
import cv2
import os
import csv 
from sklearn.preprocessing import StandardScaler
from qboost import QBoostClassifier

label_dir = "train"  #Labels directory
labels = []
with open(os.path.join(label_dir,"_classes.csv"), newline="\n") as csvfile:
    reader = csv.reader(csvfile,delimiter=",")
    for row in reader:
        labels.append((row[1]))  # 0 maggiorenne, 1 minorenne
    #Conversione per quantum (1:Minorenne,-1:Maggiorenne)
    result = []
    for values in labels:
        if values == 0:
            result.append(1)
        else:
            result.append(-1)


#Feature extraction via ORB
orb = cv2.ORB_create()
des_list=[]
img_list = []
path = 'train'
dirs = os.listdir(path)
for file in dirs:
    im=cv2.imread(os.sep + 'workspace' + os.sep + 'qboost' + os.sep + 'train' + os.sep + file)
    kp = orb.detect(im,None)
    keypoints, descriptor = orb.compute(im,kp)
    des_list.append((file,descriptor))
descriptors=des_list[0][1]
for path,descriptor in des_list[1:]:
    descriptors=np.vstack((descriptors,descriptor))
descriptors.shape
descriptors_float=descriptors.astype(float)

#Clustering K Means on Descriptors
from scipy.cluster.vq import kmeans,vq
features = pd.DataFrame.from_records(descriptors_float)
k=200
voc,variance=kmeans(descriptors_float,k,1)
#Histogram of training image
im_features=np.zeros((len(dirs),k),"float32")
for i in range(len(dirs)):
    words,distance=vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w]+=1
#Standardisation of training feature
stdslr=StandardScaler().fit(im_features)
im_features=stdslr.transform(im_features)

#Qboost Classify
X=descriptors_float
y=result
lam=0.01
clf=QBoostClassifier(X,y,lam,None,None)
clf.fit(im_features,np.array(y))


# np.flatten(features.values)
#features['Id'] = descriptors
