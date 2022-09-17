import numpy as np  # sudo apt-get update
import pandas as pd  # sudo apt-get install ffmpeg libsm6 libxext6  -y
import cv2
import os
import csv
from sys import exit
from qboost import QBoostClassifier
from sklearn.preprocessing import StandardScaler

#import matplotlib.pyplot as plt

label_dir = "ckioto"  # Labels directory
labels = []
with open(os.path.join(label_dir, "_classes.csv"), newline="\n") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        labels.append((row[0], row[1]))  # 0 maggiorenne, 1 minorenne
    # Conversione per quantum (1:Minorenne,-1:Maggiorenne)
    result = []
    for values in labels:
        if values == 0:
            result.append(1)
        else:
            result.append(-1)

# Feature extraction
sift = cv2.SIFT_create()
orb = cv2.ORB_create()
flat_data_arr=[]
des_list = []
img_list = []
path = 'kioto'
dirs = os.listdir(path)
for file in dirs:
    im=cv2.imread(os.sep + 'workspace' + os.sep + 'qboost' + os.sep + path + os.sep + file)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kp, descriptor = sift.detectAndCompute(gray, None)
    #keypoints, descriptor = orb.compute(im, kp)
    #img = cv2.drawKeypoints(gray, kp, None, (0, 255, 0), flags=0)
    #plt.imshow(img)
    #plt.show()
    des_list.append((file, descriptor))
#  Per ogni elemento di descriptors, ho una lista di descrittori. Se ho 4 immagini, ho 4 liste di descrittori.



descriptors = []

for path, n_descriptors in des_list:
    descriptors.append(n_descriptors)
    
descriptors = np.array(descriptors)

flat_data_arr = descriptors.flatten()
#.append(descriptors.flatten())

#print(type(des_list[0][1]))

#descriptors = np.asanyarray(descriptors, dtype=object)

    

# Clustering K Means on Descriptors
from scipy.cluster.vq import kmeans, vq

features = pd.DataFrame.from_records(flat_data_arr)
k = 200
voc, variance = kmeans(flat_data_arr, k, 1)

# Histogram of training image
im_features = np.zeros((len(dirs), k), "float32")
for i in range(len(dirs)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

#Standardisation of training feature
stdslr=StandardScaler().fit(im_features)
im_features=stdslr.transform(im_features)

#Qboost Classify
X=flat_data_arr
y=result
lam=0.01
clf=QBoostClassifier(X,y,lam)
clf.fit(im_features,np.array(y))