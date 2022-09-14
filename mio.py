import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import random
import pylab as pl
from sklearn.metrics import confusion_matrix,accuracy_score

train_path="kioto"
class_names=os.listdir(train_path)

image_paths=[]
image_classes=[]
def img_list(path):
    return (os.path.join(path,f) for f in os.listdir(path))
##
train_dir = "ckioto"  # Directory di etichette
labels = []
with open(os.path.join(train_dir,"_classes.csv"), newline="\n") as csvfile:
    reader = csv.reader(csvfile,delimiter=",")
    for row in reader:
        labels.append((row[1]))  # 0 se è maggiorenne, 1 se è minorenne

des_list=[]
orb=cv2.ORB_create()
for image_pat in image_paths:
    im=cv2.imread(image_pat)
    kp=orb.detect(im,None)
    keypoints,descriptor= orb.compute(im, kp)
    des_list.append((image_pat,descriptor))
descriptors=des_list[0][1]
for image_path,descriptor in des_list[1:]:
    descriptors=np.vstack((descriptors,descriptor))
descriptors.shape
descriptors_float=descriptors.astype(float)

