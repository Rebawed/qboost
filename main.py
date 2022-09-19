import pandas as pd                                 #sudo apt-get update -y
import os                                           #sudo apt-get install -y python3-skimage
from skimage.transform import resize
from skimage.io import imread
from skimage import color,data
#from skimage.filters import threshold_otsu
from PIL import Image
#import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from qboost import QBoostClassifier,qboost_lambda_sweep,EnsembleClassifier
from sys import exit
#from classify.classificator import Classifier

'''
        pca_184 = PCA(n_components=184)
#        img_pca_184_recovered = pca_184.inverse_transform(img_pca_184_reduced)
        img_array184 = img_pca_184_recovered[1,:].reshape([28,28])
'''

Categories=['Maggiorenne','Minorenne']
flat_data_arr=[] #input array
target_arr=[] #output array
datadir='littletrain' 
#path which contains all the categories of images
for i in Categories:
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(150,150,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
#df=pd.DataFrame(flat_data) #dataframe
#df['Target']=target
#X=df.iloc[:,:-1] #input data 
#y=df.iloc[:,-1] #output data
X = flat_data
y = target
#y=pd.Series(y).values
result = []
for values in y:
    result.append(values * 2 - 1)

x_train, x_test, y_train, y_test = train_test_split(X, result, train_size=0.8, test_size=0.2, shuffle=True)#stratify=y
print('Number of features:', np.size(X, 1))
print('Number of training samples:', len(x_train))
print('Number of test samples:', len(x_test))

normalized_lambdas = np.linspace(0.0, 1.75, 10)
n_features = np.size(X, 1)
lambdas = normalized_lambdas / n_features
print('Performing cross-validation using {} values of lambda, this make take several minutes...'.format(len(lambdas)))
qboost, lam = qboost_lambda_sweep(x_train, y_train, lambdas, verbose=True)
qboost.report_baseline(x_test,y_test)
print('Number of selected features:',len(qboost.get_selected_features()))
print('Score on test set: {:.3f}'.format(qboost.score(x_test, y_test)))

'''
lam = 0.4
qboost = QBoostClassifier (x_train, y_train, lam, weak_clf_scale=None, drop_unused=True)
qboost.report_baseline(x_test,y_test)
print('Number of selected features:',len(qboost.get_selected_features()))
print('Score on test set: {:.3f}'.format(qboost.score(x_test, y_test)))
  
'''