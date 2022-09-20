import os                                   #sudo apt-get update -y
import numpy as np                          #sudo apt-get install -y python3-skimage
import pandas as pd                                                       
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from qboost import QBoostClassifier,qboost_lambda_sweep
from sklearn.metrics import accuracy_score
from sys import exit
#from skimage import color,data
#import matplotlib.pyplot as plt
#from PIL import Image

Categories=['Maggiorenne','Minorenne']
flat_data_arr=[] #input array
target_arr=[] #output array
datadir='littletrain' #path di training
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
X = flat_data
y = target
result = []
for values in y:
    result.append(values * 2 - 1)

x_train, x_test, y_train, y_test = train_test_split(X, result, train_size=0.8, test_size=0.2, random_state=0)#stratify=y
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
pca = PCA(n_components=240)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print('Number of features:', np.size(X, 1))
print('Number of training samples:', len(x_train))
print('Number of test samples:', len(x_test))
lam = 0.4
qboost = QBoostClassifier (x_train, y_train, lam, weak_clf_scale=None, drop_unused=True)
qboost.report_baseline(x_test,y_test)
print('Number of selected features:',len(qboost.get_selected_features()))
print('Score on test set: {:.3f}'.format(qboost.score(x_test, y_test)))

y_pred=qboost.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))

#print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

'''
normalized_lambdas = np.linspace(0.0, 1.75, 10)
n_features = np.size(X, 1)
lambdas = normalized_lambdas / n_features
print('Performing cross-validation using {} values of lambda, this make take several minutes...'.format(len(lambdas)))
qboost, lam = qboost_lambda_sweep(x_train, y_train, lambdas, verbose=True)
qboost.report_baseline(x_test,y_test)
print('Number of selected features:',len(qboost.get_selected_features()))
print('Score on test set: {:.3f}'.format(qboost.score(x_test, y_test)))
'''