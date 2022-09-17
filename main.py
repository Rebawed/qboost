import pandas as pd                                 #sudo apt-get update -y
import os                                           #sudo apt-get install -y python3-skimage
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#from classify.classificator import Classifier
from sklearn.model_selection import train_test_split
from configuration import *
from qboost import qboost_lambda_sweep

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
df=pd.DataFrame(flat_data) #dataframe
df['Target']=target
X=df.iloc[:,:-1] #input data 
y=df.iloc[:,-1] #output data

y=pd.Series(y).values
result = []
for values in y:
    result.append(values * 2 - 1)
print(result)
X = X.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, result, train_size=0.8, test_size=0.2, shuffle=True)#stratify=y
print('Splitted Successfully')

#classifier = Classifier(x_train, x_test, y_train, y_test)
normalized_lambdas = np.linspace(0.0, 1.75, 10)
n_features = np.size(X, 1)
lambdas = normalized_lambdas / n_features
print('Performing cross-validation using {} values of lambda, this make take several minutes...'.format(len(lambdas)))
qboost, lam = qboost_lambda_sweep(x_train, y_train, lambdas, verbose=True)

#qboost = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
#qboost.fit(X_train, y_train, emb_sampler, lmd=lmd, **DW_PARAMS)
'''
y_pred=classifier.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

from sklearn import svm
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
print('Splitted Successfully')
model.fit(x_train,y_train)    
print('The Model is trained well with the given images')
model.best_params_ contains the best parameters obtained from GridSearchCV

from qboost import QBoostClassifier
dwave_sampler = DWaveSampler()
emb_sampler = EmbeddingComposite(dwave_sampler)
lmd = 0.04
dwave_sampler = DWaveSampler(token=DEVtoken)
emb_sampler = EmbeddingComposite(dwave_sampler)
lmd = 0.5
qboost = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
qboost.fit(self.X_train, self.y_train, emb_sampler, lmd=lmd, **DW_PARAMS)

'''