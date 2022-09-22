import os                                   #sudo apt-get update -y
import numpy as np                          #sudo apt-get install -y python3-skimage
import pandas as pd     
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from qboost import QBoostClassifier,qboost_lambda_sweep
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#from skimage import color,data
#from PIL import Image

#caricamento dataset
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
X = flat_data #array di features 2D
y = target # 0 e 1 labels(Maggiorenne,Minorenne)
result = []
for values in y: # -1,1 for qboost
    result.append(values * 2 - 1)

#Split Dataset & Principal Component Analysis
x_train, x_test, y_train, y_test = train_test_split(X, result, train_size=0.8, test_size=0.2, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
pca = PCA(n_components=150)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print('Number of features:', np.size(X, 1))
print('Number of training samples:', len(x_train))
print('Number of test samples:', len(x_test))

#Qboost classifier with lambda_sweep
normalized_lambdas = np.linspace(60, 150, 5)
n_features = np.size(X, 1)
lambdas = normalized_lambdas / n_features
print('Performing cross-validation using {} values of lambda, this make take several minutes...'.format(len(lambdas)))
qboost, lam = qboost_lambda_sweep(x_train, y_train, lambdas, verbose=True)
qboost.report_baseline(x_test,y_test)
print('Number of selected features:',len(qboost.get_selected_features()))
print('Score on test set: {:.3f}'.format(qboost.score(x_test, y_test)))

#Report metrics
y_pred=qboost.predict_class(x_train)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:" )
print(np.array(y_train))
print('Precision score %s' % precision_score(y_train, y_pred))
print('Recall score %s' % recall_score(y_train, y_pred))
print('F1-score score %s' % f1_score(y_train, y_pred))
print('Accuracy score %s' % accuracy_score(y_train, y_pred))

print("\nClassification report:")
print(classification_report(y_test,y_pred, target_names=[Maggiorenne,Minorenne]))

'''
#Qboost Classifier
lam = 0.4
qboost = QBoostClassifier (x_train, y_train, lam, weak_clf_scale=None, drop_unused=True)
qboost.report_baseline(x_test,y_test)
print('Number of selected features:',len(qboost.get_selected_features()))
print('Score on test set: {:.3f}'.format(qboost.score(x_test, y_test)))
'''