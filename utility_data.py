'''
import os
from PIL import Image
import shutil

imgpath = "C:\\Users\\marco\\Desktop\\estrazionee\\images"
labelpath = "C:\\Users\\marco\\Desktop\\estrazionee\\labels"
maggiorpath = "C:\\Users\\marco\\Desktop\\estrazionee\\Maggiorenne"
minorpath = "C:\\Users\\marco\\Desktop\\estrazionee\\Minorenne"

directory_label = os.listdir(labelpath)
for elem in directory_label:
    with(open(labelpath + "\\" + elem)) as txt:
        parts = txt.readline().split(" ")
        img = Image.open(imgpath + "\\" + elem[:-4] + ".jpg")
        if parts[0] == str(1):
            shutil.copy(imgpath + "\\" + elem[:-4] + ".jpg", maggiorpath)
        elif parts[0] == str(0):
            shutil.copy(imgpath + "\\" + elem[:-4] + ".jpg", minorpath)

#print matrice
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

Categories = ['Maggiorenne','Minorenne']
y_pred = 
y_train = 

cm = confusion_matrix(y_train,y_pred)
print("Confusion matrix:")
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Categories)
disp.plot()
plt.show()
'''