import os
from PIL import Image
import shutil

imgpath = "C:\\Users\\marco\\Desktop\\estrazionee\\images"
labelpath = "C:\\Users\\marco\\Desktop\\estrazionee\\labels"

maggiorpath = "C:\\Users\\marco\\Desktop\\estrazionee\\Documento"
minorpath = "C:\\Users\\marco\\Desktop\\estrazionee\\Passaporto"
directory_label = os.listdir(labelpath)

for elem in directory_label:
    with(open(labelpath + "\\" + elem)) as txt:
        parts = txt.readline().split(" ")  # Ottieni X elementi, il primo verrà sempre saltato.
        img = Image.open(imgpath + "\\" + elem[:-4] + ".jpg")
        if parts[0] == str(1):
            shutil.copy(imgpath + "\\" + elem[:-4] + ".jpg", maggiorpath)
        elif parts[0] == str(0):
            shutil.copy(imgpath + "\\" + elem[:-4] + ".jpg", minorpath)
