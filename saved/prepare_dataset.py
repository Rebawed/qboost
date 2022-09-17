import os
from PIL import Image
import shutil

imgpath = "C:\\Users\\marco\\Desktop\\Nuova cartella\\images"
labelpath = "C:\\Users\\marco\\Desktop\\Nuova cartella\\labelTxt"

maggiorpath = "Maggiorenne"
minorpath = "Minorenne" # poi fai tu, settalo e runna sto pezzo di codice dovrebbe andare ciao mangio buon appetitooo
directory_label = os.listdir(labelpath)
polys = []
for elem in directory_label:
    with(open(labelpath + "\\" + elem)) as txt:
        parts = txt.readline().split(" ")  # Ottieni X elementi, il primo verrà sempre saltato.
        img = Image.open(imgpath + "\\" + elem[:-4] + ".jpg")
        coords = []
        imgw, imgh = img.size  # spessore + altezza immagine
        if len(parts) == 5:
            x = round(imgw * float(parts[1]))
            y = round(imgh * float(parts[2]))
            w = round(imgw * float(parts[3]))
            h = round(imgh * float(parts[4]))

            left = x - round(w / 2)
            upper = y - round(h / 2)
            right = x + round(w / 2)
            lower = y + round(h / 2)

            coords.append((left, upper))
            coords.append((right, upper))
            coords.append((right, lower))
            coords.append((left, lower))
        else:
            for i in range(1, len(parts) - 1, 2):  # Salti sempre il primo elemento
                x = round(imgw * float(parts[i]))
                y = round(imgh * float(parts[i + 1]))
                coords.append((x, y))
        polys.append(coords)

        if parts[0] == 1:
            shutil.move(imgpath + "\\" + elem[:-4] + ".jpg", minorpath)
        elif parts[0] == 0:
            shutil.move(imgpath + "\\" + elem[:-4] + ".jpg", maggiorpath)