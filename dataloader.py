import os
import cv2
import numpy as np
from utilities import showimage
from tqdm.auto import tqdm
import csv
def train_(datapath='data' , save_=True , modelpath=f'.{os.sep}facerecon.xml' , reshape=(512,512)):
    faces , labels = get_faces_labels(datapath)
    facerecon = cv2.face.FisherFaceRecognizer.create()
    print(f"training_ contains {len(faces)} faces and {len(labels)} lables!")
    ## convert to (512 ,512) to set one dimm
    faces = [cv2.resize(src=faces[i] , dsize=reshape) for i in range(len(faces))]
    facerecon.train(faces , labels)
    if save_ : facerecon.save(modelpath)
    return facerecon

def predict_( faceroi , model ,distanceerr = 1000, reshape=(512,512)):
    faceroi = cv2.resize(src=faceroi , dsize=reshape , interpolation=cv2.INTER_LINEAR)
    labeltemp , distance = model.predict(faceroi)
    out = {'label':labeltemp , 'confidence':90} if  distance < distanceerr else None
    return out

def updateStudents(traindir , outputpath = 'AttendenceReport.csv'):
    H = ['RollNo' , 'Name' , 'Surname' , 'Status']
    out = csv.writer(open(outputpath, "w") ,delimiter=',', quoting=csv.QUOTE_ALL)
    out.writerow(H)
    for i in os.listdir(traindir):
        data = i.split('_')
        data.insert(len(data),'A')
        out.writerow(data)
    return
def get_faces_labels(datapath='data' , scalesize=2):
    facelist = []
    labels = []
    updateStudents(traindir=datapath)
    for dir_ in os.listdir(datapath):
        label = int(dir_.split('_')[0])
        print(f"reading data for rollno :{label}")
        for imagename in tqdm(os.listdir(datapath + os.sep + dir_)):
            if not imagename.__contains__('.jpg'): continue
            imagepath = os.path.join(datapath,dir_ , imagename)
            image = cv2.imread(imagepath,cv2.COLOR_RGB2BGR)
            if int(image.shape[1] / 512) > scalesize or int(image.shape[0] / 512) > scalesize: image = cv2.resize(src=image, dsize=(
            image.shape[1] // scalesize ,image.shape[0] // scalesize))
            face_roi , faces = detect_face(image)
            for i in face_roi:
                showimage(i , waitkey=1)
            if len(faces) < 1 : continue
            facelist.extend(face_roi)
            labels.append(label)
    return facelist , np.array(labels)

faceclassifier = cv2.CascadeClassifier(f'haarcascades{os.sep}haarcascade_frontalface_default.xml')
def detect_face(image , confthresh = 30 ):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces , conf = faceclassifier.detectMultiScale2(image=gray, scaleFactor=1.1, minNeighbors=3 , minSize=(40, 40))
    facelist = [faces[i] for i in range(len(faces)) if conf[i] > confthresh]
    face_roi = [gray[y:y+w , x:x+h] for (x , y, w ,h) in facelist]
    return face_roi , facelist

if __name__ == '__main__':
    train_()
