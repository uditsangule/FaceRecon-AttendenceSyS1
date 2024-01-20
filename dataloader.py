import os
import cv2
import numpy as np
from utilities import showimage
def train_(datapath='data' , save_=True , modelpath=f'.{os.sep}facerecon.xml' , reshape=(512,512)):
    faces , labels = get_faces_labels(datapath)
    facerecon = cv2.face.FisherFaceRecognizer.create()
    print(f"training_ contains {len(faces)} faces and {len(labels)} lables ranging from {labels.min()} to {labels.max()}")
    ## convert to (512 ,512) to set one dimm
    faces = [cv2.resize(src=faces[i] , dsize=reshape) for i in range(len(faces))]
    facerecon.train(faces , labels)
    if save_ : facerecon.save(modelpath)
    return facerecon

def predict_( faceroi , model , reshape=(512,512)):
    faceroi = cv2.resize(src=faceroi , dsize=reshape , interpolation=cv2.INTER_LINEAR)
    labeltemp = model.predict_label(faceroi)
    out = {'label':labeltemp , 'confidence':90}
    return out

def get_faces_labels(datapath='data'):
    facelist = []
    labels = []
    for dir_ in os.listdir(datapath):
        if not  dir_.__contains__('S'): continue
        label = int(dir_.replace("S" , ""))
        for imagename in os.listdir(datapath + os.sep + dir_):
            if not imagename.__contains__('.jpg'): continue
            imagepath = os.path.join(datapath,dir_ , imagename)
            image = cv2.imread(imagepath,cv2.COLOR_RGB2BGR)
            if int(image.shape[1] / 512) > 4 or int(image.shape[1] / 512) > 4: image = cv2.resize(src=image, dsize=(
            image.shape[1] // 4 ,image.shape[1] // 4))
            face_roi , faces = detect_face(image)
            for i in face_roi:
                showimage(i , waitkey=1)
            if len(faces) < 1 : continue
            facelist.extend(face_roi)
            labels.append(label)
    return facelist , np.array(labels)

faceclassifier = cv2.CascadeClassifier(f'haarcascades{os.sep}haarcascade_frontalface_default.xml')
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceclassifier.detectMultiScale(image=gray, scaleFactor=1.1, minNeighbors=3)
    face_roi = [gray[y:y+w , x:x+h] for (x , y, w ,h) in faces]
    return face_roi , faces

if __name__ == '__main__':
    train_()
