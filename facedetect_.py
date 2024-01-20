import cv2
import numpy as np
import os
import time
from utilities import showimage
from dataloader import predict_ , detect_face


def drawrect(image , coords , color= (0,255,0)):
    if len(coords) < 1 : return image
    for (x,y,w,h) in coords:
        conf = 80
        cv2.rectangle(image , (x,y) , (x + w , y + h) , color=color , thickness=2)
        face_roi = image[y:y+w , x:x+h]
    return image

def puttext(image,text , pos = (30,30) , color = (255,0,0)):
    return cv2.putText(img=image , text=str(text) , org=pos , fontFace=cv2.FONT_HERSHEY_SIMPLEX ,fontScale=0.5, thickness=1 ,color=color,lineType=cv2.LINE_AA )

def run(inputvideopath = None , _ret= False):
    if inputvideopath is not None:
        if not os.path.exists(inputvideopath):print(f"videopath:{inputvideopath} doesn't exists! Default webcam is started !")
        inputvideopath = None
    cap = cv2.VideoCapture(0 if inputvideopath is None else inputvideopath)
    fcount = 0
    model = cv2.face.FisherFaceRecognizer.create()
    model.read(filename=f'.{os.sep}facerecon.xml')
    faceframe = []
    while True:
        tic = time.time()
        ret , frame = cap.read()
        fw, fh = frame.shape[:2]
        if (not ret) or cv2.waitKey(1) == ord('q'): break
        fcount+=1
        face_roi , faces = detect_face(frame)
        frame = puttext(frame , text=f"fps:{round(1/(time.time() - tic),2)}" , color=(255,0,0))
        for i in range(len(faces)):
            output = predict_(faceroi=face_roi[i] , model=model)
            #frame = puttext(image=frame , text=output['confidence'] , pos=(faces[i][2] , faces[i][3]) ,color=(0,255,255))
            frame = puttext(image=frame , text=output['label'] , pos=(faces[i][0]-10 , faces[i][1]-10), color=(255,0,0))

        cv2.imshow('window', drawrect(frame , faces))
    cap.release()
    cv2.destroyAllWindows()
    return 0

def run_folderpath(datapath = 'data'):
    for dir_ in os.listdir(datapath):
        if not dir_.__contains__('S'): continue
        label = int(dir_.replace("S", ""))
        for imagename in os.listdir(datapath + os.sep + dir_):
            if not imagename.__contains__('.jpg'): continue
            imagepath = os.path.join(datapath, dir_, imagename)
            image = cv2.imread(imagepath , cv2.COLOR_RGB2BGR)
            if image.shape[0] > 512 or image.shape[1] > 512:image = cv2.resize(src=image, dsize=(image.shape[1] // int(image.shape[1]/512), image.shape[1]//int(image.shape[0]/512)))
            _ , faces = detect_face(image)
            #frame = putfps(frame, fps=round(1 / (time.time() - tic), 2))
            image = drawrect(image , faces)
            showimage(image=image , windowname='out' , waitkey=0)

    return

if __name__ == '__main__':
    inputvideo = "/home/udit/Downloads/RARBG.COM.mp4"
    run(inputvideo)