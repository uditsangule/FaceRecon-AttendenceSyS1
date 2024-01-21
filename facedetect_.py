import cv2
import numpy as np
import os
import time
from utilities import showimage
from dataloader import predict_ , detect_face
from datetime import datetime
from pandas import read_csv
def getCurrenttime(format = "%Y%m%d_%H-%M"):
    return str(datetime.now().strftime(format))

def drawrect(image , coords , color= (0,255,0)):
    if len(coords) < 1 : return image
    for (x,y,w,h) in coords:
        conf = 80
        cv2.rectangle(image , (x,y) , (x + w , y + h) , color=color , thickness=2)
        face_roi = image[y:y+w , x:x+h]
    return image

def puttext(image,text , pos = (30,30) , color = (255,0,0)):
    return cv2.putText(img=image , text=str(text) , org=pos , fontFace=cv2.FONT_HERSHEY_SIMPLEX ,fontScale=0.5, thickness=1 ,color=color,lineType=cv2.LINE_AA )

def writereport(outdict , outputpath = 'Attendence.csv'):
    Header = ['RollNo' ,'FirstName','Surname','Status']



def run(inputvideopath = None , _ret= False , scalesize=2 , aspectratio=4/3):
    if inputvideopath is not None:
        if not os.path.exists(inputvideopath):
            print(f"videopath:{inputvideopath} doesn't exists! Default webcam is started !")
            inputvideopath = None
    cap = cv2.VideoCapture(0 if inputvideopath is None else inputvideopath)
    fcount = 0
    model = cv2.face.FisherFaceRecognizer.create()
    model.read(filename=f'.{os.sep}facerecon.xml')
    faceframe = []
    ## reading old data
    Report = read_csv(filepath_or_buffer='AttendenceReport.csv')
    Report[getCurrenttime().split('_')[0]] = Report.apply(lambda x:getCurrenttime().split('_')[1] , axis=1)
    Resultset = set()
    while True:
        tic = time.time()
        ret , frame = cap.read()
        if (not ret) or cv2.waitKey(1) == ord('q'): break
        if int(frame.shape[1] / 512) > scalesize or int(frame.shape[0] / 512) > scalesize:
            frame = cv2.resize(src=frame,dsize=(frame.shape[1] // scalesize,frame.shape[0] // scalesize))
        fw, fh = frame.shape[:2]

        fcount+=1
        face_roi , faces = detect_face(frame)
        frame = puttext(frame , text=f"fps:{round(1/(time.time() - tic),2)}" , color=(255,0,0))
        for i in range(len(faces)):
            output = predict_(faceroi=face_roi[i] , model=model)
            if output is None: continue
            #frame = puttext(image=frame , text=output['confidence'] , pos=(faces[i][2] , faces[i][3]) ,color=(0,255,255))
            frame = puttext(image=frame , text=output['label'] , pos=(faces[i][0]-10 , faces[i][1]-10), color=(255,0,0))
            Report.loc[Report['RollNo']==output['label'],'Status'] = 'P'
            currDT = getCurrenttime().split('_')
            Report.loc[Report['RollNo']==output['label'],currDT[0]] = currDT[1]
        cv2.imshow('window', drawrect(frame , faces))
    cap.release()
    cv2.destroyAllWindows()
    print("SavingReport")
    Report.to_csv(path_or_buf='AttendenceReport.csv' , index=False)
    print("Done")
    return 0

def run_folderpath(datapath = 'data'):
    for dir_ in os.listdir(datapath):
        for imagename in os.listdir(datapath + os.sep + dir_):
            if not imagename.__contains__('.jpg'): continue
            imagepath = os.path.join(datapath, dir_, imagename)
            image = cv2.imread(imagepath , cv2.COLOR_RGB2BGR)
            if int(image.shape[1] / 512) > 2 or int(image.shape[0] / 512) > 2: image = cv2.resize(src=image, dsize=(
            image.shape[1] // 2 ,image.shape[0] // 2))
            _ , faces = detect_face(image)
            #frame = putfps(frame, fps=round(1 / (time.time() - tic), 2))
            image = drawrect(image , faces)
            showimage(image=image , windowname='out' , waitkey=0)

    return

if __name__ == '__main__':
    inputvideo = "/home/udit/Downloads/28_vinayak_malviya.mp4"
    run(inputvideopath=inputvideo)
    #run_folderpath()