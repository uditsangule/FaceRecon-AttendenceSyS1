import cv2
import os
from tqdm.auto import tqdm
def writeframes(inputvideopath = None , outputpath = f'data' , ext='jpg'):
    if not os.path.exists(inputvideopath):
        print(f"videopath:{inputvideopath} doesn't exists! Default webcam is started !")
        return 1
    cap = cv2.VideoCapture(inputvideopath)
    filename = os.path.basename(inputvideopath).split('.')[0]
    os.makedirs(name=outputpath + os.sep + filename , exist_ok=True)
    for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret , frame = cap.read()
        if (not ret) or cv2.waitKey(1) == ord('q'): break
        cv2.imwrite('{}Img_{}.{}'.format(outputpath+os.sep+filename+os.sep, str(i), ext), frame)
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")
    return 0

if __name__ == '__main__':
    writeframes(inputvideopath='/home/udit/Downloads/0_udit_sangule.mp4')