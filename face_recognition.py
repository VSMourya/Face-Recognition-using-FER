

import cv2
from fer import FER 
import os
import shutil

def createFolder():

    if os.path.exists("images"):
        shutil.rmtree("images")
    
    os.mkdir("images")

    print("Folder created")

def getMaxEmotion(data):

    bbox = data["box"]
    emotion = max(data["emotions"],key= lambda x: data["emotions"][x])
    conf = round(data["emotions"][emotion]*100)

    return emotion,conf,bbox


def main(currDirectory):

    vid = cv2.VideoCapture(0)
    detector = FER(mtcnn=True)
    

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter("video.mp4",fourcc,30,(1920,1080))

    i = 0
    while vid.isOpened():
        i+=1

        ret,frame = vid.read()
        if ret:
            frame = cv2.resize(frame,(1920,1080))
            detections = detector.detect_emotions(frame)

            for detection in detections:

                emotion,conf,bbox = getMaxEmotion(detection)
                x,y,w,h = bbox

                try:
                    cropped = frame[y:y+h,x:x+w]
                    print(f"{emotion} , {conf} --- saved")
                    cv2.imwrite(f"{currDirectory}/images/{i}.jpg",cropped)
                except:
                    continue
                

                #circles
                cv2.circle(frame, (x, y), 20, (0,0,255),3)
                cv2.circle(frame, (x+w, y+h), 20, (0,0,255),3)


                #rectangleBox
                srtPoint = (x,y)
                endPoint = (x+w, y+h)
                colorRec = (0,255,0)
                thicknessRec = 5
                cv2.rectangle(frame, srtPoint, endPoint, colorRec, thicknessRec)

                text = f"{emotion} : {conf}"

                #text
                font = cv2.FONT_HERSHEY_SIMPLEX 
                org = (x-50, y-50) 
                fontScale = 3
                color = (255, 0, 0) 
                thickness = 3
                image = cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA) 


                #writes frame into the videowriter
                vw.write(frame)

            cv2.imshow("frame",frame)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    vid.release()


if __name__ == "__main__":

    createFolder()

    currDirectory = "/home/mourya/Desktop/Personal_Projects/FER"

    main(currDirectory)
