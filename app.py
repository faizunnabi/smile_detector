import cv2
import pandas as pd
from _datetime import datetime

face_classifier = cv2.CascadeClassifier('haarcascade_frontface.xml');
smile_classifier = cv2.CascadeClassifier('haarcascade_smile.xml');
times=[]
smile_ratios=[]
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_img = img[y:y + h, x:x + w]
        smile = smile_classifier.detectMultiScale(roi_gray, scaleFactor=1.2,
                                                  minNeighbors=22,
                                                  minSize=(25, 25))
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_img, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
            sm_ratio = str(round(sw / sx, 3))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Smile meter : ' + sm_ratio, (10, 50), font, 1, (200, 255, 155), 2, cv2.LINE_AA)
            if float(sm_ratio)>1.8:
                smile_ratios.append(float(sm_ratio))
                times.append(datetime.now())
    cv2.imshow('Smile Detector', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

ds={'smile_ratio':smile_ratios,'times':times}
df=pd.DataFrame(ds)
df.to_csv('smile_records.csv')
cap.release()
cv2.destroyAllWindows()
