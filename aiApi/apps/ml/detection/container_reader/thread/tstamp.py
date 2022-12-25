import cv2
import time

url = "rtsp://admin:Dtcdvr1176@10.223.1.134/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_POS_MSEC, time.time())
start = time.time()
while cap.isOpened():
    ret, img = cap.read()
    captime=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
    print(captime, start+captime, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start+captime)))