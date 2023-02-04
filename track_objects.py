import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def get_stream(source):
    vid = cv2.VideoCapture(source)
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            yield frame
        else:
            break
    vid.release()


detector = YOLO("./runs/detect/train2/weights/best.pt")


for frame in get_stream("./challenge/images/test/test.mp4"):
    #cv2.imshow('frame',frame)
    #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.predict(frame,mode='detect', stream=False,verbose =False)
    for r in results:
        r = r.to('cpu')
        for box in r.boxes:
            if box.conf>0.78:
                x,y,x2,y2 = [int(item.item()) for item in torch.tensor_split(box.xyxy,4,dim=1)]
                cv2.rectangle(frame, (x, y), (x2,y2), (0,255,0), 2)
        

    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    