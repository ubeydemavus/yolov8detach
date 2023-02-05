import cv2
import torch
import numpy as np 
import motmetrics as mm
import json
from os.path import dirname, join
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm

current_dir = Path(dirname(__file__)).as_posix()

# read annotation to calculate tracking metrics MOTA and MOTP
jsn = json.load(open(join(current_dir, "challenge/annotations/instances_test.json")))
getannbyframeid = lambda frame_id, jsn: [ann for ann in jsn["annotations"] if ann["image_id"]==(frame_id+1)]
getboxesinframe = lambda ann: [box['bbox'] for box in ann]
gettrackidsinframe = lambda ann: [box['track_id'] for box in ann]


np.float = np.float32 # hot fix deepsort package: backward compatibility issues in deepsort package with newer version of numpy

def yolo2cocobbox(bbox,  image_w, image_h):
    """
    converts yolo bbox to coco bbox format. Needed for deepsort package.
    """
    x_center, y_center, w, h = bbox
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    return [x1, y1, w, h]

def get_stream(vid):
    """Take a source and turn it into a stream."""
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            yield frame
        else:
            break
    vid.release()

# Load the model which is trained for bolts and nuts dataset.
detector = YOLO(join(current_dir,"runs/detect/train2/weights/best.pt"))

# initialize deepsort
tracker = DeepSort(max_age=5,
                n_init=2,
                nms_max_overlap=1.0,
                max_cosine_distance=0.3,
                nn_budget=None,
                #override_track_class=None,
                #embedder="clip_RN50", # defaultsto mobilenetv3
                half=True,
                bgr=True,
                embedder_gpu=True, 
                #embedder_model_name=None,
                #embedder_wts="RN50.pt", # path to weights.
                polygon=False,
                today=None)

tracker.__dict__['tracker'].kf.__dict__['_std_weight_position'] = 1  # hot fix position and velocity weights, deepsort doesnt provide an interface to change kalman filter, or its parameters.
tracker.__dict__['tracker'].kf.__dict__['_std_weight_velocity'] = 3  # (trained model measurements is good enough -> kalman model uncertainty should be high).

# challange test file.
vid = cv2.VideoCapture(join(current_dir,"challenge/images/test/test.mp4"))
frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)

# mot metrics
acc = mm.MOTAccumulator(auto_id=True)


for idx, frame in enumerate(tqdm(get_stream(vid),total = frame_count, desc = "Processing: ")):
    results = detector.predict(frame,mode='detect', stream=False,verbose =False)
    for r in results:
        r = r.to('cpu')
        tracker_list = []
        for box in r.boxes:
            if box.conf>0.75: # only pass objects where confidence higher than 0.75 to deepsort
                box_xywh = [item.item() for item in torch.tensor_split(box.xywhn,4,dim=1)]
                box_xywh = yolo2cocobbox(box_xywh,frame.shape[0],frame.shape[1])
                box_cls = int(box.cls.item())
                box_conf = box.conf
                tracker_list.append((box_xywh,box_conf,box_cls))
        

        if len(tracker_list)>0: 
            # if there is detection update kalman filter. 
            # This is not ideal, you want to update kalman filter every step with or without measurements but deepsort package implementation is problematic.
            tracks = tracker.update_tracks(tracker_list, frame=frame)

            tracks_boxes = []
            track_ids = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x,y,w,h = map(int,ltrb)
                
                # record for MOT metrics.
                tracks_boxes.append(list(map(int,ltrb)))
                track_ids.append(track_id)
                
                # draw bbox and related info onto the frame.
                cv2.rectangle(frame, (x, y), (w,h), (0,255,0), 2)
                cv2.putText(frame,detector.names[track.det_class] + " (T. id: " + str(track_id) + ")", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            
            # Accumulate MOT metrics.
            ground_truth_ann = getannbyframeid(idx,jsn)
            gt_boxes = getboxesinframe(ground_truth_ann)
            gt_track_ids = gettrackidsinframe(ground_truth_ann)
            distance_matrix = mm.distances.iou_matrix(np.array(gt_boxes),np.array(tracks_boxes), max_iou=0.7)
            acc.update(
                        gt_track_ids,               # Ground truth object ID in this frame
                        track_ids,                  # Detector hypothesis ID in this frame
                        [distance_matrix])

    # draw the frame.
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate metrics and print.
mh = mm.metrics.create()
report = mh.compute(acc, metrics=['num_frames', 'num_objects','num_matches' ,'mota','motp'], name='acc')

print(report) 