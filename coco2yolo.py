import json
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import glob
from os.path import dirname, join

current_dir = Path(dirname(__file__)).as_posix()


def cocobbox2yolobbox(bbox,img_w=640,img_h=640):
    limit_precision = lambda x: format(x,".6f")
    
    x,y,w,h = bbox
    x_mid = (x + (x+w))/2
    y_mid = (y + (y+h))/2
    
    # Normalization
    x_mid = limit_precision(x_mid / img_w)
    y_mid = limit_precision(y_mid / img_h)
    w_new = limit_precision(w / img_w)
    h_new = limit_precision(h / img_h)
    
    return [x_mid,y_mid,w_new,h_new]

def readvid_writeimg(source_vid,target_folder):
    cap = cv2.VideoCapture(source_vid)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num2name = lambda x: "0" * (len(str(frame_count)) - len(str(x))) + str(x) 
    _,filename = os.path.split(source_vid)
    
    Path(target_folder).mkdir(parents=True,exist_ok = True)
    
    frame_no = 0
    with tqdm(total=frame_count, desc = f"Processing '{filename}'") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f"{target_folder}/{num2name(frame_no)}.jpg",frame)
                pbar.update(1)
            else:
                break
            frame_no +=1
        cap.release()
        
def cocoformat2yoloformat_annotations(images_root,jsn_file):
    getannbyimgid = lambda img_id, jsn: [ann for ann in jsn["annotations"] if ann["image_id"]==img_id]
    name2id = lambda fname: int(fname.split(".")[0]) + 1 
    fname2ftxt = lambda fname: fname.replace(".jpg",".txt")
    
    bn_json = json.load(open(jsn_file))
    
    Path(images_root.replace("images","labels")).mkdir(parents=True,exist_ok=True)
    
    flist = glob.glob(images_root.replace("labels","images") + "/*.jpg")
    fnamelist = [os.path.split(fp)[-1] for fp in flist]
    fidlist = [name2id(fn) for fn in fnamelist]
    fann = [getannbyimgid(idx,bn_json) for idx in fidlist]
    
    for fpath, fname, fid, fannotations in zip(flist,fnamelist,fidlist,fann):
        txtpath = images_root.replace("images","labels") + f"/{fname2ftxt(fname)}"
        with open(txtpath,"w") as t:
            for item in fannotations:
                t.write(f"{item['category_id']-1} " + " ".join(cocobbox2yolobbox(item["bbox"])) +"\n" )    


source_train = join(current_dir,"challenge/images/train/train.mp4")
source_test = join(current_dir,"challenge/images/test/test.mp4")
source_val = join(current_dir,"challenge/images/val/val.mp4")
target_train = join(current_dir,"challenge_yolo_format/images/train/")
target_test = join(current_dir,"challenge_yolo_format/images/test/")
target_val = join(current_dir,"challenge_yolo_format/images/val/")

assert Path(source_train).is_file(), f"{source_train} doesn't exist. Place video file in {source_train}"
assert Path(source_test).is_file(), f"{source_test} doesn't exist. Place video file in {source_test}"
assert Path(source_val).is_file(), f"{source_val} doesn't exist. Place video file in {source_val}"

readvid_writeimg(source_train,target_train)
cocoformat2yoloformat_annotations(target_train,join(current_dir,"challenge/annotations/instances_train.json"))
readvid_writeimg(source_test,target_test)
cocoformat2yoloformat_annotations(target_test,join(current_dir,"challenge/annotations/instances_test.json"))
readvid_writeimg(source_val,target_val)
cocoformat2yoloformat_annotations(target_val,join(current_dir,"challenge/annotations/instances_val.json"))


yamlcontent = f"""
path: {Path(os.getcwd()).as_posix()}/challenge_yolo_format  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  images/test # (relative to 'path')

# Classes
names:
  0: bolt
  1: nut
"""
with open("challenge_yolo_format/challange.yaml","w") as f:
    f.write(yamlcontent)