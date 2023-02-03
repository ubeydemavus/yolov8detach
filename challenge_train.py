from ultralytics import YOLO
import torch
from pathlib import Path
# Load a model

model = YOLO("yolov8n.pt") 

if torch.cuda.is_available():
    device = 0
else:
    device = 'cpu'


if __name__ == '__main__':
    # transfer learning for 50 epochs.
    results = model.train(data="challenge_yolo_format/challange.yaml", epochs=1, optimizer="Adam", freeze=22, cache="disk", image_weights=True, batch=64, workers=8, pretrained=True, lr0=0.01, device=device)  # train the model
    results = model.val()  # evaluate model performance on the validation set
