from ultralytics import YOLO
import torch

model = YOLO("yolov8n.pt") # select model atchitecture (also downloads pretrained network if not available)

if torch.cuda.is_available(): # check if cuda is available
    device = 0 
else:
    device = 'cpu'


if __name__ == '__main__':
    # Transfer learning of yolo v8 nano network
    # The API implements data augmentation automatically, one can customize options which are describes in "ultralytics\yolo\cfg\default.yaml" as he wants.
    # data described in challange.yaml, trained with Adam opt. for 80 epoch while freezing every layer except last layer. Cache data to disk to improve training time.
    results = model.train(data="challenge_yolo_format/challange.yaml", epochs=80, optimizer="Adam", freeze=22, cache="disk", image_weights=True, batch=64, workers=8, pretrained=True, lr0=0.001, device=device)  # train the model
    results = model.val()  # evaluate model performance on the validation set
