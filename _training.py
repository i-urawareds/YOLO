import cv2
from ultralytics import YOLO
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    model = YOLO('C:/Users/ikeya/yolo11n.pt')

    """
    .yaml -> train,valの画像のフォルダのパスとクラス数、クラスの名前を格納したファイル

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    result = model.train(
        data = 'D:/detection/yolo/config/vessel.yaml',
        epochs = 100,
        batch = 16,
        workers = 8,
        patience = 30,
        imgsz = 512, #正方形が良さそう
        device = device
    )

    result = model(source='./ikeya_detect/YOLO/',
                   save=True,
                   device = device,
                   conf=0.2,
                   iou=0.5,
                   project = "D:/detection/yolo/output"
                   )

if __name__ == '__main__':
    main()