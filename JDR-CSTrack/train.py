from ultralytics import YOLO
import os
def train():
    # Load a model 这里选择预训练模型或者网络结构 有不同模型可以选择 我这使用yolov8n.pt
    model = YOLO("ultralytics/yolov8x.pt")  # load a pretrained model (recommended for training)
    # Use the model0 data指定你的yaml文件，然后设置超参数
    results = model.train(data='cocovoc/mydata.yaml',epochs=100,imgsz=640,batch=2,save_period=10,amp=False)
if __name__ == "__main__":
    train()
