

## JDR-CSTrack

<details open>
<summary>Install</summary>

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```bash
pip install ultralytics
pip install -r requirement.txt
```

For alternative installation methods including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and Git, please refer to the [Quickstart Guide](https://docs.ultralytics.com/quickstart).

#### Python

<details open>
<summary>Train</summary>

```python
from ultralytics import YOLO
import os
def train():
    # Load a model 
    model = YOLO("ultralytics/yolov8x.pt")  # load a pretrained model (recommended for training)
    results = model.train(data='cocovoc/mydata.yaml',epochs=100,imgsz=640,batch=2,save_period=10,amp=False)
if __name__ == "__main__":
    train()

```

<details open>
<summary>Predict</summary>

```Python
from ultralytics import YOLO
import os
def train():
    model = YOLO('ultralytics/pretrained/best_origin.pt')  # load a pretrained model (recommended for training)
    results = model.val(data='cocovoc/mydata.yaml', imgsz=640,batch=4, split='test')
if __name__ == "__main__":
    train()

```

<details open>
<summary>Track</summary>

```Python
import datetime
from collections import defaultdict
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np

from ultralytics import YOLO
# Load the YOLOv8 model
model = YOLO('ultralytics/pretrained/best_origin.pt')
# for video in os.listdir("G:/BaiduNetdiskDownload/DanceTrack/val/"):
with open("test.txt", "r", encoding="utf-8") as file:
# The format of test.txt is "1 2 3 4 5",which indicates the sequence names of videos
    for video in file:
        path=video.strip()
        video_path=os.path.join("shipMOT/data/test",path,path+".mp4")
        if not os.path.exists('ultralytics/runs/track'):
            os.makedirs('ultralytics/runs/track')
        file_path=os.path.join("ultralytics/runs/track", path + ".txt")
        # Open the video file
        # video_path = "shipMOT/data/test/2/2.mp4"
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, height)
        count=0;
        # Store the cbs_track history
        track_history = defaultdict(lambda: [])
        with open(file_path,'a') as file:
            # Loop through the video frames
            while cap.isOpened():
                    # Read a frame from the video
                success, frame = cap.read()

                if success:
                        # Run YOLOv8 tracking on the frame, persisting tracks between frames，choose the tracker：bytetrack/cstrack/botsort
                    results = model.track(frame, persist=True, tracker="ultralytics/cfg/trackers/cstrack.yaml",show=False)
                    count+=1
                    # Get the boxes and cbs_track IDs
                    if results[0].boxes.id != None:
                        boxes1= results[0].boxes.data.cpu()
                        boxes2=results[0].boxes.xywh.cpu()
                        track_ids = results[0].boxes.id.int().cpu().tolist()

                        # Visualize the results on the frame
                        annotated_frame = results[0].plot()

                        # Plot the tracks
                        for box1, track_id,box2 in zip(boxes1, track_ids,boxes2):
                            x, y, w1, h1 ,i,j,k= box1
                            x1,y1,w,h=box2
                            track = track_history[track_id]
                            track.append((float(x), float(y)))  # x, y center point
                            if len(track) > 30:  # retain 90 tracks for 90 frames
                                track.pop(0)
                            file.write(str(count)+','+str(track_id)+','+str(x.item())+','+str(y.item())+','+str(w.item())+','+str(h.item())+','+'1'+','+'0'+','+'1'+'\n')
                            # file.write(str(count)+','+'ship'+','+str(track_id)+','+str(x)[7:14]+','+str(y)[7:14]+','+str(w)[7:14]+','+str(h)[7:14]+'\n')
                            # points = np.hstack(cbs_track).astype(np.int32).reshape((-1, 1, 2))
                            # cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)

                            # Display the annotated frame
                            # cv2.imshow("YOLOv8 Tracking", annotated_frame)

                            # videoWriter.write(annotated_frame)

                            # Break the loop if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                else:
                        # Break the loop if the end of the video is reached
                    break
        file.close()
        # Release the video capture object and close the display window

```

#### Models

<details open>
<summary>Multi_CSP</summary>

JDR-CSTrack/ultralytics/nn/modules/block.py

***replace  # /MutliCSP***

<details open>
<summary>RFB_CA</summary>

JDR-CSTrack/ultralytics/cfg/models/v8

***yolov8l-BasicRFB.yaml***

<details open>
<summary>Wiou</summary>

JDR-CSTrack/ultralytics/utils/loss.py

***replace  # /Wiou***

