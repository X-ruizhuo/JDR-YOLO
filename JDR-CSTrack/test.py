# from ultralytics import YOLO
# if __name__ == '__main__':
#     model = YOLO('ultralytics/best_all.pt')
#     results=model.cbs_track(source='H:/shipMOT/data/test/2/2.mp4',
#                 imgsz=640,
#                 project='runs/cbs_track',
#                 name='exp',
#                 save=False,
#                 show=False,
#                 tracker="ultralytics/cfg/trackers/bytetrack.yaml",
#                 conf=0.1,
#                 iou=0.5)
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
with open("D:/test.txt", "r", encoding="utf-8") as file:
    # 读取所有行
    for video in file:
        path=video.strip()
        video_path=os.path.join("H:/shipMOT/data/test",path,path+".mp4")
        if not os.path.exists('I:/ultralytics/runs/track'):
            os.makedirs('I:/ultralytics/runs/track')
        file_path=os.path.join("I:/ultralytics/runs/track", path + ".txt")
        # Open the video file
        # video_path = "H:/shipMOT/data/test/2/2.mp4"
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
                        # Run YOLOv8 tracking on the frame, persisting tracks between frames
                    results = model.track(frame, persist=True, tracker="ultralytics/cfg/trackers/cbstrack.yaml",show=False)
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
