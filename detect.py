import cv2 as cv
from glob import glob
import os
import random
from ultralytics import YOLO

# read in video paths
videos = glob("dataset/car.mp4")
print(videos)

model_pretrained = YOLO('yolov5s.pt')

video = cv.VideoCapture(videos[0])

# get video dims
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('car_detect.avi', fourcc, 20.0, size)

# read frames
ret = True

while ret:
    ret, frame = video.read()

    if ret:
        # detect & track objects
        results = model_pretrained.track(frame, persist=True, classes=[2, 7])

        # plot results
        composed = results[0].plot()

        # save video
        out.write(composed)

out.release()
video.release()