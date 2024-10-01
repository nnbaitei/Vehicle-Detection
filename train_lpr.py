from ultralytics import YOLO

if __name__ == '__main__':
    # โหลดโมเดล YOLOv8
    model = YOLO(r'runs\detect\license_plate_detection4\weights\best.pt')  # หรือเลือกรุ่นอื่น

    # เริ่มการเทรน
    model.train(data='D:/Y4/สหกิจ/Vehicle-Detection/data.yaml', epochs=100, imgsz=640, batch=16, name='license_plate_detection', save_period=5)

# Test Model
# import cv2 as cv
# from glob import glob
# import os
# import random
# from ultralytics import YOLO

# # read in video paths
# videos = glob("dataset/car.mp4")
# print(videos)

# model_pretrained = YOLO(r'runs\detect\license_plate_detection4\weights\best.pt')

# video = cv.VideoCapture(videos[0])

# # get video dims
# frame_width = int(video.get(3))
# frame_height = int(video.get(4))
# size = (frame_width, frame_height)

# # Define the codec and create VideoWriter object
# fourcc = cv.VideoWriter_fourcc(*'DIVX')
# out = cv.VideoWriter('lpr_detect.avi', fourcc, 20.0, size)

# # read frames
# ret = True

# while ret:
#     ret, frame = video.read()

#     if ret:
#         # detect & track objects
#         results = model_pretrained.track(frame, persist=True)

#         # plot results
#         composed = results[0].plot()

#         # save video
#         out.write(composed)

# out.release()
# video.release()