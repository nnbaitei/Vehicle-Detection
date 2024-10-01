import ast
import cv2 as cv
import easyocr
from glob import glob
import numpy as np
import pandas as pd
import string
from ultralytics import YOLO
import matplotlib.pyplot as plt

# regular pre-trained yolov8 model for car recognition
# coco_model = YOLO('yolov8n.pt')
coco_model = YOLO('yolov8n.pt')
# yolov8 model trained to detect number plates
np_model = YOLO(r'runs\detect\license_plate_detection\weights\best.pt')

# read in test video paths
videos = glob(r'D:\Y4\สหกิจ\Vehicle-Detection\car_detect.avi')

reader = easyocr.Reader(['th'], gpu=True)

def read_license_plate(license_plate_crop):
    # plt.imshow(license_plate_crop)
    # plt.show()
    detections = reader.readtext(license_plate_crop)
    # print("-----------------------", detections)
    
    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')
        # print("-----------------------", text)
        
        return text, score

    return None, None

def write_csv(results, output_path):
    
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format(
            'frame_number', 'track_id', 'car_bbox', 'car_bbox_score',
            'license_plate_bbox', 'license_plate_bbox_score', 'license_plate_number',
            'license_text_score'))

        for frame_number in results.keys():
            for track_id in results[frame_number].keys():
                # print(results[frame_number][track_id])
                if 'car' in results[frame_number][track_id].keys() and \
                   'license_plate' in results[frame_number][track_id].keys() and \
                   'number' in results[frame_number][track_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{},{}\n'.format(
                        frame_number,
                        track_id,
                        '[{} {} {} {}]'.format(
                            results[frame_number][track_id]['car']['bbox'][0],
                            results[frame_number][track_id]['car']['bbox'][1],
                            results[frame_number][track_id]['car']['bbox'][2],
                            results[frame_number][track_id]['car']['bbox'][3]
                        ),
                        results[frame_number][track_id]['car']['bbox_score'],
                        '[{} {} {} {}]'.format(
                            results[frame_number][track_id]['license_plate']['bbox'][0],
                            results[frame_number][track_id]['license_plate']['bbox'][1],
                            results[frame_number][track_id]['license_plate']['bbox'][2],
                            results[frame_number][track_id]['license_plate']['bbox'][3]
                        ),
                        results[frame_number][track_id]['license_plate']['bbox_score'],
                        results[frame_number][track_id]['license_plate']['number'],
                        results[frame_number][track_id]['license_plate']['text_score'])
                    )
                if 'truck' in results[frame_number][track_id].keys() and \
                   'license_plate' in results[frame_number][track_id].keys() and \
                   'number' in results[frame_number][track_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{},{}\n'.format(
                        frame_number,
                        track_id,
                        '[{} {} {} {}]'.format(
                            results[frame_number][track_id]['truck']['bbox'][0],
                            results[frame_number][track_id]['truck']['bbox'][1],
                            results[frame_number][track_id]['truck']['bbox'][2],
                            results[frame_number][track_id]['truck']['bbox'][3]
                        ),
                        results[frame_number][track_id]['truck']['bbox_score'],
                        '[{} {} {} {}]'.format(
                            results[frame_number][track_id]['license_plate']['bbox'][0],
                            results[frame_number][track_id]['license_plate']['bbox'][1],
                            results[frame_number][track_id]['license_plate']['bbox'][2],
                            results[frame_number][track_id]['license_plate']['bbox'][3]
                        ),
                        results[frame_number][track_id]['license_plate']['bbox_score'],
                        results[frame_number][track_id]['license_plate']['number'],
                        results[frame_number][track_id]['license_plate']['text_score'])
                    )
        f.close()

results = {}

# read video by index
video = cv.VideoCapture(videos[0])

ret = True
frame_number = -1
vehicles = [2, 5, 7]
frame_height, frame_width, _ = video.shape

# read the 10 first frames
while ret:
    frame_number += 1
    ret, frame = video.read()
    

    if ret and frame_number < 100:
        results[frame_number] = {}
        
        # vehicle detector
        detections = coco_model.track(frame, persist=True, classes=[2, 5, 7])[0]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score, class_id = detection
            if int(class_id) in vehicles and score > 0.5:
                vehicle_bounding_boxes = []
                vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])
                for bbox in vehicle_bounding_boxes:
                    # print(bbox)
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    # license plate detector for region of interest
                    license_plates = np_model(roi)[0]
                    
                    # process license plate
                    for license_plate in license_plates.boxes.data.tolist():
                        plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate

                        # cv.rectangle(frame, (int(x1 + plate_x1), int(y1 + plate_y1)), 
                        #   (int(x1 + plate_x2), int(y1 + plate_y2)), 
                        #   (0, 255, 0), 2)  # Green rectangle
                        # plt.imshow(frame)
                        # plt.show()
                        # print(plate_y1, frame.shape)
                        # Crop the plate from the region of interest
                        plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                        
                        # Check if the license plate is within the bottom half of the ROI
                        if y2 > (frame_height / 2):
                            
                            # de-colorize
                            plate_gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
                            
                            # Show the cropped license plate for debugging
                            
                            # posterize
                            # _, plate_treshold = cv.threshold(plate_gray, 0, 200, cv.THRESH_BINARY_INV)
                            plate_treshold = cv.adaptiveThreshold(plate_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 4)
                            # plt.imshow(plate_treshold)
                            # plt.show()
                            # OCR
                            np_text, np_score = read_license_plate(plate_treshold)
                            print("-----------------------", np_text)
                            # if plate could be read write results
                            if np_text is not None:
                                results[frame_number][track_id] = {
                                    'car': {
                                        'bbox': [x1, y1, x2, y2],
                                        'bbox_score': score
                                    },
                                    'license_plate': {
                                        'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                                        'bbox_score': plate_score,
                                        'number': np_text,
                                        'text_score': np_score
                                    }
                                }


write_csv(results, './results.csv')
video.release()