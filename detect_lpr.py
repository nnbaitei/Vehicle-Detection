import cv2 as cv
import easyocr
import os
from glob import glob
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Pre-trained YOLOv8 model for car recognition
coco_model = YOLO('yolov8n.pt')
# YOLOv8 model trained to detect license plates
np_model = YOLO(r'runs\detect\license_plate_detection\weights\best.pt')

# Read in test video paths
videos = glob(r'D:\Y4\สหกิจ\Vehicle-Detection\car_detect.avi')

# Set up EasyOCR reader
reader = easyocr.Reader(['th'], gpu=True)

# Create folders for saving license plate and vehicle images
output_folder = './detected_license_plates'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to read license plates using EasyOCR
def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        return text, score
    return None, None

# Function to save vehicle images (if not already saved)
def save_vehicle_image(frame_number, track_id, vehicle_img, saved_track_ids):
    if track_id not in saved_track_ids:
        output_path = os.path.join(output_folder, f"vehicle_{track_id}.jpg")
        cv.imwrite(output_path, vehicle_img)
        saved_track_ids.add(track_id)
        return output_path  # Return the image file path for CSV
    return None  # If image was already saved, return None

# Function to save license plate images (if detected)
def save_license_plate_image(track_id, plate_img):
    output_path = os.path.join(output_folder, f"license_plate_{track_id}.jpg")
    cv.imwrite(output_path, plate_img)
    return output_path  # Return the image file path for CSV

# Function to write the results to CSV
def write_csv(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('frame_number,track_id,class,x1,y1,x2,y2,Number_license_plate,Vehicle_image_path,License_plate_image_path\n')
        for frame_number, objects in results.items():
            for track_id, data in objects.items():
                class_name = data['vehicle']['class_name']
                bbox_obj = data['vehicle']['bbox']
                license_plate = data.get('license_plate', None)
                vehicle_image_path = data['vehicle']['image_path']
                license_plate_image_path = data['license_plate'].get('image_path', None)

                if license_plate:
                    f.write('{},{},{},{},{},{},{},{},{}\n'.format(
                        frame_number, track_id, class_name, 
                        bbox_obj[0], bbox_obj[1], bbox_obj[2], bbox_obj[3], 
                        license_plate['number'], vehicle_image_path, license_plate_image_path
                    ))
                else:
                    f.write('{},{},{},{},{},{},{},{}\n'.format(
                        frame_number, track_id, class_name, 
                        bbox_obj[0], bbox_obj[1], bbox_obj[2], bbox_obj[3], 
                        'No License Plate', vehicle_image_path, 'No License Plate Image'
                    ))

def preprocess_image(img):
    scale_factor = 20  # Increase image size by 20 times
    resized_image = cv.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LANCZOS4)

    # Sharpening the image
    sharpening_kernel = np.array([[0, -1, 0],
                                [-1, 5.5, -1],
                                [0, -1, 0]])
    sharpened_image = cv.filter2D(resized_image, -1, sharpening_kernel)

    # Convert to grayscale and remove noise using bilateral filter
    gray = cv.cvtColor(sharpened_image, cv.COLOR_BGR2GRAY)
    bfilter = cv.bilateralFilter(gray, 1, 1, 1)

    # Increase contrast with CLAHE
    clahe = cv.createCLAHE(clipLimit=10, tileGridSize=(8,8))
    contrast_image = clahe.apply(bfilter)
    return contrast_image

# Initialize results dictionary
results = {}
saved_track_ids = set()

# Read video by index
video = cv.VideoCapture(videos[0])

ret = True
frame_number = -1
vehicles = {2: 'car', 5: 'bus', 7: 'truck'}  # Map class_id to vehicle types
frame_width = 1920  # Adjust the width according to your video
frame_height = 1080  # Adjust the height according to your video
output_video_path = 'output_video.avi'
fourcc = cv.VideoWriter_fourcc(*'XVID')  # Codec for video writing
video_writer = cv.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

frame_skip = 5  # จำนวนเฟรมที่จะข้าม
frame_count = 0  # ตัวนับเฟรม

while ret:
    frame_number += 1
    ret, frame = video.read()

    if frame_count % frame_skip == 0:
        results[frame_number] = {}

        # Vehicle detector
        detections = coco_model.track(frame, persist=True, classes=list(vehicles.keys()))[0]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score, class_id = detection
            if int(class_id) in vehicles and score > 0.5:
                vehicle_class = vehicles[int(class_id)]  # Get the class name (car, bus, truck)
                
                # Crop the vehicle image
                vehicle_img = frame[int(y1):int(y2), int(x1):int(x2)]

                # Save the vehicle image (only once per track_id)
                vehicle_image_path = save_vehicle_image(frame_number, track_id, vehicle_img, saved_track_ids)

                # Region of interest (ROI) for license plate detection
                roi = frame[int(y1):int(y2), int(x1):int(x2)]

                # License plate detector for region of interest
                license_plates = np_model(roi)[0]

                for license_plate in license_plates.boxes.data.tolist():
                    plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate

                    # Crop the plate from the region of interest
                    plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                    # Save the license plate image
                    license_plate_image_path = save_license_plate_image(track_id, plate)

                    # Check if the license plate is within the bottom half of the frame
                    if y2 > (frame_height * 0.5):
                        plate_thresh = preprocess_image(plate)
                        # OCR
                        np_text, np_score = read_license_plate(plate_thresh)

                        if np_text is not None:

                            # Store results
                            results[frame_number][track_id] = {
                                'vehicle': {
                                    'class_name': vehicle_class,
                                    'bbox': [x1, y1, x2, y2],
                                    'bbox_score': score,
                                    'image_path': vehicle_image_path
                                },
                                'license_plate': {
                                    'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                                    'bbox_score': plate_score,
                                    'number': np_text,
                                    'text_score': np_score,
                                    'image_path': license_plate_image_path  # Add the license plate image path
                                }
                            }
                        else:
                            results[frame_number][track_id] = {
                                'vehicle': {
                                    'class_name': vehicle_class,
                                    'bbox': [x1, y1, x2, y2],
                                    'bbox_score': score,
                                    'image_path': vehicle_image_path
                                },
                                'license_plate': {
                                    'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                                    'bbox_score': plate_score,
                                    'number': None,
                                    'text_score': None,
                                    'image_path': None  # No license plate image if not detected
                                }
                            }
                        cv.rectangle(frame, (int(x1) + int(plate_x1), int(y1) + int(plate_y1)), 
                                     (int(x1) + int(plate_x2), int(y1) + int(plate_y2)), (255, 0, 0), 2)
                        cv.putText(frame, f'License: {np_text}', (int(x1), int(y2) + 30), 
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
    frame_count = 0  # เมื่อเริ่มต้นใหม่
    video_writer.write(frame)

video_writer.release()
# Save results to CSV
write_csv(results, './results_test.csv')

# Release video capture
video.release()
