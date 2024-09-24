import cv2

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ไม่สามารถเปิดไฟล์วิดีโอได้")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps != 0 else 0

    cap.release()

    print(f"ความละเอียดของวิดีโอ: {width} x {height} pixels")
    print(f"เฟรมต่อวินาที (FPS): {fps}")
    print(f"จำนวนเฟรมทั้งหมด: {total_frames}")
    print(f"ความยาวของคลิป: {duration:.2f} วินาที")

video_path = r"dataset/car.mp4"
get_video_properties(video_path)

'''
ความละเอียดของวิดีโอ: 1920x1080 px
จำนวนเฟรมต่อวินาที (FPS): 30.0
จำนวนเฟรมทั้งหมด: 512
ความยาวของคลิป: 17.07 วินาที
'''