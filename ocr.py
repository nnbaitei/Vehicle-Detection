import cv2 
from matplotlib import pyplot as plt
import numpy as np
import easyocr


# อ่านภาพ
img = cv2.imread("dataset/license/4.jpg")

# ปรับขนาดภาพ (resize image)
scale_factor = 20  # เพิ่มขนาดของภาพขึ้น 20 เท่า
resized_image = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)

# การเพิ่มความคมชัดของภาพ (image sharpening)
sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5.5, -1],
                              [0, -1, 0]])
sharpened_image = cv2.filter2D(resized_image, -1, sharpening_kernel)

# แปลงเป็นภาพสีเทา (grayscale) และลบ noise ด้วย bilateral filter
gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 1, 1, 1)

# เพิ่ม contrast ด้วย CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8,8))
contrast_image = clahe.apply(bfilter)  # Apply to the grayscale image
# การ Erosion เพื่อทำให้เส้นบางลง
kernel = np.ones((3,3), np.uint8)  # ขนาด kernel สามารถปรับได้ เช่น (3,3), (2,2) หรือเล็กกว่า
eroded_edges = cv2.dilate(contrast_image, kernel, iterations=5)  # การทำ Erosion เพื่อลดขนาดของเส้นขอบ

# ลบ noise เพิ่มเติมด้วย Non-local means denoising
denoised_image = cv2.fastNlMeansDenoising(eroded_edges, None, 10, 7, 21)

# ใช้ Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(denoised_image, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 
                                         21, 2)  # Block size and C can be adjusted

# ใช้ Canny Edge Detection
edges = cv2.Canny(denoised_image, 0, 0)

# แสดงผลภาพทั้งหมดเป็น subplot
plt.figure(figsize=(15, 10))

# # ภาพต้นฉบับ
# plt.subplot(2, 3, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')

# ภาพที่ถูกปรับขนาด
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# ภาพที่ถูกเพิ่มความคมชัด
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
plt.title('Sharpened Image')
plt.axis('off')

# ภาพที่ใช้ CLAHE เพิ่ม contrast
plt.subplot(2, 3, 3)
plt.imshow(contrast_image, cmap='gray')
plt.title('Increased Contrast Image')
plt.axis('off')

# ภาพที่ถูกลบ noise ด้วย Non-local means denoising
plt.subplot(2, 3, 4)
plt.imshow(denoised_image, cmap='gray')
plt.title('Denoised Image')
plt.axis('off')

# ภาพที่ใช้ adaptive threshold
plt.subplot(2, 3, 5)
plt.imshow(adaptive_thresh, cmap='gray')
plt.title('Adaptive Threshold Image')
plt.axis('off')

# ภาพที่ใช้ edge detection (Canny)
plt.subplot(2, 3, 6)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detected Image')
plt.axis('off')

plt.tight_layout()

# อ่านป้ายทะเบียนด้วย EasyOCR
reader = easyocr.Reader(['th'], gpu=True)

# adaptive = reader.readtext(adaptive_thresh)
# for detection in adaptive:
#     bbox, text, score = detection
#     text = text.upper().replace(' ', '')
#     print("Adap:", text)
denoise = reader.readtext(contrast_image)
for detection in denoise:
    bbox, text, score = detection
    text = text.upper().replace(' ', '')
    print("Denoiised:", text)
# edge = reader.readtext(edges)
# for detection in edge:
#     bbox, text, score = detection
#     text = text.upper().replace(' ', '')
#     print("Edge:", text)

plt.show()