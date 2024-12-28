import cv2
import numpy as np
import io
from PIL import Image
import requests
import os
from datetime import datetime

def ensure_output_dir():
    """Ensure the output directory exists"""
    output_dir = "processed_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def download_and_preprocess_image(image_url):
    """Download and preprocess image from URL"""
    output_dir = ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Download image
    response = requests.get(image_url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Save original image
    original_path = os.path.join(output_dir, f"{timestamp}_original.png")
    cv2.imwrite(original_path, original_img)
    
    # Apply preprocessing steps and save intermediate results
    processed = process_image_for_ocr(original_img, output_dir, timestamp)
    
    # Convert back to bytes for Azure
    success, encoded_img = cv2.imencode('.png', processed)
    return io.BytesIO(encoded_img.tobytes())

def rotate_image_to_center(img):
    """Rotate the image to correct the orientation and center the document."""
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Use the Hough Transform to detect edges and lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        # 3. Calculate the angle of rotation based on the detected lines
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90  # Convert from radians to degrees
            angles.append(angle)

        # 4. Find the average angle to rotate
        median_angle = np.median(angles)

        # 5. Get the rotation matrix
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # 6. Rotate the image
        rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated
    else:
        # If no lines are detected, return the original image
        return img

def process_image_for_ocr(img, output_dir, timestamp):
    """Apply various preprocessing techniques and save intermediate steps"""
    
    img = rotate_image_to_center(img)

    # 1. Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, f"{timestamp}_1_gray.png"), img)


    # Histogram equalization to improve contrast
    img = cv2.equalizeHist(img)
    cv2.imwrite(os.path.join(output_dir, f"{timestamp}_21_threshold.png"), img)


    # 2. Apply thresholding
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite(os.path.join(output_dir, f"{timestamp}_2_threshold.png"), img)
    
    # 3. Remove noise
    img = cv2.fastNlMeansDenoising(img)
    cv2.imwrite(os.path.join(output_dir, f"{timestamp}_3_denoised.png"), img)
    
    # 4. Enhance contrast
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = img.apply(img)
    cv2.imwrite(os.path.join(output_dir, f"{timestamp}_4_enhanced.png"), img)
    
    # 5. Sharpen image
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    cv2.imwrite(os.path.join(output_dir, f"{timestamp}_5_sharpened.png"), sharpened)
    
    # 6. Dilate to enhance digit connectivity
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(sharpened, kernel, iterations=1)
    cv2.imwrite(os.path.join(output_dir, f"{timestamp}_6_dilated.png"), dilated)
    
    # 7. Apply Gaussian blur
    final = cv2.GaussianBlur(dilated, (3,3), 0)
    cv2.imwrite(os.path.join(output_dir, f"{timestamp}_7_final.png"), final)
    
    return enhanced

