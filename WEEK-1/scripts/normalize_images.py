import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

def normalize_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    normalized_image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
    normalized_image = array_to_img(normalized_image_array)
    return normalized_image

def normalize_images(input_folder, output_folder, target_size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            normalized_image = normalize_image(input_path, target_size)
            output_path = os.path.join(output_folder, filename)
            normalized_image.save(output_path)
            print(f"Normalized and saved {filename} to {output_folder}")


normalize_images('E:/INTERNSHIP-2024/WEEK-1/processed_images/train', 'E:/INTERNSHIP-2024/WEEK-1/normalized_images/train')
