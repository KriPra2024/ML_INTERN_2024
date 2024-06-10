import pandas as pd
import numpy as np
import os
import cv2

# Paths
train_csv_path = 'E:/INTERNSHIP-2024/WEEK-1/dataset/train_images/train.csv'
train_images_path = 'E:/INTERNSHIP-2024/WEEK-1/dataset/train_images'
output_path = 'E:/INTERNSHIP-2024/WEEK-1/dataset/masks'

os.makedirs(output_path, exist_ok=True)


df = pd.read_csv(train_csv_path)

def rle_to_mask(rle_string, height, width):
    rows, cols = height, width
    img = np.zeros(rows * cols, dtype=np.uint8)
    if rle_string != -1:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 255
    img = img.reshape((rows, cols), order='F')
    return img

for idx, row in df.iterrows():
    image_id = row['ImageId']
    class_id = row['ClassId']
    rle = row['EncodedPixels']

    if pd.isna(rle):
        continue
    
    mask = rle_to_mask(rle, 256, 1600)
    
    mask_output_path = os.path.join(output_path, f"{image_id}_{class_id}.png")
    cv2.imwrite(mask_output_path, mask)
