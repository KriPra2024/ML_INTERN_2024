import cv2
import os
def resize_images(input_folder, output_folder, size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(input_folder, filename))
            resized_img = cv2.resize(img, size)
            cv2.imwrite(os.path.join(output_folder, filename), resized_img)

resize_images('E:/INTERNSHIP-2024/WEEK-1/dataset/masks','E:/INTERNSHIP-2024/WEEK-1/processed_images/train')
