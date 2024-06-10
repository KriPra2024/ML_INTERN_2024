from imgaug import augmenters as iaa
import cv2
import os

def augment_images(input_folder, output_folder, n_augmentations=5):
    os.makedirs(output_folder, exist_ok=True)
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.Affine(rotate=(-20, 20)),
        iaa.Multiply((0.8, 1.2))
    ])
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(input_folder, filename))
            for i in range(n_augmentations):
                augmented_image = aug.augment_image(img)
                cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"), augmented_image)
augment_images('E:/INTERNSHIP-2024/WEEK-1/dataset/masks','E:/INTERNSHIP-2024/WEEK-1/processed_images/train')
