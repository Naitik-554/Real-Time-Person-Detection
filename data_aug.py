import os
import imageio
import imgaug.augmenters as iaa

# Path to the folder containing the face images
input_folder = r'data2\modi_aug'
output_folder = r'data2\modi_aug'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the augmentation pipeline
augmentation_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Crop(percent=(0, 0.1)),  # randomly crop images
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),  # apply Gaussian blur to 50% of the images
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20% (per axis)
        rotate=(-25, 25),  # rotate by -25 to +25 degrees
        shear=(-8, 8)  # shear by -8 to +8 degrees
    )
])

# List all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# Augment and save the images
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = imageio.imread(image_path)
    
    for i in range(5):  # Generate 5 augmented images for each input image
        augmented_image = augmentation_pipeline(image=image)
        output_path = os.path.join(output_folder, f'aug_{i}_{image_file}')
        imageio.imwrite(output_path, augmented_image)
