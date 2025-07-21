# =============================================================================================================
# ================================ PRE PROCESSING OF ALL IMAGES ON THE DATASET ================================
# =============================================================================================================
# = As i saw in the eda.ipynb the images of each class have different dimensions, so i reshape them using the 
# following technique.
# =============================================================================================================
# = Technique used: padding + reshape
# Padding allow to preserve the original proportion of each animal, so that images are not flattened.
# The reshape dimension i used 224x244 (as to be a square for the CNN)
# =============================================================================================================
# =============================================================================================================

import os
import shutil
import random
from pathlib import Path
from PIL import Image

# =============================================================================
#  Preprocessing Function
# =============================================================================
def preprocess_image(source_path, dest_path, target_size=(224, 224)):
    """
    Opens an image, applies padding to make it square, resizes
    and saves it in the destination.

    Args:
        source_path (str or Path): Path of the original image.
        dest_path (str or Path): Path to save the processed image.
        target_size (tuple): Final size of the image (width, height).
    """
    try:
        img = Image.open(source_path)

        # Conversion to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # New dimension
        original_width, original_height = img.size
        
        # Find the larger dimension between width and height to build the square
        max_dim = max(original_width, original_height)
        
        # Create a new image with black background
        padded_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
        
        # Calculate the position to put the image on the center
        paste_x = (max_dim - original_width) // 2
        paste_y = (max_dim - original_height) // 2
        
        padded_img.paste(img, (paste_x, paste_y))
        
        # Reshape the square image for the target lenght (using an high quality filter)
        # Image.Resampling.LANCZOS 
        resized_img = padded_img.resize(target_size, Image.Resampling.LANCZOS)
        
        resized_img.save(dest_path)

    except Exception as e:
        print(f"Failed to process {source_path}: {e}")


# =============================================================================
#  2. Function to split the data(train/validation/test) and process the dataset (all classes)
# =============================================================================
def process_and_split_dataset(source_dir, processed_dir, split_ratios=(0.8, 0.1, 0.1)):
    """
    It goes through the raw data directory, divides the files into training, validation and test,
    applies the pre-processing and saves them in the correct folder structure.

    Args:
        source_dir (str): Directory with the folders for each class (raw data).
        processed_dir (str): Directory where the processed data will be stored.
        split_ratios (tuple): Ratio for (train, validation, test). The sum must be 1.
    """
    source_path = Path(source_dir)
    processed_path = Path(processed_dir)

    train_path = processed_path / 'train'
    val_path = processed_path / 'validation'
    test_path = processed_path / 'test'

    # Create the train/validation/test folders
    for path in [train_path, val_path, test_path]:
        path.mkdir(parents=True, exist_ok=True)

    # Name of the folders of each class
    class_names = [d.name for d in source_path.iterdir() if d.is_dir()]

    print(f"Found {len(class_names)} classes: {class_names}")

    for class_name in class_names:
        print(f"\nProcessing the class: '{class_name}'")
        
        # Create train/test/validation folders for each class
        (train_path / class_name).mkdir(exist_ok=True)
        (val_path / class_name).mkdir(exist_ok=True)
        (test_path / class_name).mkdir(exist_ok=True)

        class_source_path = source_path / class_name
        image_files = list(class_source_path.glob('*.[jp][pn]g')) + list(class_source_path.glob('*.jpeg'))
        
        # Randomly shuffle the images
        random.shuffle(image_files)

        total_images = len(image_files)
        train_end = int(total_images * split_ratios[0])
        val_end = train_end + int(total_images * split_ratios[1])

        # Split the files
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        datasets = {
            'train': (train_files, train_path),
            'validation': (val_files, val_path),
            'test': (test_files, test_path)
        }

        # Process and save each image
        for split_name, (files, dest_dir_base) in datasets.items():
            dest_class_dir = dest_dir_base / class_name
            print(f"Processing {len(files)} images for '{split_name}'folder.")
            
            for source_image_path in files:
                dest_image_path = dest_class_dir / source_image_path.name
                preprocess_image(source_image_path, dest_image_path)
                
    print("\nProcessing ended.")


if __name__ == '__main__':
    SOURCE_DIRECTORY = r'C:\\Users\\Miguel António\\Desktop\\PORTFOLIO\\image_classifier\\data\\raw\\archive (2)\\raw-img'
    PROCESSED_DIRECTORY = r'C:\\Users\\Miguel António\Desktop\\PORTFOLIO\\image_classifier\\data\\processed'
    
    # Ratio of division: 80% train, 10% validation, 10% test
    SPLIT_RATIOS = (0.8, 0.1, 0.1)
    
    process_and_split_dataset(SOURCE_DIRECTORY, PROCESSED_DIRECTORY, SPLIT_RATIOS)