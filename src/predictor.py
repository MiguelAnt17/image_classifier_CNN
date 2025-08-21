import numpy as np
from PIL import Image
import argparse
import yaml  
import os
import tensorflow as tf

def predict_image(model_path, image_path, config):
    """
    Loads a trained model and classifies a single image
    using settings from a file.
    """
    # GET THE PARAMETERS
    img_size = tuple(config['image_size']) 
    class_names = config['class_names']
    
    # UPLOAD TRAINED MODEL
    print(f"Uploading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return
    model = tf.keras.models.load_model(model_path)

    # IMAGE PROCESS
    print(f"Processing the image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return
        
    img = Image.open(image_path).convert('RGB').resize(img_size) 
    img_array = np.array(img)
    
    
    img_array = np.expand_dims(img_array, axis=0) 

    # PREDICT
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) 

    # RESULT
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    print("\n--- Prediction Result ---")
    print(f"The image as classified as: '{predicted_class}'")
    print(f"Prediction Confidense: {confidence:.2f}%")
    print("-----------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify an animal image using a trained model and a configuration file.")
    parser.add_argument("--model", type=str, required=True, help="Path to .keras model file.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to classify.")
    parser.add_argument("--config", type=str, default="config/data_config.yaml", help="Path to YAML configuration file.")
    
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        exit()
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        exit()

    predict_image(args.model, args.image, config)


# python src/predictor.py --model "C:\Users\Miguel António\Desktop\PORTFOLIO\image_classifier\experiments\2025-08-12_11-29-39\best_model.keras" --image "C:\Users\Miguel António\Desktop\PORTFOLIO\image_classifier\data\processed\test\cane\OIP-_JSASsAoNCi8Yi31z3u7SgHaJ4.jpeg" --config "C:\Users\Miguel António\Desktop\PORTFOLIO\image_classifier\config\data_config.yaml"