import os
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse

def run(args):
    model_dir = args.model_dir
    image_path = args.image_path
    model_path = os.path.join(model_dir, 'best_model.h5')
    
    print("----------- Starting Inference -----------")
    
    # Check if image path is provided
    if not image_path:
        print("Please provide an image path for inference using --image_path")
        return
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    print(f"Loading and processing image: {image_path}")
    image = Image.open(image_path).resize((30, 30))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict the class
    print("Predicting the class of the traffic sign...")
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    print(f"Predicted Class: {predicted_class}")
    
    print("----------- Inference Complete -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Inference for Traffic Sign Recognition")
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--image_path', type=str, required=True, help='Path of the image to predict')
    args = parser.parse_args()
    run(args)
