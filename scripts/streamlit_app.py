import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse

def run(args):
    model_path = args.model_path
    
    print("----------- Starting Streamlit App -----------")
    st.title("Traffic Sign Recognition App")
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).resize((30, 30))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predict the class
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        st.write(f"Predicted Class: {predicted_class}")
        print(f"Predicted Class: {predicted_class}")
    
    print("----------- Streamlit App Running -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlit App for Traffic Sign Recognition")
    parser.add_argument('--model_path', type=str, default='models/best_model.h5', help='Path of the trained model')
    args = parser.parse_args()
    run(args)
