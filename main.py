#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf

# Load the model with the custom object
model = load_model("./Training/model/Leaf Deases(96,88).h5")


# model = load_model("./Training/model/Leaf Deases(96,88).h5", custom_objects=custom_objects)


#Name of Classes
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

#Setting Title of App
st.title("Plant Disease Classification")
st.markdown("Upload an image of the plant leaf")

#Uploading the image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict Disease')

#On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (224, 224))
        
        #Convert image to 4 Dimensions
        opencv_image = np.expand_dims(opencv_image, axis=0)
        
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]

        # Splitting the result to display condition and type separately
        if '-' in result:
            plant_type, condition = result.split('-')
            display_text = f"This is {plant_type} leaf with {condition}"
        else:
            display_text = f"Detected class: {result}"

        st.title(display_text)