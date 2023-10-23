import gradio as gr
import tensorflow as tf
import gdown
import numpy as np
from keras.models import load_model
import os

URL = 'https://drive.google.com/file/d/1-1wyt9PG0g1ORgUoxjZY68rGcBub2wi4/view?usp=sharing'
output_path = 'classlabel.txt'
gdown.download(URL, output_path, quiet=False,fuzzy=True)

with open(output_path,'r') as file:
    CATEGORIES = [x.strip() for x in file.readlines()]

IMG_SIZE = (160, 160)

model_path = 'best_mobilenet.h5'
model = load_model(model_path)

def classify_image(image):
    # Preprocess the image (resize, normalize, etc.)
    image = tf.image.resize(image, IMG_SIZE)

    image = image.numpy().astype("uint8")
    image = tf.expand_dims(image,0)
    predictions = model.predict(image)

    # Get the top 3 predictions
    top3_indices = predictions[0].argsort()[-3:][::-1]
    top3_labels = [CATEGORIES[i] for i in top3_indices]
    top3_probabilities = predictions[0][top3_indices].tolist()

    # Create a dictionary to return top 3 labels and probabilities
    results = {}
    for label, prob in zip(top3_labels, top3_probabilities):
        results[label] = prob

    print(results)

    return results

# 11, 26, 179
path  = [['0067.jpg'], ['0150.jpg'], ['1075.jpg']]

gr.Interface(
    classify_image, 
    gr.inputs.Image(type='pil', label="Upload your image"),
    outputs='label',
    title="Bird Classification Model using MobileNet V2",
    description="Upload an image to get top 3 classifications.",
    examples=path
).launch(debug=True)