import flask
import os
import pickle
import pandas as pd
from skimage import io
from skimage import transform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

app = flask.Flask(__name__, template_folder='templates')

with open("finalized_model.sav", "rb") as p_f:
	H = pickle.load(p_f)

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        # Get the input from the user.
        file = flask.request.files['user_input_image']

        TEST_PATH = "Datasets/testing"
        BS = 32
        totalTest = len(list(paths.list_images(TEST_PATH)))
        valAug = ImageDataGenerator(rescale=1 / 255.0)

        testGen = valAug.flow_from_directory(
	        TEST_PATH,
	        class_mode="categorical",
	        target_size=(48, 48),
	        color_mode="rgb",
	        shuffle=False,
	        batch_size=BS)

        testGen.reset()
        predIdxs = H.predict(x=testGen, steps=(totalTest // BS) + 1)
        predIdxs = np.argmax(predIdxs, axis=1)
        print(classification_report(testGen.classes, predIdxs,
	    target_names=testGen.class_indices.keys()))

        return (flask.render_template('index.html'))

if __name__ == '__main__':
    app.run(debug=True)

