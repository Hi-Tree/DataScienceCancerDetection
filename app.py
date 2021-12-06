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
print(os.path.exists("models/finalized_mod.pkl")) # this prints out true

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        # Get the input from the user.
        user_input_text = flask.request.form['user_input_image']

        BASE_PATH = "static/datasets"
        TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
        BS = 32
        H = pickle.load(open("models/finalized_mod.pkl","rb"))
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
        predIdxs = image_model.predict(x=testGen, steps=(totalTest // BS) + 1)
        predIdxs = np.argmax(predIdxs, axis=1)
        print(classification_report(testGen.classes, predIdxs,
	    target_names=testGen.class_indices.keys()))

    
    return flask.render_template('index.html', 
            input_text=user_input_image,
    )

if __name__ == '__main__':
    app.run(debug=True)

