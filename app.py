import flask
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from imutils import paths
import os
import numpy as np
import sys
from werkzeug.utils import secure_filename




app = flask.Flask(__name__, template_folder='templates')

with open("models/finalized_model.sav", "rb") as p_f:
	H = pickle.load(p_f)

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        # Get the input from the user.
        file = flask.request.files['user_input_image']
        
        name = secure_filename(file.filename)
        print(name)
        file.save('Dataset/0/'+name)
        os.chmod('Dataset/0/'+name, 0o0777)
        TEST_PATH = 'Dataset'
        BS = 1
       
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
        p = "Cancerous"
        pe = round(100 - H.predict(testGen)[:,1][0] * 100, 2)
        if predIdxs[0] == 0:
            p = "Not Cancerous"
        return flask.render_template('index.html', prediction=p, percent = pe)

@app.route('/graphs/')
def graphs():
    return flask.render_template('graphs.html')

if __name__ == '__main__':
    app.run(debug=True)

