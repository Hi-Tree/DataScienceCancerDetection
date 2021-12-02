import flask
import os
import pandas as pd

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        # Get the input from the user.
        user_input_text = flask.request.form['user_input_image']

    
    return flask.render_template('index.html', 
            input_text=user_input_image,
    )

if __name__ == '__main__':
    app.run(debug=True)

