import flask
import pickle
import pandas as pd
import keras

# Import des pickles des models entrainés (vecCount, tfidf, TensorFlow)

vecCount = pickle.load(open('./model/countVec.pkl', "rb"))
tfidf = pickle.load(open('./model/tfidf.pkl', "rb"))
model = keras.models.load_model('./model/tensorFlow.h5')

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        # Extract the input
        question = flask.request.form['question']

        df_question = pd.Series([question])

        # Get the tag prediction
        tags = vecCount.transform(df_question)
        tags = tfidf.transform(tags)

        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'question':question},
                                     result="coucou",
                                     )

if __name__ == '__main__':
    app.run()
