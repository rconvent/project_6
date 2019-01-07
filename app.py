import flask
import pickle
import pandas as pd


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout



# Model keras mise en place du réseau + load des poids entrainés

model = Sequential()
model.add(Dense(2000, activation='relu', input_dim=2000))
model.add(Dropout(0.1))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(31, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model = model.load_weights('./model/my_model.h5')

# Import des pickles des models entrainés (vecCount, tfidf)

vecCount = pickle.load(open('./model/countVec.pkl', "rb"))
tfidf = pickle.load(open('./model/tfidf.pkl', "rb"))


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
