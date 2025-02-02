import flask
import pickle
import pandas as pd
import numpy as np
import re

# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

# Keras
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.models import load_model


# Model keras mise en place du réseau + load des poids entrainés

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('./model/tensorflow.h5')
model._make_predict_function()


# Import des pickles des models entrainés (vecCount, tfidf)

vecCount = pickle.load(open('./model/countVec.pkl', "rb"))
tfidf = pickle.load(open('./model/tfidf.pkl', "rb"))

# tags du model
tags_columns = ['.net', 'asp.net', 'asp.net-mvc', 'c', 'c#', 'c++', 'css',
                'database','html', 'iphone', 'java', 'javascript', 'jquery',
                'linux', 'mysql','objective-c', 'performance', 'php', 'python',
                'regex', 'ruby','ruby-on-rails', 'sql', 'sql-server', 'vb.net',
                 'visual-studio','visual-studio-2008', 'windows', 'winforms',
                 'wpf', 'xml']

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

        # 1. On enlève les tags HTML
        #question = BeautifulSoup(question, 'lxml').get_text()

        # 2. On enlève les caractères qui ne sont pas des lettres
        question = re.sub("[^a-zA-Z]", " ", question)

        # 3. Conversion en minuscule et split en mots individuels
        question = question.lower().split()

        # 4. Creation du set de stopwords (nltk)
        stops = set(stopwords.words('english'))

        # 5. On enlève ces stop words et on lemmentise
        stemmer = EnglishStemmer()
        question = [stemmer.stem(w) for w in question if not w in stops]

        # 6. On remet tout en un seul string
        question = " ".join(question)

        # Transformation en Pandas Series pour intégrer au model
        df_question = pd.Series([question])

        # Transformation en bag of word ensuite tfidf
        tags = vecCount.transform(df_question)
        tags = tfidf.transform(tags)

        # Prediction des tags
        tags = pd.DataFrame(model.predict(tags))
        tags.columns = tags_columns

        # Tags dans une variable string
        tags_result = ''

        for item in (tags>=0.5).any()[(tags>=0.5).any()].index :
            tags_result = tags_result + '  '  + item

        #print(tags[tags>=0.5].columns.astype(str))

        # Affichage des tags
        return flask.render_template('main.html',
                                     original_input={'question':question},
                                     result=tags_result,
                                     )

if __name__ == '__main__':
    app.run()
