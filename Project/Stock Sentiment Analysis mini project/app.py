
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import bz2file as bz2
import string
from sklearn.feature_extraction.text import CountVectorizer

## Function to decompress the model
def decompress_pickle(file):
  data = bz2.BZ2File(file, 'rb')
  data = pickle.load(data)
  return data

## FUnction to predict the stock movement
def predict_sentiment(headline_string):
  headline_string = headline_string.translate(str.maketrans('', '', string.punctuation))
  headline_string = headline_string.lower()
  
  vect = cv.transform([headline_string])
  my_prediction = model.predict(vect)

  return my_prediction

app = Flask(__name__)

# Extract countvectorizer object and the model
cv = decompress_pickle('senti-cv.pbz2')
model = decompress_pickle('senti-model.pbz2')

# Create a CountVectorizer object
vectorizer = CountVectorizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        headline_names = []
        html_inputs = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',
                       'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20',
                       'h21', 'h22', 'h23', 'h24', 'h25']

        for i in range(len(html_inputs)):
          input_value = request.form[html_inputs[i]]
          headline_names.append(input_value)

        headline_string = ' '.join(x for x in headline_names)

    prediction = predict_sentiment(headline_string)

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
