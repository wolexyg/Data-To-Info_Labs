import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import bz2file as bz2
from PIL import Image

## Function to decompress the model
def decompress_pickle(file):
  data = bz2.BZ2File(file, 'rb')
  data = pickle.load(data)
  return data


# Extract countvectorizer object and the model
model = decompress_pickle('model2.pbz2')

## Function to convert image into pixel values
def convert_grayscale(file):
    # Load the image
    image = Image.open(file)

    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))

    # Convert the image to grayscale
    image = image.convert("L")

    # Extract pixel values into 2D array
    pixel_data = np.array(image)

    return pixel_data

## FUnction to predict the stock movement
def predict_digit(grayscale_data):
  my_prediction = model.predict(grayscale_data.reshape(1, 784))
  return my_prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    # convert file into grayscale
    grayscale_data = convert_grayscale(file)

    prediction = predict_digit(grayscale_data)

    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)



